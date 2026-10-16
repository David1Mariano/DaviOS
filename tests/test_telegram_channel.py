"""Testes do TelegramChannel (sem rede, sem sleep real, sem llama-server).

Sincronização via threading.Event / Queue.join com timeout de 5 s. O FakeEngine
cria um sqlite3.connect(":memory:") no __init__ e o usa dentro de process():
se o canal usar o engine fora da thread que o criou, o sqlite levanta
ProgrammingError e o teste QUEBRA (é exatamente o que queremos detectar).
"""

from __future__ import annotations

import dataclasses
import queue
import sqlite3
import threading

import pytest

from brain.channels.base import IncomingMessage
from brain.channels.telegram_channel import (
    BUSY_MESSAGE,
    NO_RESPONSE_MESSAGE,
    PROCESS_ERROR_MESSAGE,
    RESET_MESSAGE,
    START_MESSAGE,
    TelegramChannel,
    UNKNOWN_COMMAND_MESSAGE,
    effective_channel_config,
)
from brain.channels.telegram_client import TelegramAPIError
from config.davios_config import DaviosConfig

ALLOWED = [111]


def _private(update_id, sender, chat, text):
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": sender},
            "chat": {"id": chat, "type": "private"},
            "text": text,
        },
    }


class FakeEngine:
    """Engine falso que usa SQLite :memory: para detectar uso cross-thread."""

    def __init__(self, response="ok", raise_on_process=None, long_response=None):
        self._conn = sqlite3.connect(":memory:")  # criado na thread do caller
        self._conn.execute("CREATE TABLE t(x)")
        self._conn.commit()
        self.response = response
        self.raise_on_process = raise_on_process
        self.long_response = long_response
        self.processed = []           # textos chegados em process()
        self.reset_count = 0
        self.closed = False
        self.process_thread_ids = set()
        self.init_thread_id = threading.get_ident()

    def process(self, text):
        # Toca o SQLite: só funciona na thread que criou a conexão.
        self._conn.execute("INSERT INTO t VALUES (1)")
        self._conn.commit()
        self.processed.append(text)
        self.process_thread_ids.add(threading.get_ident())
        if self.raise_on_process is not None:
            raise self.raise_on_process
        resp = self.long_response if self.long_response is not None else self.response

        class _Result:
            pass

        r = _Result()
        r.response = resp
        r.should_exit = False
        return r

    def reset_context(self):
        self.reset_count += 1

    def close(self):
        # Fecha a conexão SQLite criada nesta thread. Se rodar em outra thread,
        # o sqlite3 levanta ProgrammingError (detecta uso cross-thread).
        self._conn.close()
        self.closed = True


class FakeClient:
    """Cliente falso com semântica realista de ``getUpdates`` (sem rede).

    Modela o contrato documentado da Bot API:
      * updates possuem ``update_id`` crescente e só SOMEM quando confirmados
        por um offset maior;
      * ``get_updates(offset=N>0)`` confirma (descarta) os updates com
        ``update_id < N`` e devolve os restantes;
      * ``get_updates(offset=-1)`` esquece todos menos o último (que continua
        pendente) e devolve o restante;
      * ``get_updates(offset=None)`` devolve TODOS os não confirmados.

    ``backlog`` semeia updates já pendentes antes do ``start()``. ``script``
    entrega novos updates a cada chamada normal (ou levanta a ``Exception`` ali
    contida, consumindo-a). ``discard_effects`` faz as chamadas de descarte
    (``offset=-1``) levantarem erros, para testar o retry fail-closed.
    """

    def __init__(self, script=None, default_updates=None, backlog=None,
                 discard_effects=None):
        self._script = list(script or [])
        self._default_updates = default_updates if default_updates is not None else []
        self._pending = list(backlog or [])  # updates ainda não confirmados
        self._discard_effects = list(discard_effects or [])
        self.sent = []                # (chat_id, text)
        self.actions = []             # (chat_id, action)
        self.get_updates_calls = []   # kwargs de cada chamada
        self._lock = threading.Lock()

    @staticmethod
    def _uid(update):
        v = update.get("update_id") if isinstance(update, dict) else None
        return v if isinstance(v, int) else -1

    def get_updates(self, offset=None, timeout=30):
        with self._lock:
            self.get_updates_calls.append({"offset": offset, "timeout": timeout})
            is_discard = (offset == -1 or timeout == 0)

            if is_discard:
                # efeito roteirizado (erro) no descarte, se houver
                if self._discard_effects:
                    effect = self._discard_effects.pop(0)
                    if isinstance(effect, Exception):
                        raise effect
                # esquece todos menos o último (que continua pendente)
                if self._pending:
                    self._pending = [max(self._pending, key=self._uid)]
                return list(self._pending)

            # chamada normal: escolhe o lote ANTES de mutar (falha => nada muda)
            if self._script:
                item = self._script.pop(0)
            else:
                item = self._default_updates
            if isinstance(item, Exception):
                raise item

            # confirma por offset (descarta update_id < offset)
            if offset is not None and offset > 0:
                self._pending = [
                    u for u in self._pending
                    if not (self._uid(u) >= 0 and self._uid(u) < offset)
                ]
            if item:
                self._pending.extend(item)
            return list(self._pending)

    def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))

    def send_chat_action(self, chat_id, action="typing"):
        self.actions.append((chat_id, action))


class RaisingSendClient(FakeClient):
    """``send_message`` SEMPRE levanta ``error`` (conta tentativas em send_count).

    Usa a semântica normal de ``get_updates`` (herdada); só o envio falha.
    """

    def __init__(self, error, **kw):
        super().__init__(**kw)
        self._error = error
        self.send_count = 0

    def send_message(self, chat_id, text):
        self.send_count += 1
        raise self._error


class SeqSendErrorClient(FakeClient):
    """``send_message`` levanta os erros de ``errors`` em ordem; depois, sucesso.

    Registra apenas envios BEM-sucedidos em ``sent`` (como o FakeClient).
    """

    def __init__(self, errors, **kw):
        super().__init__(**kw)
        self._errors = list(errors or [])

    def send_message(self, chat_id, text):
        if self._errors:
            raise self._errors.pop(0)
        self.sent.append((chat_id, text))


class BadActionClient(FakeClient):
    """``send_chat_action`` SEMPRE levanta ``error`` (após registrar a action)."""

    def __init__(self, error, **kw):
        super().__init__(**kw)
        self._error = error

    def send_chat_action(self, chat_id, action="typing"):
        self.actions.append((chat_id, action))
        raise self._error


class FakeSleep:
    """Relógio falso: registra as esperas sem dormir de verdade."""

    def __init__(self):
        self.waits = []
        self._event = threading.Event()

    def __call__(self, seconds):
        self.waits.append(seconds)

    def wake(self):
        self._event.set()


class FakeClock:
    """Relógio injetável para ``TelegramChannel(now=...)`` (avança sob demanda)."""

    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt



def _wait_for(predicate, timeout=5.0):
    """Espera até predicate() ser verdadeiro (ou estourar o timeout)."""
    import time as _time

    end = _time.time() + timeout
    while _time.time() < end:
        if predicate():
            return True
        _time.sleep(0.01)
    return predicate()


@pytest.fixture
def channel_factory():
    """Fabrica um canal com fakes. Guarda engines criados em _test_holder.

    ``engine_kwargs`` são passados ao FakeEngine, que é construído DENTRO da
    factory (ou seja, na thread worker) — nunca passe um engine pré-construído,
    pois o SQLite :memory: do FakeEngine se prende à thread que o criou.
    """
    created = {"engines": []}

    def _make(allowed_user_ids=ALLOWED, queue_max=20, chunk_limit=4000,
              poll_timeout=30, engine_kwargs=None, client=None,
              engine_factory=None, now=None, engine_cleanup=None):
        client = client if client is not None else FakeClient()
        holder = {"engine": None}
        engine_kwargs = engine_kwargs or {}

        def factory():
            eng = FakeEngine(**engine_kwargs)
            holder["engine"] = eng
            created["engines"].append(eng)
            return eng

        if engine_factory is not None:
            factory = engine_factory

        extra = {}
        if engine_cleanup is not None:
            extra["engine_cleanup"] = engine_cleanup
        ch = TelegramChannel(
            client=client,
            engine_factory=factory,
            allowed_user_ids=allowed_user_ids,
            poll_timeout=poll_timeout,
            queue_max=queue_max,
            chunk_limit=chunk_limit,
            sleep=FakeSleep(),
            now=now,
            **extra,
        )
        ch._test_holder = holder  # type: ignore[attr-defined]
        ch._test_engines = created["engines"]  # type: ignore[attr-defined]
        return ch

    return _make



# --------------------------------------------------------------------------- #
# Engine criado e usado na MESMA thread worker
# --------------------------------------------------------------------------- #
class TestEngineThreadIsolation:
    def test_engine_built_and_used_in_worker_thread(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 111, "oi")]])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: bool(ch._test_holder["engine"]
                                          and ch._test_holder["engine"].processed))
        finally:
            ch.stop()
            ch.join(2.0)
        engine = ch._test_holder["engine"]
        # process() rodou (tocou SQLite sem ProgrammingError) e tudo na worker.
        assert engine.processed == ["oi"]
        assert engine.init_thread_id in engine.process_thread_ids

    def test_poller_never_touches_engine(self, channel_factory):
        # Muitas mensagens: o poller só enfileira; process() ocorre só no worker.
        ups = [_private(i, 111, 111, "msg%d" % i) for i in range(1, 6)]
        client = FakeClient(script=[ups])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(
                lambda: ch._test_holder["engine"]
                and len(ch._test_holder["engine"].processed) == 5
            )
        finally:
            ch.stop()
            ch.join(2.0)
        engine = ch._test_holder["engine"]
        # Todas as chamadas a process() ocorreram em uma única thread.
        assert len(engine.process_thread_ids) == 1


# --------------------------------------------------------------------------- #
# Não autorizado
# --------------------------------------------------------------------------- #
class TestUnauthorized:
    def test_unauthorized_is_silently_dropped(self, channel_factory, caplog):
        import logging

        ups = [_private(1, 999, 999, "texto secreto proibido")]
        client = FakeClient(script=[ups])
        ch = channel_factory(client=client)  # allowed = [111]
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: client.get_updates_calls)
                # Dá tempo do poller processar o update; o worker NÃO deve agir.
                import time as _t

                _t.sleep(0.2)
            finally:
                ch.stop()
                ch.join(2.0)
        # Zero process/send/action/enqueue.
        assert client.sent == []
        assert client.actions == []
        assert ch._queue.qsize() == 0
        # Log tem sender_id, mas NÃO o texto.
        assert "999" in caplog.text
        assert "texto secreto proibido" not in caplog.text

    def test_unauthorized_throttled_to_once_per_10min(self, channel_factory, caplog):
        import logging

        # 3 updates do MESMO sender não autorizado, em polls distintos.
        client = FakeClient(
            script=[
                [_private(1, 999, 999, "a")],
                [_private(2, 999, 999, "b")],
                [_private(3, 999, 999, "c")],
                [],  # fica devolvendo vazio depois
            ]
        )
        ch = channel_factory(client=client)
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: len(client.get_updates_calls) >= 4)
            finally:
                ch.stop()
                ch.join(2.0)
        # Logado só UMA vez apesar de 3 mensagens (throttle 10 min).
        occurrences = caplog.text.count("remetente nao autorizado sender_id=999")
        assert occurrences == 1


# --------------------------------------------------------------------------- #
# Fluxo autorizado básico
# --------------------------------------------------------------------------- #
class TestAuthorizedFlow:
    def test_response_sent_to_chat_id(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 555, "oi")]])
        ch = channel_factory(client=client, engine_kwargs={"response": "ola!"})
        ch.start()
        try:
            assert _wait_for(lambda: any(t == "ola!" for _, t in client.sent))
        finally:
            ch.stop()
            ch.join(2.0)
        assert (555, "ola!") in client.sent
        # enviou indicador "typing" antes.
        assert (555, "typing") in client.actions

    def test_long_response_chunked_in_order(self, channel_factory):
        long_resp = " ".join("bloco%d" % i for i in range(2000))  # > 4000 chars
        client = FakeClient(script=[[_private(1, 111, 555, "fala muito")]])
        ch = channel_factory(client=client, engine_kwargs={"response": long_resp})
        ch.start()
        try:
            assert _wait_for(lambda: len(client.sent) >= 3)
        finally:
            ch.stop()
            ch.join(2.0)
        texts = [t for _, t in client.sent]
        assert len(texts) >= 3
        assert all(len(t) <= 4000 for t in texts)
        assert "".join(texts) == long_resp  # ordem preservada

    def test_empty_response_gets_fixed_message(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 555, "oi")]])
        ch = channel_factory(client=client, engine_kwargs={"response": "   "})
        ch.start()
        try:
            assert _wait_for(lambda: client.sent)
        finally:
            ch.stop()
            ch.join(2.0)
        assert client.sent[-1] == (555, NO_RESPONSE_MESSAGE)


# --------------------------------------------------------------------------- #
# Comandos e resiliência do worker
# --------------------------------------------------------------------------- #
class TestCommands:
    def test_start_command(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 555, "/start")]])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: any(t == START_MESSAGE for _, t in client.sent))
        finally:
            ch.stop()
            ch.join(2.0)
        # /start não vai para o LLM.
        assert ch._test_holder["engine"].processed == []

    def test_reset_command_calls_reset_in_worker(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 555, "/reset")]])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: any(t == RESET_MESSAGE for _, t in client.sent))
        finally:
            ch.stop()
            ch.join(2.0)
        engine = ch._test_holder["engine"]
        assert engine.reset_count == 1
        assert engine.processed == []  # reset não passa pelo process()

    def test_unknown_command(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 555, "/qualquercoisa")]])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(
                lambda: any(t == UNKNOWN_COMMAND_MESSAGE for _, t in client.sent)
            )
        finally:
            ch.stop()
            ch.join(2.0)
        assert ch._test_holder["engine"].processed == []

    def test_command_with_botname_and_case_normalized(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 555, "/Reset@MeuBot")]])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: any(t == RESET_MESSAGE for _, t in client.sent))
        finally:
            ch.stop()
            ch.join(2.0)
        assert ch._test_holder["engine"].reset_count == 1

    def test_too_long_message_rejected(self, channel_factory):
        big = "x" * 2001
        client = FakeClient(script=[[_private(1, 111, 555, big)]])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: client.sent)
        finally:
            ch.stop()
            ch.join(2.0)
        # Mensagem longa NÃO vai para o LLM.
        assert ch._test_holder["engine"].processed == []
        assert "muito longa" in client.sent[-1][1]


class TestWorkerResilience:
    def test_process_exception_does_not_kill_channel(self, channel_factory):
        # 1ª msg levanta; 2ª deve funcionar (canal continua vivo).
        client = FakeClient(
            script=[
                [_private(1, 111, 555, "boom")],
                [_private(2, 111, 555, "ok")],
            ]
        )
        ch = channel_factory(
            client=client,
            engine_kwargs={"raise_on_process": RuntimeError("falha interna")},
        )
        ch.start()
        try:
            # A 1ª responde a msg de erro; a 2ª... mas o engine sempre levanta.
            # Para provar "continua vivo", checamos que a msg de erro foi enviada
            # e o worker segue de pé (is_running).
            assert _wait_for(
                lambda: any(t == PROCESS_ERROR_MESSAGE for _, t in client.sent)
            )
            assert ch.is_running is True
        finally:
            ch.stop()
            ch.join(2.0)

    def test_process_exception_then_success_next_message(self, channel_factory):
        # Engine que levanta só na 1ª chamada.
        class FlakyEngine(FakeEngine):
            def __init__(self):
                super().__init__()
                self._calls = 0

            def process(self, text):
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("falha interna")
                return super().process(text)

        client = FakeClient(
            script=[
                [_private(1, 111, 555, "boom")],
                [_private(2, 111, 555, "ok")],
            ]
        )
        ch = channel_factory(client=client, engine_factory=FlakyEngine)
        ch.start()
        try:
            assert _wait_for(
                lambda: any(t == PROCESS_ERROR_MESSAGE for _, t in client.sent)
                and any(t == "ok" for _, t in client.sent)
            )
        finally:
            ch.stop()
            ch.join(2.0)
        assert any(t == PROCESS_ERROR_MESSAGE for _, t in client.sent)
        assert any(t == "ok" for _, t in client.sent)

    def test_should_exit_does_not_stop_channel(self, channel_factory):
        class ExitEngine(FakeEngine):
            def process(self, text):
                # toca SQLite e marca should_exit=True
                self._conn.execute("INSERT INTO t VALUES (1)")
                self._conn.commit()
                self.processed.append(text)

                class _R:
                    pass

                r = _R()
                r.response = "tchau"
                r.should_exit = True
                return r

        client = FakeClient(script=[[_private(1, 111, 555, "tchau")]])
        ch = channel_factory(client=client, engine_factory=ExitEngine)
        ch.start()
        try:
            assert _wait_for(lambda: any(t == "tchau" for _, t in client.sent))
            # D4: should_exit NUNCA encerra o canal.
            assert ch.is_running is True
        finally:
            ch.stop()
            ch.join(2.0)



# --------------------------------------------------------------------------- #
# Poller: backlog, offset, fila cheia
# --------------------------------------------------------------------------- #
class TestPollerBehavior:
    def test_backlog_discarded_at_start(self, channel_factory):
        # Mensagem antiga no backlog NÃO deve ser processada.
        backlog = [_private(1, 111, 555, "comando antigo")]
        client = FakeClient(backlog=backlog, script=[[]])
        ch = channel_factory(client=client)
        ch.start()
        try:
            # Dá tempo do poller rodar; o worker não deve processar o backlog.
            import time as _t

            _t.sleep(0.3)
        finally:
            ch.stop()
            ch.join(2.0)
        # A primeira chamada é o descarte de backlog (offset=-1, timeout=0).
        assert client.get_updates_calls[0]["offset"] == -1
        assert client.get_updates_calls[0]["timeout"] == 0
        assert ch._test_holder["engine"].processed == []  # nada processado

    def test_offset_advances_after_enqueue(self, channel_factory):
        client = FakeClient(script=[[_private(7, 111, 555, "oi")], []])
        ch = channel_factory(client=client)
        ch.start()
        try:
            # Após processar update_id=7, o próximo poll usa offset=8.
            assert _wait_for(
                lambda: any(c["offset"] == 8 for c in client.get_updates_calls)
            )
        finally:
            ch.stop()
            ch.join(2.0)

    def test_queue_full_replies_busy(self, channel_factory):
        # queue_max=1 e o worker "lento": enchemos a fila e forçamos o ocupado.
        import time as _time

        class SlowEngine(FakeEngine):
            def process(self, text):
                _time.sleep(0.3)  # segura o worker
                return super().process(text)

        # 3 mensagens de uma vez; fila=1 => o poller deve responder "ocupado".
        ups = [
            _private(1, 111, 555, "m1"),
            _private(2, 111, 555, "m2"),
            _private(3, 111, 555, "m3"),
        ]
        client = FakeClient(script=[ups])
        ch = channel_factory(client=client, queue_max=1, engine_factory=SlowEngine)
        ch.start()
        try:
            assert _wait_for(
                lambda: any(t == BUSY_MESSAGE for _, t in client.sent), timeout=5.0
            )
        finally:
            ch.stop()
            ch.join(2.0)


# --------------------------------------------------------------------------- #
# Poller: tratamento de erros (401/409/429/backoff)
# --------------------------------------------------------------------------- #
class TestPollerErrors:
    def test_401_is_fatal(self, channel_factory, caplog):
        import logging

        client = FakeClient(
            script=[TelegramAPIError("Unauthorized", error_code=401)]
        )
        ch = channel_factory(client=client)
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: ch.is_failed)
            finally:
                ch.join(2.0)
        assert ch.is_failed is True
        assert "token invalido" in caplog.text

    def test_409_waits_30s(self, channel_factory):
        client = FakeClient(
            script=[TelegramAPIError("Conflict", error_code=409), []]
        )
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: 30.0 in ch._sleep.waits)
        finally:
            ch.stop()
            ch.join(2.0)
        assert 30.0 in ch._sleep.waits

    def test_429_waits_retry_after(self, channel_factory):
        client = FakeClient(
            script=[
                TelegramAPIError("Too Many", error_code=429, retry_after=12),
                [],
            ]
        )
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: 12.0 in ch._sleep.waits)
        finally:
            ch.stop()
            ch.join(2.0)
        assert 12.0 in ch._sleep.waits

    def test_generic_error_exponential_backoff_then_reset(self, channel_factory):
        # 4 erros genéricos (backoff 1,2,4,8) e depois sucesso.
        err = TelegramAPIError("network down", error_code=500)
        client = FakeClient(
            script=[err, err, err, err, []]
        )
        ch = channel_factory(client=client)
        ch.start()
        try:
            # Espera backoffs 1,2,4 aparecerem (em infectado FakeSleep).
            assert _wait_for(
                lambda: all(w in ch._sleep.waits for w in (1.0, 2.0, 4.0))
            )
        finally:
            ch.stop()
            ch.join(2.0)
        waits = ch._sleep.waits
        assert 1.0 in waits and 2.0 in waits and 4.0 in waits

    def test_backoff_sequence_main_loop(self, channel_factory):
        # 8 erros genéricos seguidos -> esperas EXATAS (teto em 60).
        err = TelegramAPIError("network down", error_code=500)
        client = FakeClient(script=[err] * 8 + [[]])  # 8 erros, depois sucesso
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: len(ch._sleep.waits) >= 8)
        finally:
            ch.stop()
            ch.join(2.0)
        assert ch._sleep.waits == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]

    def test_backoff_resets_after_success(self, channel_factory):
        # erro, sucesso, erro -> após o sucesso o backoff volta a 1.0.
        err = TelegramAPIError("network down", error_code=500)
        client = FakeClient(script=[err, [], err, []])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: len(ch._sleep.waits) >= 2)
        finally:
            ch.stop()
            ch.join(2.0)
        assert ch._sleep.waits[:2] == [1.0, 1.0]

    def test_backoff_sequence_discard_loop(self, channel_factory):
        # MESMA sequência exata no laço de descarte (_startup_backlog_offset).
        err = TelegramAPIError("network down", error_code=500)
        client = FakeClient(discard_effects=[err] * 8, script=[[]])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(lambda: len(ch._sleep.waits) >= 8)
        finally:
            ch.stop()
            ch.join(2.0)
        assert ch._sleep.waits == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]



# --------------------------------------------------------------------------- #
# Ciclo de vida: start/stop, start duplo, engine_factory que falha
# --------------------------------------------------------------------------- #
class TestLifecycle:
    def test_start_does_not_block(self, channel_factory):
        ch = channel_factory(client=FakeClient(default_updates=[]))
        done = threading.Event()

        def _run():
            ch.start()
            done.set()

        t = threading.Thread(target=_run)
        t.start()
        assert done.wait(2.0)  # start() retorna rápido (threads daemon)
        t.join(1.0)
        ch.stop()
        ch.join(2.0)

    def test_double_start_raises(self, channel_factory):
        ch = channel_factory(client=FakeClient(default_updates=[]))
        ch.start()
        try:
            with pytest.raises(RuntimeError):
                ch.start()
        finally:
            ch.stop()
            ch.join(2.0)

    def test_stop_is_idempotent_and_threads_terminate(self, channel_factory):
        ch = channel_factory(client=FakeClient(default_updates=[]))
        ch.start()
        ch.stop()
        ch.stop()  # idempotente
        ch.join(3.0)
        assert all(not t.is_alive() for t in ch._threads)
        assert ch.is_running is False

    def test_engine_factory_failure_puts_channel_in_failed_state(
        self, channel_factory, caplog
    ):
        import logging

        def bad_factory():
            raise RuntimeError("nao deu para carregar o modelo")

        ch = channel_factory(
            client=FakeClient(default_updates=[]), engine_factory=bad_factory
        )
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: ch.is_failed)
            finally:
                ch.join(2.0)
        assert ch.is_failed is True
        assert ch.is_running is False
        # Erro redigido: só o tipo, não a mensagem interna.
        assert "RuntimeError" in caplog.text
        assert "nao deu para carregar o modelo" not in caplog.text
        # Threads terminaram (nada pendurado).
        assert all(not t.is_alive() for t in ch._threads)


# --------------------------------------------------------------------------- #
# effective_channel_config (D3)
# --------------------------------------------------------------------------- #
class TestEffectiveChannelConfig:
    def test_flag_off_disables_three_capabilities(self):
        cfg = DaviosConfig(
            tools_visible_to_llm=True,
            actions_enabled=True,
            web_enabled=True,
            telegram_tools_enabled=False,
        )
        eff = effective_channel_config(cfg)
        assert eff.tools_visible_to_llm is False
        assert eff.actions_enabled is False
        assert eff.web_enabled is False

    def test_flag_on_keeps_config(self):
        cfg = DaviosConfig(
            tools_visible_to_llm=True,
            actions_enabled=True,
            web_enabled=False,
            telegram_tools_enabled=True,
        )
        eff = effective_channel_config(cfg)
        assert eff.tools_visible_to_llm is True
        assert eff.actions_enabled is True

    def test_original_not_mutated(self):
        cfg = DaviosConfig(
            tools_visible_to_llm=True,
            actions_enabled=True,
            web_enabled=True,
            telegram_tools_enabled=False,
        )
        _ = effective_channel_config(cfg)
        # A original permanece intacta.
        assert cfg.tools_visible_to_llm is True
        assert cfg.actions_enabled is True
        assert cfg.web_enabled is True

    def test_returns_copy_not_same_object_when_off(self):
        cfg = DaviosConfig(telegram_tools_enabled=False)
        eff = effective_channel_config(cfg)
        assert eff is not cfg



# --------------------------------------------------------------------------- #
# Privacidade de logs (transversal): token, URL e texto de mensagem
# --------------------------------------------------------------------------- #
SECRET_TEXT = "minha senha do banco eh 1234"
FAKE_TOKEN = "123456:ABCDEFghij-KLMNOP_qrstuvw01"


class TestLogPrivacy:
    def test_no_token_url_or_message_text_in_logs(self, channel_factory, caplog):
        import logging

        # Mistura: autorizado (texto sensível), não autorizado, comando, erro.
        client = FakeClient(
            script=[
                [
                    _private(1, 111, 555, SECRET_TEXT),
                    _private(2, 999, 999, "nao autorizado sensivel"),
                    _private(3, 111, 555, "/start"),
                ],
                [],
            ]
        )
        ch = channel_factory(client=client, engine_kwargs={"response": "resposta secreta"})
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: client.sent)
                import time as _t

                _t.sleep(0.2)
            finally:
                ch.stop()
                ch.join(2.0)
        logs = caplog.text
        # Nenhum texto de mensagem do usuário.
        assert SECRET_TEXT not in logs
        assert "nao autorizado sensivel" not in logs
        # Nenhuma resposta do assistente.
        assert "resposta secreta" not in logs
        # Nenhum token e nenhuma URL de API.
        assert FAKE_TOKEN not in logs
        assert "api.telegram.org" not in logs

    def test_message_text_never_logged_on_process_error(
        self, channel_factory, caplog
    ):
        import logging

        client = FakeClient(script=[[_private(1, 111, 555, SECRET_TEXT)]])
        ch = channel_factory(
            client=client,
            engine_kwargs={"raise_on_process": RuntimeError("segredo interno")},
        )
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(
                    lambda: any(
                        t == PROCESS_ERROR_MESSAGE for _, t in client.sent
                    )
                )
            finally:
                ch.stop()
                ch.join(2.0)
        assert SECRET_TEXT not in caplog.text
        assert "segredo interno" not in caplog.text


# --------------------------------------------------------------------------- #
# T4b — Backlog realista, fail-closed, robustez do poller, _fail e throttle
# --------------------------------------------------------------------------- #
class TestBacklogRealista:
    """Item 1: o descarte de backlog precisa virar offset do loop principal."""

    def test_backlog_not_reprocessed_only_new_message_processed(self, channel_factory):
        # 3 updates pendentes ANTES do start + 1 nova DEPOIS:
        # só a nova pode ser processada (as antigas são descartadas de verdade).
        backlog = [
            _private(1, 111, 555, "antiga-1"),
            _private(2, 111, 555, "antiga-2"),
            _private(3, 111, 555, "antiga-3"),
        ]
        nova = _private(4, 111, 555, "NOVA")
        client = FakeClient(backlog=backlog, script=[[nova]])
        ch = channel_factory(client=client)
        ch.start()
        try:
            assert _wait_for(
                lambda: ch._test_holder["engine"]
                and ch._test_holder["engine"].processed
            )
            import time as _t

            _t.sleep(0.2)  # dá tempo de eventuais mensagens antigas vazarem
        finally:
            ch.stop()
            ch.join(2.0)
        processed = ch._test_holder["engine"].processed
        # SOMENTE a nova; nenhuma das 3 antigas.
        assert processed == ["NOVA"]

    def test_discard_returns_initial_offset(self, channel_factory):
        backlog = [_private(1, 111, 555, "a"), _private(2, 111, 555, "b")]
        client = FakeClient(backlog=backlog, script=[[]])
        ch = channel_factory(client=client)
        # _discard_backlog devolve max(update_id)+1 (ou None se vazio).
        off = ch._discard_backlog()
        assert off == 3

    def test_discard_returns_none_when_empty(self, channel_factory):
        client = FakeClient(script=[[]])
        ch = channel_factory(client=client)
        assert ch._discard_backlog() is None


class TestBacklogFailClosed:
    """Item 2: erros no descarte usam a MESMA política e são repetidos até dar certo."""

    def test_discard_401_is_fatal_and_no_dispatch(self, channel_factory, caplog):
        import logging

        client = FakeClient(
            discard_effects=[TelegramAPIError("Unauthorized", error_code=401)],
            backlog=[_private(1, 111, 555, "antiga")],
            script=[[_private(2, 111, 555, "NOVA")]],
        )
        ch = channel_factory(client=client)
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: ch.is_failed)
            finally:
                ch.join(2.0)
        assert ch.is_failed is True
        assert ch.is_running is False
        # fail-closed: nada foi despachado/processado.
        assert client.sent == []
        eng = ch._test_holder["engine"]
        assert eng is None or eng.processed == []
        assert "token invalido" in caplog.text  # 401 durante o descarte

    def test_discard_409_waits_30s_then_retries_and_succeeds(self, channel_factory):
        client = FakeClient(
            discard_effects=[TelegramAPIError("Conflict", error_code=409)],
            script=[[_private(1, 111, 555, "oi")]],
        )
        ch = channel_factory(client=client, engine_kwargs={"response": "ola"})
        ch.start()
        try:
            assert _wait_for(lambda: 30.0 in ch._sleep.waits)
            assert _wait_for(lambda: any(t == "ola" for _, t in client.sent))
        finally:
            ch.stop()
            ch.join(2.0)
        assert 30.0 in ch._sleep.waits
        assert ch.is_failed is False

    def test_discard_429_waits_retry_after_then_succeeds(self, channel_factory):
        client = FakeClient(
            discard_effects=[TelegramAPIError("Too Many", error_code=429, retry_after=12)],
            script=[[_private(1, 111, 555, "oi")]],
        )
        ch = channel_factory(client=client, engine_kwargs={"response": "ola"})
        ch.start()
        try:
            assert _wait_for(lambda: 12.0 in ch._sleep.waits)
            assert _wait_for(lambda: any(t == "ola" for _, t in client.sent))
        finally:
            ch.stop()
            ch.join(2.0)
        assert 12.0 in ch._sleep.waits
        assert ch.is_failed is False

    def test_discard_generic_error_backoff_1_2_4_then_succeeds(self, channel_factory):
        err = TelegramAPIError("network down", error_code=500)
        client = FakeClient(
            discard_effects=[err, err, err],
            script=[[_private(1, 111, 555, "oi")]],
        )
        ch = channel_factory(client=client, engine_kwargs={"response": "ola"})
        ch.start()
        try:
            assert _wait_for(
                lambda: all(w in ch._sleep.waits for w in (1.0, 2.0, 4.0))
            )
            assert _wait_for(lambda: any(t == "ola" for _, t in client.sent))
        finally:
            ch.stop()
            ch.join(2.0)
        assert all(w in ch._sleep.waits for w in (1.0, 2.0, 4.0))
        assert ch.is_failed is False

    def test_discard_no_dispatch_until_discard_succeeds(self, channel_factory):
        # fail-closed: enquanto o descarte falha, NADA é processado.
        err = TelegramAPIError("down", error_code=500)
        client = FakeClient(
            discard_effects=[err, err],
            script=[[_private(1, 111, 555, "primeira")]],
        )
        ch = channel_factory(client=client)
        ch.start()
        try:
            # espera os backoffs 1 e 2 do descarte
            assert _wait_for(lambda: all(w in ch._sleep.waits for w in (1.0, 2.0)))
            # a mensagem só é processada após o descarte dar certo
            assert _wait_for(
                lambda: ch._test_holder["engine"]
                and ch._test_holder["engine"].processed
            )
        finally:
            ch.stop()
            ch.join(2.0)
        # a única mensagem veio do script (nova), nunca backlog
        assert ch._test_holder["engine"].processed == ["primeira"]


class TestPollerRobustness:
    """Item 3: exceção inesperada => loga o TIPO, falha, e is_running exige threads vivas."""

    def test_runtime_error_in_get_updates_fails_channel_without_hang(
        self, channel_factory, caplog
    ):
        import logging

        client = FakeClient(script=[RuntimeError("boom interno")])
        ch = channel_factory(client=client)
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: ch.is_failed)
                assert ch.is_running is False  # failed + threads
            finally:
                ch.join(2.0)  # não pendura
        assert ch.is_failed is True
        assert ch.is_running is False
        # loga o TIPO da exceção (não a mensagem interna).
        assert "RuntimeError" in caplog.text
        assert "boom interno" not in caplog.text
        # threads terminaram (nada pendurado).
        assert all(not t.is_alive() for t in ch._threads)

    def test_runtime_error_during_discard_fails_channel(self, channel_factory, caplog):
        import logging

        client = FakeClient(
            discard_effects=[RuntimeError("boom discard")],
            script=[[_private(1, 111, 555, "NOVA")]],
        )
        ch = channel_factory(client=client)
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: ch.is_failed)
            finally:
                ch.join(2.0)
        assert ch.is_failed is True
        assert ch.is_running is False
        assert "RuntimeError" in caplog.text
        assert "boom discard" not in caplog.text
        # fail-closed: a mensagem do script nunca foi processada.
        eng = ch._test_holder["engine"]
        assert eng is None or eng.processed == []
        assert all(not t.is_alive() for t in ch._threads)

    def test_is_running_requires_threads_alive(self, channel_factory):
        ch = channel_factory(client=FakeClient(default_updates=[]))
        assert ch.is_running is False  # ainda não iniciado
        ch.start()
        try:
            assert ch.is_running is True
        finally:
            ch.stop()
            ch.join(3.0)
        # após stop+join as threads morreram -> False
        assert ch.is_running is False
        assert all(not t.is_alive() for t in ch._threads)


class TestFailLogsReason:
    """Item 4: _fail(reason) registra o motivo (visível via caplog)."""

    def test_fail_logs_reason_for_401_during_discard(self, channel_factory, caplog):
        import logging

        client = FakeClient(
            discard_effects=[TelegramAPIError("Unauthorized", error_code=401)]
        )
        ch = channel_factory(client=client)
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: ch.is_failed)
            finally:
                ch.join(2.0)
        assert ch.is_failed is True
        # _fail registra o motivo lido pelo caplog.
        assert "canal falhou: token invalido" in caplog.text

    def test_fail_logs_reason_for_engine_factory_failure(self, channel_factory, caplog):
        import logging

        def bad_factory():
            raise RuntimeError("nao deu para carregar o modelo")

        ch = channel_factory(
            client=FakeClient(default_updates=[]), engine_factory=bad_factory
        )
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: ch.is_failed)
            finally:
                ch.join(2.0)
        assert "canal falhou: engine_factory falhou" in caplog.text
        # não vaza a mensagem interna do erro.
        assert "nao deu para carregar o modelo" not in caplog.text


class TestUnauthorizedThrottleLimit:
    """Item 5: throttle com teto (256) e relógio injetável (expira em 10 min)."""

    def test_repeated_sender_logs_once_within_window(self, channel_factory, caplog):
        import logging

        clock = FakeClock()
        ch = channel_factory(now=clock)
        with caplog.at_level(logging.DEBUG):
            for _ in range(5):
                ch._log_unauthorized(999)
        occurrences = caplog.text.count("remetente nao autorizado sender_id=999")
        assert occurrences == 1

    def test_expires_after_10_minutes(self, channel_factory, caplog):
        import logging

        clock = FakeClock()
        ch = channel_factory(now=clock)
        with caplog.at_level(logging.DEBUG):
            ch._log_unauthorized(999)          # 1o log (t=0)
            clock.advance(599.0)               # ainda dentro da janela
            ch._log_unauthorized(999)           # suprimido
            clock.advance(2.0)                 # total 601s >= 600s -> expirou
            ch._log_unauthorized(999)           # 2o log
        occurrences = caplog.text.count("remetente nao autorizado sender_id=999")
        assert occurrences == 2

    def test_dict_capped_at_256_entries(self, channel_factory):
        from brain.channels.telegram_channel import UNAUTHORIZED_LOG_MAX_ENTRIES

        clock = FakeClock()
        ch = channel_factory(now=clock)
        for sid in range(1000):  # muito mais que o teto
            ch._log_unauthorized(sid)
        assert len(ch._unauth_last_logged) <= UNAUTHORIZED_LOG_MAX_ENTRIES
        assert UNAUTHORIZED_LOG_MAX_ENTRIES == 256

    def test_cap_evicts_oldest_when_full_and_not_expired(self, channel_factory):
        from brain.channels.telegram_channel import UNAUTHORIZED_LOG_MAX_ENTRIES

        clock = FakeClock()
        ch = channel_factory(now=clock)
        # enche até o teto com senders distintos (nenhum expira: relógio parado)
        for sid in range(UNAUTHORIZED_LOG_MAX_ENTRIES):
            ch._log_unauthorized(sid)
        assert len(ch._unauth_last_logged) == UNAUTHORIZED_LOG_MAX_ENTRIES
        # um novo sender estoura -> remove o mais antigo, mantém o teto
        ch._log_unauthorized(999999)
        assert len(ch._unauth_last_logged) == UNAUTHORIZED_LOG_MAX_ENTRIES
        assert 999999 in ch._unauth_last_logged
        assert 0 not in ch._unauth_last_logged  # o mais antigo foi removido




# --------------------------------------------------------------------------- #
# T4c-1: 429 no ENVIO (_send_with_retry): teto 30 s, UMA nova tentativa, sem
#        levantar; só loga o error_code.
# --------------------------------------------------------------------------- #
class TestSendRetryOn429:
    def _chan(self, channel_factory, retry_after):
        err = TelegramAPIError("Too Many", error_code=429, retry_after=retry_after)
        client = RaisingSendClient(err)  # send_message SEMPRE levanta 429
        return channel_factory(client=client), client

    def test_send_retry_max_seconds_is_30(self, channel_factory):
        from brain.channels.telegram_channel import SEND_RETRY_MAX_SECONDS
        assert SEND_RETRY_MAX_SECONDS == 30.0

    def test_retry_after_99_capped_at_30(self, channel_factory):
        ch, client = self._chan(channel_factory, 99)
        ch._send_with_retry(555, "oi")  # não deve levantar
        assert ch._sleep.waits == [30.0]

    def test_retry_after_5_uses_5(self, channel_factory):
        ch, client = self._chan(channel_factory, 5)
        ch._send_with_retry(555, "oi")
        assert ch._sleep.waits == [5.0]

    def test_exactly_two_send_attempts(self, channel_factory):
        ch, client = self._chan(channel_factory, 5)
        ch._send_with_retry(555, "oi")
        assert client.send_count == 2

    def test_second_failure_does_not_raise(self, channel_factory):
        ch, client = self._chan(channel_factory, 5)
        # A segunda falha é engolida (só loga o código); não propaga.
        ch._send_with_retry(555, "oi")
        assert client.send_count == 2


# --------------------------------------------------------------------------- #
# T4c-2: _reply_safe (aviso "ocupado" do poller) NUNCA dorme: UMA tentativa,
#        sem retry e sem sleep, engolindo qualquer exceção.
# --------------------------------------------------------------------------- #
class TestReplySafeNeverSleeps:
    def test_busy_notice_never_sleeps_or_retries(self, channel_factory):
        err = TelegramAPIError("Conflict", error_code=429, retry_after=30)
        client = RaisingSendClient(err)
        ch = channel_factory(client=client)
        ch._reply_safe(555, BUSY_MESSAGE)  # não deve levantar
        assert ch._sleep.waits == []       # poller NUNCA dorme no envio
        assert client.send_count == 1      # UMA única tentativa

    def test_poller_keeps_consuming_when_busy_send_429(self, channel_factory):
        import time as _time

        class SlowEngine(FakeEngine):
            def process(self, text):
                _time.sleep(0.3)
                return super().process(text)

        ups = [_private(i, 111, 555, "m%d" % i) for i in range(1, 4)]
        client = RaisingSendClient(
            TelegramAPIError("Conflict", error_code=429, retry_after=30),
            script=[ups, [], []],
        )
        ch = channel_factory(client=client, queue_max=1, engine_factory=SlowEngine)
        ch.start()
        try:
            # o poller continua chamando get_updates (não trava nem dorme no envio)
            assert _wait_for(lambda: len(client.get_updates_calls) >= 3)
        finally:
            ch.stop()
            ch.join(2.0)
        assert ch.is_failed is False


# --------------------------------------------------------------------------- #
# T4c-4: /reset protegido — reset_context() em try/except; loga só o tipo e
#        responde PROCESS_ERROR_MESSAGE.
# --------------------------------------------------------------------------- #
class TestResetCommandGuarded:
    def test_reset_exception_replies_process_error(self, channel_factory, caplog):
        import logging

        class ResetFailEngine(FakeEngine):
            def reset_context(self):
                raise RuntimeError("nao deu para limpar o contexto")

        client = FakeClient(script=[[_private(1, 111, 555, "/reset")]])
        ch = channel_factory(client=client, engine_factory=ResetFailEngine)
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(
                    lambda: any(t == PROCESS_ERROR_MESSAGE for _, t in client.sent)
                )
            finally:
                ch.stop()
                ch.join(2.0)
        assert any(t == PROCESS_ERROR_MESSAGE for _, t in client.sent)
        # loga só o TIPO; a mensagem interna não vaza.
        assert "RuntimeError" in caplog.text
        assert "nao deu para limpar o contexto" not in caplog.text


# --------------------------------------------------------------------------- #
# T4c-5: result.response não-string (None/int/objeto) => NO_RESPONSE_MESSAGE.
# --------------------------------------------------------------------------- #
class TestNonStringResponse:
    def _result_engine(self, value):
        class _Engine(FakeEngine):
            def process(self, text):
                self._conn.execute("INSERT INTO t VALUES (1)")
                self._conn.commit()
                self.processed.append(text)

                class _R:
                    pass

                r = _R()
                r.response = value
                r.should_exit = False
                return r

        return _Engine

    def test_none_response(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 555, "oi")]])
        ch = channel_factory(client=client, engine_factory=self._result_engine(None))
        ch.start()
        try:
            assert _wait_for(
                lambda: any(t == NO_RESPONSE_MESSAGE for _, t in client.sent)
            )
        finally:
            ch.stop()
            ch.join(2.0)
        assert any(t == NO_RESPONSE_MESSAGE for _, t in client.sent)

    def test_int_response(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 555, "oi")]])
        ch = channel_factory(client=client, engine_factory=self._result_engine(123))
        ch.start()
        try:
            assert _wait_for(
                lambda: any(t == NO_RESPONSE_MESSAGE for _, t in client.sent)
            )
        finally:
            ch.stop()
            ch.join(2.0)
        assert any(t == NO_RESPONSE_MESSAGE for _, t in client.sent)

    def test_object_response(self, channel_factory):
        client = FakeClient(script=[[_private(1, 111, 555, "oi")]])
        ch = channel_factory(client=client, engine_factory=self._result_engine(object()))
        ch.start()
        try:
            assert _wait_for(
                lambda: any(t == NO_RESPONSE_MESSAGE for _, t in client.sent)
            )
        finally:
            ch.stop()
            ch.join(2.0)
        assert any(t == NO_RESPONSE_MESSAGE for _, t in client.sent)


# --------------------------------------------------------------------------- #
# T4c-6: engine_cleanup roda no finally do worker, na thread do engine, uma
#        vez; não roda se a factory falhou; erro de cleanup nunca trava.
# --------------------------------------------------------------------------- #
class TestEngineCleanup:
    def test_cleanup_runs_on_stop(self, channel_factory):
        client = FakeClient(default_updates=[])
        calls = []
        ch = channel_factory(
            client=client, engine_cleanup=lambda e: calls.append(("ok", threading.get_ident()))
        )
        ch.start()
        ch.stop()
        ch.join(3.0)
        assert calls == [("ok", ch._test_holder["engine"].init_thread_id)]

    def test_cleanup_runs_after_later_fail(self, channel_factory, caplog):
        import logging

        class BoomFactory(FakeEngine):
            pass

        created = {}

        def factory():
            created["e"] = FakeEngine()
            return created["e"]

        def cleanup(e):
            created["e"].close()

        # poller falha (erro genérico no get_updates) -> canal failed; o worker,
        # ao ser parado, roda o cleanup uma vez.
        client = FakeClient(script=[RuntimeError("boom")])
        ch = channel_factory(client=client, engine_factory=factory, engine_cleanup=cleanup)
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                assert _wait_for(lambda: ch.is_failed)
            finally:
                ch.stop()
                ch.join(3.0)
        assert created["e"].closed is True

    def test_cleanup_not_run_if_factory_failed(self, channel_factory):
        calls = []

        def bad_factory():
            raise RuntimeError("falha ao construir")

        client = FakeClient(default_updates=[])
        ch = channel_factory(client=client, engine_factory=bad_factory,
                             engine_cleanup=lambda e: calls.append(1))
        ch.start()
        try:
            assert _wait_for(lambda: ch.is_failed)
        finally:
            ch.join(3.0)
        assert calls == []

    def test_cleanup_exception_does_not_block_shutdown(self, channel_factory, caplog):
        import logging

        client = FakeClient(default_updates=[])
        ch = channel_factory(client=client,
                             engine_cleanup=lambda e: (_ for _ in ()).throw(RuntimeError("boom no cleanup")))
        with caplog.at_level(logging.DEBUG):
            ch.start()
            ch.stop()
            ch.join(3.0)
        # encerrou normalmente (threads mortas) e logou só o tipo.
        assert all(not t.is_alive() for t in ch._threads)
        assert "RuntimeError" in caplog.text

    def test_cleanup_runs_in_engine_thread(self, channel_factory):
        client = FakeClient(default_updates=[])
        holder = {}

        def cleanup(e):
            e.close()  # fecha sqlite criado na thread do engine

        ch = channel_factory(client=client, engine_cleanup=cleanup)
        ch.start()
        ch.stop()
        ch.join(3.0)
        engine = ch._test_holder["engine"]
        assert engine is not None
        assert engine.closed is True  # close() na MESMA thread (senão ProgrammingError)


# --------------------------------------------------------------------------- #
# T4c-7: (a) send_chat_action levantando é engolida e a resposta sai;
#        (b) send_message com 400 loga só o código e os blocos seguintes saem.
# --------------------------------------------------------------------------- #
class TestSendEdgeCases:
    def test_send_chat_action_error_swallowed_response_sent(self, channel_factory):
        client = BadActionClient(
            RuntimeError("sem acao"), script=[[_private(1, 111, 555, "oi")]]
        )
        ch = channel_factory(client=client, engine_kwargs={"response": "ola"})
        ch.start()
        try:
            assert _wait_for(lambda: any(t == "ola" for _, t in client.sent))
        finally:
            ch.stop()
            ch.join(2.0)
        # a resposta foi enviada mesmo com o erro da action.
        assert any(t == "ola" for _, t in client.sent)
        assert (555, "typing") in client.actions

    def test_send_400_logs_code_not_text_and_continues(self, channel_factory, caplog):
        import logging

        errors = [
            TelegramAPIError("Bad Request", error_code=400),
            TelegramAPIError("Bad Request", error_code=400),
        ]
        client = SeqSendErrorClient(
            errors,
            script=[[_private(1, 111, 555, "fala muito")]],
        )
        ch = channel_factory(client=client,
                             engine_kwargs={"long_response": " ".join("b%d" % i for i in range(2000))})
        with caplog.at_level(logging.DEBUG):
            ch.start()
            try:
                # espera até algum bloco posterior ter sido entregue
                assert _wait_for(lambda: len(client.sent) >= 1)
            finally:
                ch.stop()
                ch.join(2.0)
        # um bloco seguinte foi enviado apesar dos dois 400.
        assert len(client.sent) >= 1
        # logou o código 400, mas NÃO o texto da mensagem.
        assert "400" in caplog.text
        assert "b0" not in caplog.text  # fragmento do texto não vazou

