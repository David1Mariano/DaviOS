"""Testes do boot unificado (Fase 3): console + Telegram num unico processo.

Cenarios cobertos (propostos e aprovados na Fase 2):
  1. inicializacao UNICA de nucleo/modelo/provider (console + Telegram);
  2. nucleo/memory unicos: todas as mensagens passam pela MESMA thread core;
  3. console e Telegram simultaneos sem deadlock;
  4. historico compartilhado: mensagem do console visivel ao Telegram (e vice);
  5. falha do Telegram isolada (console continua vivo);
  6. encerramento limpo: provider.unload + voice.shutdown exatamente 1x;
  7. um so provider por boot (nunca um segundo llama-server);
  8. compatibilidade: console sem Telegram (--no-telegram / sem token /
     telegram_enabled=false) e entrada legada run_telegram.py intacta.

Todos os testes usam doubles injetaveis por monkeypatch: sem rede, sem token
real, sem llama-server e sem tocar no banco real do projeto.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import main
import run_telegram
from brain.conversation_engine import ConversationResult
from brain.core_runtime import CoreRuntime, LockedVoice
from config.davios_config import DaviosConfig, VoiceConfig as ConfigVoiceConfig

VALID_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #
class StubEngine:
    """Engine minimo: registra chamadas (texto, sender, thread) e ecoa."""

    def __init__(self):
        self.processed: list[tuple[str, object]] = []
        self.tids: list[int] = []
        self.reset_calls = 0
        self.reset_tids: list[int] = []

    def process(self, user_input, sender_id=None):
        self.processed.append((user_input, sender_id))
        self.tids.append(threading.get_ident())
        return ConversationResult(response=f"eco:{user_input}", intent="conversation")

    def reset_context(self):
        self.reset_calls += 1
        self.reset_tids.append(threading.get_ident())


class StubProvider:
    """Provider que registra o unload (encerramento do llama-server)."""

    def __init__(self):
        self.unload_calls = 0

    def is_available(self):
        return True

    def health_check(self):
        return {"model": "stub-model", "backend": "stub"}

    def unload(self):
        self.unload_calls += 1


class StubVoiceManager:
    """VoiceManager fakes: registra initialize/speak/wait/shutdown."""

    def __init__(self, config):
        self.config = config
        self.initialize_calls = 0
        self.spoken: list[str] = []
        self.wait_calls = 0
        self.shutdown_calls = 0

    def initialize(self):
        self.initialize_calls += 1
        return True

    def speak(self, text):
        self.spoken.append(text)
        return True

    def wait_for_playback(self, timeout=None):
        self.wait_calls += 1
        return True

    def shutdown(self):
        self.shutdown_calls += 1


class ScriptedTelegramClient:
    """Cliente Telegram fake: updates roteirizadas; registra mensagens enviadas."""

    script: list = []  # lista de lotes de updates (cada poll consome um lote)
    sent: list = []
    instances: list = []
    polls = 0
    get_updates_raises = None
    # Gate opcional: enquanto nao setado, polls devolvem [] (mantem ordem
    # console->Telegram deterministica em testes de historico compartilhado).
    release_gate: threading.Event | None = None

    def __init__(self, token, transport=None):
        self.token = token
        ScriptedTelegramClient.instances.append(self)

    def get_updates(self, offset=None, timeout=None):
        ScriptedTelegramClient.polls += 1
        if ScriptedTelegramClient.get_updates_raises is not None:
            raise ScriptedTelegramClient.get_updates_raises
        if ScriptedTelegramClient.script:
            gate = ScriptedTelegramClient.release_gate
            if gate is not None and not gate.is_set():
                time.sleep(0.02)
                return []
            return ScriptedTelegramClient.script.pop(0)
        time.sleep(0.02)
        return []

    def send_message(self, chat_id, text):
        ScriptedTelegramClient.sent.append((chat_id, text))

    def send_chat_action(self, chat_id, action=None):
        return None


def _update(update_id, sender_id, text):
    """Update valida no formato da API (espelha test_run_telegram.py)."""
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": sender_id, "is_bot": False, "first_name": "T"},
            "chat": {"id": sender_id, "type": "private"},
            "date": 0,
            "text": text,
        },
    }


def _wait_until(cond, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


class Harness:
    """Registra tudo que o boot unificado construiu durante um teste."""

    def __init__(self):
        self.engines: list[StubEngine] = []
        self.providers: list[StubProvider] = []
        self.voice_managers: list[StubVoiceManager] = []
        self.runtimes: list[CoreRuntime] = []
        self.build_calls = 0
        self.build_raises: Exception | None = None


@pytest.fixture(autouse=True)
def _reset_scripted_client():
    ScriptedTelegramClient.script = []
    ScriptedTelegramClient.sent = []
    ScriptedTelegramClient.instances = []
    ScriptedTelegramClient.polls = 0
    ScriptedTelegramClient.get_updates_raises = None
    ScriptedTelegramClient.release_gate = None
    yield



# --------------------------------------------------------------------------- #
# Harness de boot (substitui as bordas I/O do main)
# --------------------------------------------------------------------------- #
def _default_config(**overrides):
    base = dict(
        telegram_enabled=True,
        telegram_allowed_user_ids=[4242],
        voice=ConfigVoiceConfig(enabled=True),
    )
    base.update(overrides)
    return DaviosConfig(**base)


def _scripted_input(inputs, wait_before=None):
    """input() fake: linhas roteirizadas; espera opcional antes de cada linha."""

    state = {"i": 0}

    def fake_input(prompt=""):
        if wait_before is not None:
            assert _wait_until(wait_before), "condicao de espera nunca ocorreu"
        if state["i"] < len(inputs):
            value = inputs[state["i"]]
            state["i"] += 1
            return value
        raise EOFError

    return fake_input


def _setup(monkeypatch, harness, cfg, inputs, wait_before_input=None, token=VALID_TOKEN):
    """Substitui todas as bordas I/O do main; use em seguida _run_main."""

    class FakeHardware:
        def summary_lines(self):
            return ["cpu fake"]

    def fake_build_engine(config):
        harness.build_calls += 1
        if harness.build_raises is not None:
            raise harness.build_raises
        engine = StubEngine()
        provider = StubProvider()
        harness.engines.append(engine)
        harness.providers.append(provider)
        return (
            engine,
            FakeHardware(),
            SimpleNamespace(),
            SimpleNamespace(profile="TEST"),
            provider,
        )

    class RecordingRuntime(CoreRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            harness.runtimes.append(self)

    def fake_voice_manager(config):
        manager = StubVoiceManager(config)
        harness.voice_managers.append(manager)
        return manager

    fallback_engines = []

    def fake_conversation_engine(*args, **kwargs):
        engine = StubEngine()
        fallback_engines.append(engine)
        harness.engines.append(engine)
        return engine

    monkeypatch.setattr(main, "DaviosConfig", SimpleNamespace(load=lambda: cfg))
    monkeypatch.setattr(main, "build_engine", fake_build_engine)
    monkeypatch.setattr(main, "CoreRuntime", RecordingRuntime)
    monkeypatch.setattr(main, "VoiceManager", fake_voice_manager)
    monkeypatch.setattr(main, "ConversationEngine", fake_conversation_engine)
    monkeypatch.setattr(main, "TelegramClient", ScriptedTelegramClient)
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **k: None)
    if token is None:
        monkeypatch.delenv("DAVIOS_TELEGRAM_TOKEN", raising=False)
    else:
        monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", token)

    harness.fallback_engines = fallback_engines
    monkeypatch.setattr("builtins.input", _scripted_input(inputs, wait_before_input))
    return harness


def _run_main(harness, argv=None):
    main.main([] if argv is None else argv)
    return harness


# --------------------------------------------------------------------------- #
# Cenarios 1-4: nucleo unico, thread unica, concorrencia, memoria compartilhada
# --------------------------------------------------------------------------- #
def test_boot_unico_nucleo_e_provider_uma_vez(monkeypatch, capsys):
    """Cenario 1+7: um so build_engine/provider mesmo com Telegram ativo."""
    harness = Harness()
    ScriptedTelegramClient.script = [[], [_update(10, 4242, "mensagem do telegram")]]
    _setup(
        monkeypatch,
        harness,
        _default_config(),
        inputs=["ola console"],
        # So entrega o EOF depois da resposta do Telegram (inclui os dois canais).
        wait_before_input=None,
    )
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(
            ["ola console"], wait_before=lambda: bool(ScriptedTelegramClient.sent)
        ),
    )

    _run_main(harness)

    assert harness.build_calls == 1, "nucleo/provider construidos mais de uma vez"
    assert len(harness.providers) == 1
    engine = harness.engines[0]
    assert ("ola console", None) in engine.processed
    assert ("mensagem do telegram", 4242) in engine.processed
    assert (4242, "eco:mensagem do telegram") in ScriptedTelegramClient.sent
    # Descarga unica do provider (llama-server) no shutdown da thread core.
    assert harness.providers[0].unload_calls == 1


def test_console_e_telegram_usam_a_mesma_thread_core(monkeypatch, capsys):
    """Cenario 2: todo process() roda na MESMA thread core (SQLite seguro)."""
    harness = Harness()
    ScriptedTelegramClient.script = [[], [_update(11, 4242, "do telegram")]]
    _setup(monkeypatch, harness, _default_config(), inputs=["do console"])
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(
            ["do console"], wait_before=lambda: bool(ScriptedTelegramClient.sent)
        ),
    )

    _run_main(harness)

    engine = harness.engines[0]
    assert sorted(engine.processed) == [("do console", None), ("do telegram", 4242)]
    assert len(set(engine.tids)) == 1, "mensagens processadas em threads diferentes"
    assert engine.tids[0] != threading.get_ident(), "core deveria ser thread dedicada"


def test_console_e_telegram_concorrentes_sem_deadlock(monkeypatch, capsys):
    """Cenario 3: console espera o Telegram responder e depois conversa."""
    harness = Harness()
    ScriptedTelegramClient.script = [[], [_update(12, 4242, "sou o telegram")]]
    _setup(monkeypatch, harness, _default_config(), inputs=["sou o console"])
    # O input do console so e entregue DEPOIS da resposta do Telegram chegar:
    # os dois canais estavam ativos no mesmo intervalo sem deadlock.
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(
            ["sou o console"], wait_before=lambda: bool(ScriptedTelegramClient.sent)
        ),
    )

    _run_main(harness)

    engine = harness.engines[0]
    assert ("sou o telegram", 4242) in engine.processed
    assert ("sou o console", None) in engine.processed
    assert (4242, "eco:sou o telegram") in ScriptedTelegramClient.sent
    out = capsys.readouterr().out
    assert "DaviOS: eco:sou o console" in out


def test_historico_unico_intercalado_console_e_telegram(monkeypatch, capsys):
    """Cenario 4 (D3-A): o que o console gravou esta disponivel ao Telegram."""
    harness = Harness()
    ScriptedTelegramClient.script = [[], [_update(13, 4242, "qual e meu nome?")]]
    # Gate: o update do Telegram so e liberado DEPOIS do console processar,
    # tornando a ordem console->Telegram deterministica.
    release = threading.Event()
    ScriptedTelegramClient.release_gate = release

    _setup(monkeypatch, harness, _default_config(), inputs=["meu nome e Davi"])

    state = {"calls": 0}

    def fake_input(prompt=""):
        state["calls"] += 1
        if state["calls"] == 1:
            return "meu nome e Davi"
        assert _wait_until(
            lambda: harness.engines
            and ("meu nome e Davi", None) in harness.engines[0].processed
        ), "mensagem do console nunca processada"
        release.set()  # libera o update do Telegram agora
        assert _wait_until(
            lambda: ("qual e meu nome?", 4242) in harness.engines[0].processed
        ), "mensagem do Telegram nunca processada"
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    _run_main(harness)

    engine = harness.engines[0]
    # Um so engine = um so historico/contexto/memoria, intercalado por ordem
    # de chegada — nada de "duas memorias" paralelas.
    assert engine.processed == [
        ("meu nome e Davi", None),
        ("qual e meu nome?", 4242),
    ]


# --------------------------------------------------------------------------- #
# Cenarios 5-8: isolamento de falha, shutdown, fallback e flags do D1-A
# --------------------------------------------------------------------------- #
def test_falha_do_telegram_nao_derruba_o_console(monkeypatch, capsys):
    """Cenario 5 (R3/R11): 401 no polling avisa 1x e o console segue vivo."""
    from brain.channels.telegram_client import TelegramAPIError

    harness = Harness()
    ScriptedTelegramClient.get_updates_raises = TelegramAPIError(
        "Unauthorized", error_code=401
    )
    _setup(monkeypatch, harness, _default_config(), inputs=["ola"])
    # Espera o poller aplicar o 401 E o canal marcar is_failed antes de o
    # console imprimir o aviso (o sleep so roda apos polls>=1, dentro do loop).
    failed = {"seen": False}

    def wait_failed():
        if not failed["seen"] and ScriptedTelegramClient.polls >= 1:
            time.sleep(0.2)  # margem para o poller marcar is_failed
            failed["seen"] = True
        return failed["seen"]

    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(["ola"], wait_before=wait_failed),
    )

    _run_main(harness)  # nao levanta excecao: console sobrevive

    engine = harness.engines[0]
    assert ("ola", None) in engine.processed
    out = capsys.readouterr().out
    assert "AVISO: canal Telegram parou" in out
    assert "DaviOS: eco:ola" in out
    assert ScriptedTelegramClient.sent == []
    assert harness.providers[0].unload_calls == 1


def test_encerramento_limpo_descarrega_provider_e_voz_uma_vez(monkeypatch, capsys):
    """Cenario 6 (R8): unload/shutdown exatamente 1x e a thread core morta."""
    harness = Harness()
    _setup(monkeypatch, harness, _default_config(), inputs=["oi"])
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(["oi"], wait_before=lambda: bool(harness.engines)),
    )

    _run_main(harness)

    runtime = harness.runtimes[0]
    assert runtime.is_core_alive is False, "thread core deveria ter terminado"
    assert harness.providers[0].unload_calls == 1
    voice = harness.voice_managers[0]
    assert voice.shutdown_calls == 1
    assert voice.wait_calls >= 1
    # A fala do console passou pelo VoiceManager compartilhado (D6-A).
    assert "eco:oi" in voice.spoken
    assert "DaviOS encerrado." in capsys.readouterr().out


def test_modo_regras_quando_build_engine_falha_e_telegram_segue(monkeypatch, capsys):
    """Fallback: build_engine levanta -> engine de regras TAMBEM na thread core."""
    harness = Harness()
    harness.build_raises = RuntimeError("boot quebrado")
    ScriptedTelegramClient.script = [[], [_update(20, 4242, "no modo regras")]]
    _setup(monkeypatch, harness, _default_config(), inputs=["console regras"])
    monkeypatch.setattr(
        "builtins.input",
        _scripted_input(
            ["console regras"], wait_before=lambda: bool(ScriptedTelegramClient.sent)
        ),
    )

    _run_main(harness)

    out = capsys.readouterr().out
    assert "recursos reduzidos" in out
    fallback = harness.fallback_engines[0]
    assert ("console regras", None) in fallback.processed
    assert ("no modo regras", 4242) in fallback.processed
    assert (4242, "eco:no modo regras") in ScriptedTelegramClient.sent
    # O fallback tambem e encerrado pela thread core (sem processos pendurados).
    assert harness.runtimes[-1].is_core_alive is False


def test_no_telegram_flag_nao_constroi_cliente(monkeypatch, capsys):
    """D1-A: --no-telegram suprime o canal mesmo com config+token prontos."""
    harness = Harness()
    _setup(monkeypatch, harness, _default_config(), inputs=["ola"])

    _run_main(harness, argv=["--no-telegram"])

    assert ScriptedTelegramClient.instances == []
    out = capsys.readouterr().out
    assert "Canal Telegram desativado (--no-telegram)." in out
    assert ("ola", None) in harness.engines[0].processed


def test_sem_token_console_somente(monkeypatch, capsys):
    """D1-A: token ausente -> so console, com aviso claro (nunca zumbi)."""
    harness = Harness()
    _setup(
        monkeypatch, harness, _default_config(), inputs=["ola"], token=None
    )

    _run_main(harness)

    assert ScriptedTelegramClient.instances == []
    out = capsys.readouterr().out
    assert "token nao encontrado" in out
    assert ("ola", None) in harness.engines[0].processed


def test_telegram_desligado_na_config(monkeypatch, capsys):
    """D1-A: telegram_enabled=false -> canal nem e considerado."""
    harness = Harness()
    _setup(
        monkeypatch,
        harness,
        _default_config(telegram_enabled=False),
        inputs=["ola"],
    )

    _run_main(harness)

    assert ScriptedTelegramClient.instances == []
    out = capsys.readouterr().out
    assert "telegram_enabled=false" in out


# --------------------------------------------------------------------------- #
# Unidades do runtime: serializacao do proxy, /reset e lock de voz
# --------------------------------------------------------------------------- #
class ReentrantGuardEngine:
    """Detecta solapamento: process() nunca pode rodar concorrente consigo mesmo."""

    def __init__(self):
        self._active = 0
        self.overlaps = 0
        self.calls = 0

    def process(self, user_input, sender_id=None):
        self._active += 1
        try:
            if self._active > 1:
                self.overlaps += 1
            self.calls += 1
            time.sleep(0.01)
            return ConversationResult(response="ok")
        finally:
            self._active -= 1

    def reset_context(self):
        pass


def test_engine_proxy_serializa_acessos_concorrentes():
    """Varias threads falando com o nucleo: uma chamada por vez, sem corrupcao."""
    engine = ReentrantGuardEngine()
    runtime = CoreRuntime(lambda: engine)
    runtime.start()
    try:
        results = []

        def worker():
            for i in range(5):
                results.append(runtime.proxy.process(f"m{i}"))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5.0)

        assert len(results) == 10
        assert all(r.response == "ok" for r in results)
        assert engine.overlaps == 0, "engine executou process() concorrente"
        assert engine.calls == 10
    finally:
        runtime.shutdown()


def test_reset_context_via_proxy_roda_na_thread_core():
    """O /reset do Telegram chama proxy.reset_context na thread core."""
    engine = StubEngine()
    runtime = CoreRuntime(lambda: engine)
    runtime.start()
    try:
        runtime.proxy.process("primeiro")
        runtime.proxy.reset_context()
        assert engine.reset_calls == 1
        # process e reset na MESMA thread core.
        assert engine.reset_tids == [engine.tids[0]]
    finally:
        runtime.shutdown()


class OverlapDetectingVoice:
    """VoiceManager falso que detecta chamadas concorrentes a qualquer metodo."""

    def __init__(self):
        self._active = 0
        self.overlaps = 0
        self.shutdown_calls = 0

    def _enter(self):
        self._active += 1
        if self._active > 1:
            self.overlaps += 1

    def _leave(self):
        self._active -= 1

    def speak(self, text):
        self._enter()
        try:
            time.sleep(0.01)
            return True
        finally:
            self._leave()

    def synthesize(self, text):
        self._enter()
        try:
            time.sleep(0.01)
            return b"audio"
        finally:
            self._leave()

    def shutdown(self):
        self.shutdown_calls += 1


def test_locked_voice_serializa_console_e_voz_do_telegram():
    """D6-A: fala do console e voz do Telegram nunca rodam juntas no manager."""
    voice = OverlapDetectingVoice()
    runtime = CoreRuntime(lambda: ReentrantGuardEngine(), voice_manager=voice)
    runtime.start()
    try:
        locked = runtime.voice  # MESMA fachada usada pelo canal Telegram

        def console_worker():
            for i in range(5):
                runtime.speak(f"console {i}")

        def telegram_voice_worker():
            for i in range(5):
                locked.synthesize(f"telegram {i}")

        threads = [
            threading.Thread(target=console_worker),
            threading.Thread(target=telegram_voice_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5.0)

        assert voice.overlaps == 0, "VoiceManager operado concorrentemente"
    finally:
        runtime.shutdown()
    assert voice.shutdown_calls == 1


def test_entrada_legada_run_telegram_continua_funcional(monkeypatch):
    """D4-A: run_telegram.py segue funcionando sobre o novo main (contrato 5-tuple)."""
    calls = []

    def fake_build_engine(config):
        calls.append(config)
        engine = StubEngine()
        return engine, None, None, None, None

    monkeypatch.setattr("main.build_engine", fake_build_engine)
    cfg = _default_config()
    factory = run_telegram.build_engine_factory(cfg)
    engine = factory()
    assert len(calls) == 1
    assert isinstance(engine, StubEngine)


def test_runtime_shutdown_e_idempotente():
    """shutdown() duplo nao levanta nem re-executa a limpeza."""
    cleanup_calls = []
    engine = StubEngine()
    runtime = CoreRuntime(lambda: engine, final_cleanup=lambda: cleanup_calls.append(1))
    runtime.start()
    runtime.shutdown()
    runtime.shutdown()
    assert cleanup_calls == [1]
    assert runtime.is_core_alive is False
