"""Canal Telegram do DaviOS (adaptador fino sobre o núcleo).

Um ``TelegramChannel`` faz long polling (via um ``TelegramClient`` injetável) e
processa cada mensagem autorizada num worker serial. O núcleo
(``ConversationEngine``) NÃO é alterado: é criado por uma ``engine_factory``
DENTRO da thread worker e só usado nela (SQLite tem ``check_same_thread=True``).

Decisões honradas:
  D1  transporte HTTP injetado no client (sem rede nos testes).
  D2  fila + UM worker serial; engine criado e usado só no worker.
  D3  ``effective_channel_config`` desliga ferramentas por padrão.
  D4  ``result.should_exit`` NUNCA encerra o canal.
  D5  o token vive só no client; este módulo não o vê.
  D6  respostas em texto puro (o client não usa parse_mode).
  D7  só chats privados (``extract_messages`` já filtra o resto).
  D8  whitelist por ``from.id`` (sender); resposta para ``chat.id``.
"""

from __future__ import annotations

import dataclasses
import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

from brain.channels.base import Channel, IncomingMessage, is_authorized
from brain.channels.telegram_client import (
    TelegramAPIError,
    chunk_text,
    extract_messages,
)
from config.davios_config import DaviosConfig

# Frases fixas (nunca expõem traceback nem texto de erro ao usuário).
BUSY_MESSAGE = "Estou ocupado, tente em instantes."
NO_RESPONSE_MESSAGE = "(sem resposta)"
PROCESS_ERROR_MESSAGE = "Tive um problema ao processar sua mensagem."
UNKNOWN_COMMAND_MESSAGE = "Comando desconhecido."
START_MESSAGE = "Ola! Sou o DaviOS. Como posso ajudar?"
RESET_MESSAGE = "Conversa reiniciada."

# Política de tamanho de entrada (espelha main.py: rejeita, não trunca).
MAX_INPUT_CHARS = 2000
TOO_LONG_MESSAGE = "Essa mensagem e muito longa. Pode resumir?"

# Throttle do log de remetente não autorizado (para descobrir o próprio id).
UNAUTHORIZED_LOG_INTERVAL_SECONDS = 600.0
# Teto do dict de throttle (evita crescimento sem fim com muitos senders).
UNAUTHORIZED_LOG_MAX_ENTRIES = 256

# Backoff do poller.
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0

# Teto do retry do ENVIO (429): distinto do backoff do poller.
SEND_RETRY_MAX_SECONDS = 30.0


def effective_channel_config(config: DaviosConfig) -> DaviosConfig:
    """Devolve a config que o engine do canal deve usar (D3).

    Se ``config.telegram_tools_enabled`` for False (padrão), devolve uma CÓPIA
    com ``tools_visible_to_llm=False``, ``actions_enabled=False`` e
    ``web_enabled=False``. Se True, devolve a própria config (inalterada). Nunca
    muta o objeto original.
    """
    if getattr(config, "telegram_tools_enabled", False):
        return config
    return dataclasses.replace(
        config,
        tools_visible_to_llm=False,
        actions_enabled=False,
        web_enabled=False,
    )


class TelegramChannel(Channel):
    """Canal Telegram com poller + worker serial.

    Tudo é injetável para permitir testes sem rede e sem llama-server:
      - ``client``: get_updates/send_message/send_chat_action.
      - ``engine_factory``: chamável SEM argumentos que devolve um
        ``ConversationEngine``; é chamado DENTRO da thread worker.
      - ``sleep``: padrão ``stop_event.wait`` (torna o ``stop()`` imediato);
        nos testes, um relógio falso registra as esperas sem dormir de verdade.
    """

    name = "telegram"

    def __init__(
        self,
        client: Any,
        engine_factory: Callable[[], Any],
        allowed_user_ids,
        poll_timeout: int = 30,
        queue_max: int = 20,
        chunk_limit: int = 4000,
        sleep: Optional[Callable[[float], None]] = None,
        logger: Optional[logging.Logger] = None,
        now: Optional[Callable[[], float]] = None,
        engine_cleanup: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._client = client
        self._engine_factory = engine_factory
        self._allowed_user_ids = allowed_user_ids
        self._poll_timeout = poll_timeout
        self._chunk_limit = chunk_limit
        self._log = logger or logging.getLogger("davios.channels.telegram")

        self._queue: "queue.Queue[Optional[IncomingMessage]]" = queue.Queue(
            maxsize=queue_max
        )
        self._stop_event = threading.Event()
        self._started = False
        self._failed = False
        self._engine: Any = None
        self._threads: list[threading.Thread] = []

        # Throttle do log de não autorizado: sender_id -> último tempo logado.
        self._unauth_last_logged: dict[int, float] = {}
        self._lock = threading.Lock()

        # Relógio injetável para o throttle de log (default: time.monotonic).
        self._now = now or time.monotonic

        # Hook de limpeza do engine: chamado UMA vez, no finally do worker, na
        # MESMA thread que construiu o engine (protegido; nunca propaga).
        self._engine_cleanup = engine_cleanup

        # Espera injetável: por padrão, espera no stop_event (acordando antes se
        # stop() for chamado). Assinatura (segundos) -> bool/None.
        if sleep is None:
            self._sleep = lambda seconds: self._stop_event.wait(seconds)
        else:
            self._sleep = sleep

    # ------------------------------------------------------------------ #
    # Estado
    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        if not self._started or self._failed or self._stop_event.is_set():
            return False
        return bool(self._threads) and all(t.is_alive() for t in self._threads)

    @property
    def is_failed(self) -> bool:
        return self._failed

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Inicia poller + worker em threads daemon. NÃO bloqueia.

        Levanta ``RuntimeError`` se chamado duas vezes.
        """
        if self._started:
            raise RuntimeError("TelegramChannel ja iniciado")
        self._started = True
        self._stop_event.clear()

        poller = threading.Thread(
            target=self._poll_loop, name="davios-tg-poller", daemon=True
        )
        worker = threading.Thread(
            target=self._work_loop, name="davios-tg-worker", daemon=True
        )
        self._threads = [poller, worker]
        poller.start()
        worker.start()

    def stop(self) -> None:
        """Encerra o canal. Idempotente e imediato (acorda as esperas)."""
        self._stop_event.set()
        # Acorda o worker caso esteja bloqueado esperando a fila.
        try:
            self._queue.put_nowait(None)  # sentinela
        except queue.Full:
            pass

    def join(self, timeout: Optional[float] = None) -> None:
        """Aguarda as threads terminarem (usado em testes)."""
        for t in self._threads:
            t.join(timeout)

    # ------------------------------------------------------------------ #
    # Poller (thread separada; NUNCA toca no engine)
    # ------------------------------------------------------------------ #
    def _poll_loop(self) -> None:
        """Loop do poller. Qualquer exceção não prevista => loga o TIPO e falha (item 3)."""
        try:
            # Fail-closed (item 2): o loop principal SÓ começa depois que o
            # descarte de backlog der certo (ou o canal falhar/parar).
            offset = self._startup_backlog_offset()
            if self._failed or self._stop_event.is_set():
                return

            backoff = BACKOFF_BASE_SECONDS
            while not self._stop_event.is_set():
                try:
                    updates = self._client.get_updates(
                        offset=offset, timeout=self._poll_timeout
                    )
                except TelegramAPIError as exc:
                    wait, grow = self._poll_error_wait(exc, backoff)
                    if wait is None:  # fatal
                        self._fail("token invalido")
                        return
                    self._sleep(wait)
                    if grow:
                        backoff = min(backoff * 2.0, BACKOFF_MAX_SECONDS)
                    continue
                else:
                    backoff = BACKOFF_BASE_SECONDS  # sucesso zera o backoff

                messages, next_offset = extract_messages(updates)
                for msg in messages:
                    if self._stop_event.is_set():
                        return
                    self._dispatch(msg)
                # Avança o offset SOMENTE após enfileirar/descartar cada mensagem.
                if next_offset is not None:
                    offset = next_offset
        except Exception as exc:  # noqa: BLE001 - não derruba a thread silenciosamente
            self._log.error(
                "[TG] excecao inesperada no poller error_type=%s", type(exc).__name__
            )
            self._fail("excecao inesperada no poller")

    def _startup_backlog_offset(self) -> Optional[int]:
        """Descarta o backlog de forma fail-closed e devolve o offset inicial (item 2).

        Repete ``get_updates(offset=-1)`` até dar certo ou o canal parar. Usa a
        MESMA política de erros do loop principal: 401 fatal, 409 espera 30 s,
        429 espera ``retry_after``, demais backoff exponencial (1,2,4,... cap 60).
        Só retorna quando o descarte tem sucesso; o valor é o offset com o qual
        o loop principal COMEÇA. Nada é processado antes disso.
        """
        backoff = BACKOFF_BASE_SECONDS
        while not self._stop_event.is_set():
            try:
                next_offset = self._discard_backlog()
            except TelegramAPIError as exc:
                wait, grow = self._poll_error_wait(exc, backoff)
                if wait is None:  # fatal (401)
                    self._fail("token invalido")
                    return None
                self._sleep(wait)
                if grow:
                    backoff = min(backoff * 2.0, BACKOFF_MAX_SECONDS)
                continue
            return next_offset
        return None  # stop() acordado enquanto esperava

    def _poll_error_wait(self, exc: TelegramAPIError, backoff: float):
        """Decide a espera após um erro de get_updates.

        Retorna ``(wait_seconds, grow_backoff)``:
          - ``(None, False)`` -> fatal (401), aborta o canal.
          - 409 -> (30.0, False): espera fixa, sem crescer backoff.
          - 429 -> (retry_after teto 60, False): espera fixa do servidor.
          - demais -> (backoff, True): exponencial 1,2,4,... teto 60.
        """
        code = exc.error_code
        if code == 401:
            self._log.error("[TG] token invalido (401); canal sera encerrado")
            return None, False
        if code == 409:
            self._log.error(
                "[TG] 409 Conflict: outra instancia usando o mesmo bot "
                "ou webhook ativo; aguardando 30s"
            )
            return 30.0, False
        if code == 429:
            wait = min(float(exc.retry_after or 1.0), BACKOFF_MAX_SECONDS)
            self._log.warning("[TG] 429 no long polling; aguardando %ss", wait)
            return wait, False
        self._log.warning("[TG] erro no long polling error_code=%s", code)
        return backoff, True

    def _discard_backlog(self) -> Optional[int]:
        """Descarta mensagens pendentes ao iniciar e devolve o offset inicial (item 2).

        Retorna o ``update_id`` maior + 1 visto no descarte, ou ``None`` se a
        fila estiver vazia. O loop principal começa EXATAMENTE com esse offset,
        garantindo que mensagens antigas não sejam (re)processadas.
        """
        updates = self._client.get_updates(offset=-1, timeout=0)
        _, next_offset = extract_messages(updates)
        if next_offset is not None:
            self._log.info("[TG] backlog descartado (offset inicial=%s)", next_offset)
        return next_offset

    def _dispatch(self, msg: IncomingMessage) -> None:
        """Aplica whitelist ANTES de tudo; enfileira se autorizado."""
        if not is_authorized(msg.sender_id, self._allowed_user_ids):
            self._log_unauthorized(msg.sender_id)
            return  # sem send, sem action, sem enfileirar, sem process()
        self._enqueue_or_busy(msg)

    def _enqueue_or_busy(self, msg: IncomingMessage) -> None:
        """Enfileira; se a fila estiver cheia, responde 'ocupado' e descarta."""
        try:
            self._queue.put_nowait(msg)
        except queue.Full:
            self._reply_safe(msg.chat_id, BUSY_MESSAGE)

    def _log_unauthorized(self, sender_id: int) -> None:
        """Registra sender_id (NUNCA o texto), no máximo 1x/10min por sender.

        O dict é limitado a ``UNAUTHORIZED_LOG_MAX_ENTRIES`` entradas: ao
        estourar, remove primeiro as expiradas (>10 min) e, se ainda estourar,
        as mais antigas. O relógio é injetável (``self._now``).
        """
        now = self._now()
        with self._lock:
            last = self._unauth_last_logged.get(sender_id)
            if last is not None and (now - last) < UNAUTHORIZED_LOG_INTERVAL_SECONDS:
                return  # ainda dentro da janela: sem log duplicado
            if (
                sender_id not in self._unauth_last_logged
                and len(self._unauth_last_logged) >= UNAUTHORIZED_LOG_MAX_ENTRIES
            ):
                # 1) remove expirados primeiro.
                for key, ts in list(self._unauth_last_logged.items()):
                    if (now - ts) >= UNAUTHORIZED_LOG_INTERVAL_SECONDS:
                        del self._unauth_last_logged[key]
                # 2) se ainda estourar, remove os mais antigos.
                while len(self._unauth_last_logged) >= UNAUTHORIZED_LOG_MAX_ENTRIES:
                    oldest = min(
                        self._unauth_last_logged, key=self._unauth_last_logged.get
                    )
                    del self._unauth_last_logged[oldest]
            self._unauth_last_logged[sender_id] = now
        self._log.info(
            "[TG] mensagem de remetente nao autorizado sender_id=%s", sender_id
        )

    def _fail(self, reason: str) -> None:
        """Coloca o canal em estado de falha (sem travar threads) e registra o motivo."""
        self._log.error("[TG] canal falhou: %s", reason)
        self._failed = True
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Worker serial (thread separada; ÚNICA que cria e usa o engine)
    # ------------------------------------------------------------------ #
    def _work_loop(self) -> None:
        # O engine é criado AQUI, dentro da thread que o usará (SQLite tem
        # check_same_thread=True). Se a factory falhar, o canal entra em falha.
        try:
            self._engine = self._engine_factory()
        except Exception as exc:  # noqa: BLE001 - falha de boot do engine
            self._log.error(
                "[TG] falha ao construir o engine error_type=%s", type(exc).__name__
            )
            self._fail("engine_factory falhou")
            return

        try:
            while not self._stop_event.is_set():
                try:
                    msg = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if msg is None:  # sentinela de stop()
                    break
                try:
                    self._handle_message(msg)
                except Exception as exc:  # noqa: BLE001 - nunca derruba o worker
                    self._log.error(
                        "[TG] erro inesperado no worker error_type=%s",
                        type(exc).__name__,
                    )
                finally:
                    self._queue.task_done()
        finally:
            # Limpeza do engine: UMA vez, na MESMA thread que o construiu.
            # Só chegamos aqui se a factory teve sucesso (senão já retornamos).
            if self._engine is not None and self._engine_cleanup is not None:
                try:
                    self._engine_cleanup(self._engine)
                except Exception as exc:  # noqa: BLE001 - nunca propaga
                    self._log.error(
                        "[TG] erro no engine_cleanup error_type=%s",
                        type(exc).__name__,
                    )

    def _handle_message(self, msg: IncomingMessage) -> None:
        # 1) indicador "digitando..." em best-effort (exceção engolida).
        self._typing(msg.chat_id)
        text = (msg.text or "").strip()

        # 2) política de tamanho (espelha main.py: rejeita, não trunca).
        if len(text) > MAX_INPUT_CHARS:
            self._reply(msg.chat_id, TOO_LONG_MESSAGE)
            return

        # 3) comandos ("/...") NÃO vão para o LLM.
        if text.startswith("/"):
            self._handle_command(msg.chat_id, text)
            return

        # 4) texto normal -> engine.process (D4: should_exit NUNCA para o canal).
        try:
            result = self._engine.process(text)
        except Exception as exc:  # noqa: BLE001
            self._log.error(
                "[TG] erro ao processar error_type=%s", type(exc).__name__
            )
            self._reply(msg.chat_id, PROCESS_ERROR_MESSAGE)
            return

        response = getattr(result, "response", "") or ""
        # Resposta não-string (None/int/objeto) não deve explodir: trata como vazia.
        if not isinstance(response, str) or not response.strip():
            self._reply(msg.chat_id, NO_RESPONSE_MESSAGE)
            return
        self._send_chunks(msg.chat_id, response)

    def _handle_command(self, chat_id: int, text: str) -> None:
        # Normaliza "/cmd@nomedobot" e caixa: pega o primeiro token, minúsculo.
        first = text.split()[0] if text.split() else ""
        cmd = first[1:].split("@", 1)[0].lower()
        if cmd == "start":
            self._reply(chat_id, START_MESSAGE)
        elif cmd == "reset":
            try:
                self._engine.reset_context()  # chamado no worker (mesma thread do engine)
            except Exception as exc:  # noqa: BLE001
                self._log.error(
                    "[TG] falha ao reiniciar o contexto error_type=%s",
                    type(exc).__name__,
                )
                self._reply(chat_id, PROCESS_ERROR_MESSAGE)
                return
            self._reply(chat_id, RESET_MESSAGE)
        else:
            self._reply(chat_id, UNKNOWN_COMMAND_MESSAGE)

    # ------------------------------------------------------------------ #
    # Envio (com retry só para 429)
    # ------------------------------------------------------------------ #
    def _send_chunks(self, chat_id: int, response: str) -> None:
        """Envia a resposta em blocos, em ordem, um send_message por bloco."""
        for block in chunk_text(response, self._chunk_limit):
            self._send_with_retry(chat_id, block)

    def _send_with_retry(self, chat_id: int, text: str) -> None:
        try:
            self._client.send_message(chat_id, text)
        except TelegramAPIError as exc:
            if exc.error_code == 429:
                wait = min(float(exc.retry_after or 1.0), SEND_RETRY_MAX_SECONDS)
                self._sleep(wait)
                try:
                    self._client.send_message(chat_id, text)  # UMA nova tentativa
                except TelegramAPIError as exc2:
                    self._log.warning(
                        "[TG] falha ao reenviar apos 429 error_code=%s",
                        exc2.error_code,
                    )
            else:
                # Loga só o código (nunca o texto/descrição).
                self._log.warning("[TG] falha ao enviar error_code=%s", exc.error_code)

    def _reply(self, chat_id: int, text: str) -> None:
        """Resposta de frase fixa/curta (sem chunking)."""
        self._send_with_retry(chat_id, text)

    def _reply_safe(self, chat_id: int, text: str) -> None:
        """Como _reply, mas UMA tentativa, sem retry e sem sleep (usado pelo poller).

        O poller NUNCA deve dormir por causa do envio (nem travar). Engole
        qualquer exceção; loga só o tipo.
        """
        try:
            self._client.send_message(chat_id, text)
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                "[TG] falha ao enviar (reply_safe) error_type=%s", type(exc).__name__
            )

    def _typing(self, chat_id: int) -> None:
        """send_chat_action('typing') em best-effort (exceção engolida)."""
        try:
            self._client.send_chat_action(chat_id, action="typing")
        except Exception:  # noqa: BLE001
            pass


