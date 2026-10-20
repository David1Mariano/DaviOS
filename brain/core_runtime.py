"""Runtime unificado do DaviOS (Fase 3): UM nucleo servindo console + Telegram.

Antes da unificacao havia DOIS boots possiveis: ``python main.py`` (console,
engine criado na thread principal) e ``python run_telegram.py`` (bot, com um
SEGUNDO engine/provider criado na thread worker do canal). Rodar os dois em
paralelo duplicava modelo/provider/llama-server (~3 GB de RAM) e separava a
memoria em dois nucleos incompativeis.

Este modulo resolve o problema com uma THREAD CORE dedicada:

  - A thread core (nome estavel ``davios-core``) e a UNICA dona do
    ``ConversationEngine``, da conexao SQLite do ``MemoryManager``
    (check_same_thread=True) e do provider LLM.
  - Quem fala com o nucleo (console na thread principal, worker do canal
    Telegram) usa o ``EngineProxy``, que serializa cada chamada numa fila
    com ``concurrent.futures.Future`` — nenhuma outra thread toca no engine.
  - O encerramento e idempotente e roda a limpeza final (``provider.unload``)
    DENTRO da thread core, preservando a garantia de mesma-thread.

Decisoes honradas (aprovadas antes da implementacao):
  D2-A  mesmas tools nas duas interfaces: quem construi o engine decide a
        config; o runtime nao a altera nem aplica effective_channel_config.
  D3-A  historico/memoria UNICOS: um so ConversationEngine para os dois canais.
  D5-B  sem lock de instancia: o runtime nao impede um segundo boot; o
        provider standalone ja reutiliza servidor existente via
        _check_existing_server (a convergencia e no servidor, nao aqui).
  D6-A  voz compartilhada com protecao: ``LockedVoice`` serializa a fala do
        console e a voz ocasional do Telegram sobre o MESMO lock.

Ausencia deliberada de asyncio: todo o projeto usa threading/queue e o
console precisa do KeyboardInterrupt do ``input()`` — nada de event loop aqui.
Nenhum handler de sinal e instalado aqui: o console preserva o Ctrl+C nativo
do input(); run_telegram.py (entrada legada) continua dono dos seus handlers.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable, Optional

logger = logging.getLogger("davios.core_runtime")


class EngineProxy:
    """Aparencia de ``ConversationEngine``; despacha cada chamada para a thread core.

    O canal Telegram e o console usam este objeto como se fosse o engine. Por
    baixo, ``CoreRuntime.call`` enfileira a operacao e aguarda o resultado num
    ``Future`` — o engine em si so e tocado na thread core.
    """

    def __init__(self, runtime: "CoreRuntime") -> None:
        self._runtime = runtime

    def process(self, user_input: str, sender_id=None):
        return self._runtime.call(
            self._runtime.engine.process, user_input, sender_id
        )

    def reset_context(self) -> None:
        self._runtime.call(self._runtime.engine.reset_context)
class CoreRuntime:
    """Thread core dona do nucleo (engine + SQLite + provider) do DaviOS.

    Uso minimo::

        runtime = CoreRuntime(engine_factory, voice_manager=voice, final_cleanup=...)
        runtime.start()               # constroi o engine NA thread core
        resultado = runtime.proxy.process("ola")   # serializado na core
        runtime.shutdown()            # idempotente; limpeza na thread core

    ``engine_factory`` (sem argumentos) roda DENTRO da thread core — é ali que
    o ``ConversationEngine`` nasce, junto com a conexao SQLite do banco.
    ``final_cleanup`` (opcional, zero argumentos) tambem roda na thread core,
    no finally do loop, ANTES de a thread morrer: e o lugar certo para
    ``provider.unload()`` (encerra o llama-server filho sem criar orfao).
    """

    def __init__(
        self,
        engine_factory: Callable[[], Any],
        voice_manager: Any = None,
        final_cleanup: Optional[Callable[[], None]] = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._final_cleanup = final_cleanup
        self.voice_manager = voice_manager
        self.engine: Any = None
        self._proxy = EngineProxy(self)

        self._queue: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._voice_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._closed = False

        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False

        self.is_started = False
        self.is_failed = False
        self.failure_reason: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Propriedades
    # ------------------------------------------------------------------ #
    @property
    def proxy(self) -> EngineProxy:
        """Fachada do engine para console e canais (sempre a mesma instancia)."""
        return self._proxy

    @property
    def voice(self) -> Any:
        """VoiceManager protegido pelo lock do runtime (None se nao houver)."""
        if self.voice_manager is None:
            return None
        return LockedVoice(self.voice_manager, self._voice_lock)

    @property
    def is_core_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def start(self, timeout: Optional[float] = None) -> None:
        """Cria o engine NA thread core e aguarda o nucleo ficar pronto.

        Bloqueia ate ``engine_factory`` terminar (o boot do modelo pode levar
        minutos no primeiro carregamento). Uma falha da factory propaga aqui
        (``is_failed`` vira True) e o caller pode decidir cair para modo
        regras com um novo runtime.
        """
        if self.is_started:
            raise RuntimeError("CoreRuntime ja iniciado")
        self.is_started = True
        ready: "Future[None]" = Future()
        self._thread = threading.Thread(
            target=self._core_loop,
            args=(ready,),
            name="davios-core",
            daemon=True,
        )
        self._thread.start()
        ready.result(timeout=timeout)
        if self.is_failed:
            raise RuntimeError(
                f"falha ao iniciar o nucleo: {self.failure_reason or 'desconhecida'}"
            )

    def call(self, fn: Callable[..., Any], /, *args, **kwargs):
        """Executa ``fn(*args, **kwargs)`` NA thread core e devolve o resultado.

        Erros do ``fn`` sao relancados no chamador. Apos ``shutdown()``, novas
        chamadas levantam ``RuntimeError`` (nada e enfileirado depois da
        sentinela de parada — evita Future pendurado para sempre).
        """
        if self.is_failed:
            raise RuntimeError(
                f"nucleo indisponivel: {self.failure_reason or 'desconhecida'}"
            )
        future: "Future[Any]" = Future()
        with self._submit_lock:
            if self._closed:
                raise RuntimeError("nucleo encerrado")
            self._queue.put((fn, args, kwargs, future))
        return future.result()

    def speak(self, text: str) -> bool:
        """Fala texto pelo VoiceManager compartilhado, sob o lock (D6-A)."""
        locked = self.voice
        if locked is None:
            return False
        return bool(locked.speak(text))

    def shutdown(self, timeout: float = 10.0) -> None:
        """Para o nucleo e libera recursos. Idempotente e seguro multi-thread.

        Ordem: fecha novas submissoes, envia sentinela (as chamadas ja
        enfileiradas completam antes), aguarda a thread core — que executa o
        ``final_cleanup`` no proprio finally — e por fim encerra o
        VoiceManager sob o lock de voz.
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True
            with self._submit_lock:
                self._closed = True
                if self._thread is not None:
                    self._queue.put(None)  # sentinela de parada

        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                logger.warning(
                    "[CORE] thread core nao terminou em %ss; seguindo mesmo assim",
                    timeout,
                )

        if self.voice_manager is not None:
            with self._voice_lock:
                try:
                    self.voice_manager.shutdown()
                except Exception:  # noqa: BLE001 - voz nunca derruba o encerramento
                    logger.debug("voice shutdown ignorado", exc_info=True)

    # ------------------------------------------------------------------ #
    # Loop da thread core
    # ------------------------------------------------------------------ #
    def _core_loop(self, ready: "Future[None]") -> None:
        built = False
        try:
            try:
                self.engine = self._engine_factory()
            except BaseException as exc:  # noqa: BLE001 - falha de boot propaga ao start()
                self.is_failed = True
                self.failure_reason = type(exc).__name__
                logger.exception("[CORE] engine_factory falhou")
                ready.set_exception(exc)
                return
            built = True
            ready.set_result(None)

            while True:
                job = self._queue.get()
                if job is None:  # sentinela de shutdown()
                    break
                fn, args, kwargs, future = job
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:  # noqa: BLE001 - erro do job vai ao caller
                    future.set_exception(exc)
                else:
                    future.set_result(result)
        finally:
            # Roda NA thread core (mesma do engine/banco). Limpeza pesada:
            # provider.unload() encerra o llama-server filho sem criar orfao.
            self._closed = True
            if built and self._final_cleanup is not None:
                try:
                    self._final_cleanup()
                except Exception:  # noqa: BLE001 - limpeza nunca propaga
                    logger.exception("[CORE] falha na limpeza final do nucleo")



class LockedVoice:
    """Fachada do ``VoiceManager`` compartilhado, serializada por um lock (D6-A).

    Toda chamada METODO delegada (speak, synthesize, wait_for_playback,
    shutdown, ...) passa pelo mesmo lock do runtime, entao a fala do console e
    a voz ocasional do Telegram nunca executam operacoes do VoiceManager ao
    mesmo tempo. Atributos nao-callables (ex.: config) sao repassados direto.
    """

    def __init__(self, voice_manager: Any, lock: threading.Lock) -> None:
        self._voice = voice_manager
        self._lock = lock

    def __getattr__(self, name: str):
        attr = getattr(self._voice, name)
        if not callable(attr):
            return attr

        def _locked(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)

        return _locked
