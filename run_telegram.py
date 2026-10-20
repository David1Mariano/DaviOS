"""Ponto de entrada do canal Telegram do DaviOS.

Uso:
    python run_telegram.py                   # roda o bot (long polling)
    python run_telegram.py --check           # valida o token via getMe e sai
    python run_telegram.py --user-ids 123 456  # whitelist de ids autorizados

Codigos de saida:
    0  encerrado normalmente (sinal) ou --check valido
    1  o canal FALHOU depois de iniciar (motivo no console; detalhes em
       logs/conversation.log) — nunca fica "vivo sem polling"
    2  token ausente
    3  telegram_enabled=false na config (canal opt-in desligado)

O token pode vir de:
  1. variavel de ambiente DAVIOS_TELEGRAM_TOKEN (prioridade maxima), ou
  2. arquivo .env na raiz do projeto (fallback).

Nunca expoe o token em logs nem em saida.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

# UTF-8 de ponta a ponta no Windows (espelha main.py).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from brain.channels.telegram_channel import TelegramChannel, effective_channel_config
from brain.channels.telegram_client import TelegramClient, TelegramAPIError
from config.davios_config import DaviosConfig
from voice.config import VoiceConfig
from voice.voice_manager import VoiceManager

logger = logging.getLogger("davios.run_telegram")

TOKEN_ENV = "DAVIOS_TELEGRAM_TOKEN"


def build_engine_factory(config: DaviosConfig):
    """Devolve um ``engine_factory`` sem argumentos que constroi o engine.

    A factory e chamada *dentro* da thread worker do
    ``TelegramChannel``, então o engine (SQLite) e criado e usado sempre na
    mesma thread — respeitando o ``check_same_thread=True`` do SQLite.
    """

    def factory():
        # main.build_engine retorna (engine, hardware, model_manager,
        # selection, provider); so o engine interage com o canal.
        import main

        engine, _hw, _mm, _sel, _prov = main.build_engine(config)
        return engine

    return factory


def build_engine_cleanup():
    """Devolve um ``engine_cleanup(engine)`` que descarrega o provider de LLM.

    O canal chama o hook uma unica vez, no finally da thread worker (na mesma
    thread que criou o engine). Descarregar encerra o llama-server standalone,
    evitando processo orfao segurando a porta apos o Ctrl+C.
    """

    def cleanup(engine) -> None:
        manager = getattr(engine, "model_manager", None)
        provider = getattr(manager, "provider", None)
        if provider is None:
            core = getattr(engine, "cognitive_core", None)
            provider = getattr(core, "llm_provider", None)
        unload = getattr(provider, "unload", None)
        if callable(unload):
            unload()

    return cleanup


def run(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="DaviOS - canal Telegram (long polling)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="valida o token via getMe e sai sem rodar o bot",
    )
    parser.add_argument(
        "--user-ids",
        type=int,
        nargs="*",
        default=None,
        help="ids de usuario autorizados (whitelist)",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=30,
        help="timeout do long polling em segundos",
    )
    args = parser.parse_args(argv)

    config = DaviosConfig.load()
    channel_config = effective_channel_config(config)

    # Canal opt-in: com telegram_enabled=false o bot NAO sobe. Sair com codigo
    # distinto de zero evita o "processo vivo sem polling" (zumbi) observado
    # em falhas silenciosas. --check continua permitido (valida o token sem
    # iniciar o bot).
    if not args.check and not config.telegram_enabled:
        print(
            "ERRO: telegram_enabled=false em config/davios.json; bot nao "
            "iniciado. Ative a chave ou use --check apenas para validar o token.",
            file=sys.stderr,
        )
        return 3

    # Carrega .env da raiz do projeto (independente do diretorio atual do shell).
    # override=False preserva a prioridade: variavel de ambiente ja definida
    # vence qualquer valor presente no .env.
    _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(_env_path, override=False)

    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"ERRO: defina a variavel de ambiente {TOKEN_ENV}", file=sys.stderr)
        return 2

    client = TelegramClient(token)

    # VoiceManager (mesmo padrão de main.py: config.voice vem de
    # config/davios_config.py; VoiceManager espera voice.config.VoiceConfig,
    # que tem get_provider_config. Converte preservando os campos idênticos.)
    # A inicialização de voz NUNCA deve derrubar o canal de texto:
    # se falhar, seguimos em modo apenas-texto.
    voice_config = VoiceConfig.from_dict(asdict(config.voice))
    voice_manager = VoiceManager(voice_config)
    try:
        voice_manager.initialize()
    except Exception as exc:  # noqa: BLE001 - voz nunca derruba o bot
        logger.warning(
            "[TG] falha ao inicializar voz; seguindo em modo texto error_type=%s",
            type(exc).__name__,
        )

    # --check: validacao do token (sem iniciar o bot, sem enviar mensagens).
    if args.check:
        try:
            me = client.get_me()
            bot_id = me.get("id")
            username = me.get("username")
            print(
                f"Token valido. Bot: id={bot_id}"
                + (f" username={username}" if username else "")
            )
            return 0
        except TelegramAPIError as exc:
            print(f"Token invalido ou erro de rede: {type(exc).__name__}", file=sys.stderr)
            return 1

    # Whitelist: argumentos CLI sobescrevem a config; lista vazia = ninguem autorizado.
    if args.user_ids is not None:
        user_ids = list(args.user_ids)
    else:
        user_ids = list(config.telegram_allowed_user_ids or [])
    if not user_ids:
        print("AVISO: whitelist vazia — nenhuma mensagem sera processada.", file=sys.stderr)

    engine_factory = build_engine_factory(channel_config)

    channel = TelegramChannel(
        client=client,
        engine_factory=engine_factory,
        allowed_user_ids=user_ids,
        poll_timeout=args.poll_timeout,
        engine_cleanup=build_engine_cleanup(),
        voice_manager=voice_manager,
        voice_config=voice_config,
    )

    stop_event = threading.Event()

    def _handle_signal(signum, frame):  # noqa: ARG001
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    channel.start()
    logger.info("Canal Telegram iniciado.")
    print(
        "DaviOS (Telegram) iniciado - polling ativo. "
        f"Usuarios autorizados: {len(user_ids)}. Ctrl+C para encerrar."
    )
    try:
        while not stop_event.is_set():
            stop_event.wait(0.5)
            # Falha de boot (ex.: engine_factory) NAO pode virar zumbi:
            # sai do loop e o bloco apos o finally reporta em console.
            if channel.is_failed:
                break
    finally:
        channel.stop()
        channel.join(5.0)
        try:
            voice_manager.shutdown()
        except Exception:  # noqa: BLE001
            logger.debug("voice_manager shutdown ignorado")
        logger.info("Canal Telegram encerrado.")

    if channel.is_failed:
        reason = channel.fail_reason or "motivo desconhecido"
        print(f"ERRO: canal Telegram falhou: {reason}", file=sys.stderr)
        print(
            "Detalhes do traceback: logs/conversation.log. "
            "O processo sera encerrado (exit 1).",
            file=sys.stderr,
        )
        return 1
    print("DaviOS (Telegram) encerrado.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())

