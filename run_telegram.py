"""Ponto de entrada do canal Telegram do DaviOS.

Uso:
    python run_telegram.py                   # roda o bot (long polling)
    python run_telegram.py --check           # valida o token via getMe e sai
    python run_telegram.py --user-ids 123 456  # whitelist de ids autorizados

O token deve vir da variavel de ambiente DAVIOS_TELEGRAM_TOKEN.
Nunca expoe o token em logs nem em saida.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

# UTF-8 de ponta a ponta no Windows (espelha main.py).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from brain.channels.telegram_channel import TelegramChannel, effective_channel_config
from brain.channels.telegram_client import TelegramClient, TelegramAPIError
from config.davios_config import DaviosConfig

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

    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"ERRO: defina a variavel de ambiente {TOKEN_ENV}", file=sys.stderr)
        return 2

    client = TelegramClient(token)

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
    )

    stop_event = threading.Event()

    def _handle_signal(signum, frame):  # noqa: ARG001
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    channel.start()
    logger.info("Canal Telegram iniciado.")
    try:
        while not stop_event.is_set():
            stop_event.wait(0.5)
    finally:
        channel.stop()
        channel.join(5.0)
        logger.info("Canal Telegram encerrado.")
    print("DaviOS (Telegram) encerrado.")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    sys.exit(main())

