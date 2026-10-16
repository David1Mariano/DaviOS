"""Testes de run_telegram.py (sem rede, sem token real, sem iniciar o bot).

Todos os testes usam doubles injetaveis por monkeypatch; nenhum token real e
nenhuma chamada a api.telegram.org e feita.
"""

from __future__ import annotations

import types

import pytest

import run_telegram
from brain.channels.telegram_client import TelegramAPIError

VALID_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"


class FakeClient:
    """Double de TelegramClient configuravel por atributo de classe."""

    get_me_returns = {"id": 123, "username": "test_bot"}
    get_me_raises = None
    instances = []

    def __init__(self, token, transport=None):
        self.token = token
        self.transport = transport
        FakeClient.instances.append(self)

    def get_me(self):
        if FakeClient.get_me_raises is not None:
            raise FakeClient.get_me_raises
        return dict(FakeClient.get_me_returns)


@pytest.fixture(autouse=True)
def _reset_fake_client():
    FakeClient.instances.clear()
    FakeClient.get_me_returns = {"id": 123, "username": "test_bot"}
    FakeClient.get_me_raises = None
    yield


def test_check_valido_retorna_zero(monkeypatch):
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(run_telegram, "TelegramClient", FakeClient)
    rc = run_telegram.run(["--check"])
    assert rc == 0
    assert FakeClient.instances
    assert FakeClient.instances[0].token == VALID_TOKEN


def test_check_valido_imprime_bot_sem_token(monkeypatch, capsys):
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(run_telegram, "TelegramClient", FakeClient)
    rc = run_telegram.run(["--check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "id=123" in out
    assert "username=test_bot" in out
    # O token NUNCA aparece na saida.
    assert VALID_TOKEN not in out
    assert "123456789" not in out


def test_check_invalido_retorna_um(monkeypatch):
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", VALID_TOKEN)
    FakeClient.get_me_raises = TelegramAPIError("Unauthorized", error_code=401)
    monkeypatch.setattr(run_telegram, "TelegramClient", FakeClient)
    rc = run_telegram.run(["--check"])
    assert rc == 1


def test_check_sem_token_retorna_dois(monkeypatch):
    monkeypatch.delenv("DAVIOS_TELEGRAM_TOKEN", raising=False)
    rc = run_telegram.run(["--check"])
    assert rc == 2


def test_build_engine_factory_chama_build_engine(monkeypatch):
    from config.davios_config import DaviosConfig

    calls = []

    def fake_build_engine(config):
        calls.append(config)
        engine = types.SimpleNamespace(process=lambda t: types.SimpleNamespace(response="ok", should_exit=False))
        return engine, None, None, None, None

    monkeypatch.setattr("main.build_engine", fake_build_engine)
    config = DaviosConfig.load()
    factory = run_telegram.build_engine_factory(config)
    engine = factory()
    assert len(calls) == 1
    assert calls[0] is config
    assert engine is not None


def test_effective_channel_config_desliga_ferramentas():
    from config.davios_config import DaviosConfig
    from brain.channels.telegram_channel import effective_channel_config

    config = DaviosConfig()
    cfg = effective_channel_config(config)
    assert cfg is not config  # nunca muta o original
    assert cfg.tools_visible_to_llm is False
    assert cfg.actions_enabled is False
    assert cfg.web_enabled is False

