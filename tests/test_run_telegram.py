"""Testes de run_telegram.py (sem rede, sem token real, sem iniciar o bot).

Todos os testes usam doubles injetaveis por monkeypatch; nenhum token real e
nenhuma chamada a api.telegram.org e feita.
"""

from __future__ import annotations

import os
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
    # Isola: impede que load_dotenv leia .env e redefina a variavel apos o delenv.
    monkeypatch.setattr(run_telegram, "load_dotenv", lambda *a, **k: None)
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


# --------------------------------------------------------------------------- #
# Testes de carregamento de credencial (variavel de ambiente + arquivo .env) #
# --------------------------------------------------------------------------- #

def test_check_com_token_apenas_no_dotenv(monkeypatch):
    """Token presente apenas no .env (nao no ambiente) deve ser encontrado."""
    monkeypatch.delenv("DAVIOS_TELEGRAM_TOKEN", raising=False)

    def fake_load_dotenv(path, override=False):
        """Simula carregamento de .env definindo o token se ainda nao houver."""
        if not override and "DAVIOS_TELEGRAM_TOKEN" not in os.environ:
            os.environ["DAVIOS_TELEGRAM_TOKEN"] = VALID_TOKEN

    import os as _os
    monkeypatch.setattr(run_telegram, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(run_telegram, "TelegramClient", FakeClient)
    rc = run_telegram.run(["--check"])
    assert rc == 0
    assert FakeClient.instances[0].token == VALID_TOKEN


def test_prioridade_env_var_sobre_dotenv(monkeypatch):
    """Variavel de ambiente ja definida vence o valor do .env."""
    token_env = "111111111:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    token_dotenv = "222222222:BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", token_env)

    def fake_load_dotenv(path, override=False):
        """Simula .env com um valor diferente; nao deve sobrescrever env var."""
        if not override and "DAVIOS_TELEGRAM_TOKEN" not in os.environ:
            os.environ["DAVIOS_TELEGRAM_TOKEN"] = token_dotenv

    monkeypatch.setattr(run_telegram, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(run_telegram, "TelegramClient", FakeClient)
    rc = run_telegram.run(["--check"])
    assert rc == 0
    # O TelegramClient deve ter recebido o token do ambiente, nao do .env.
    assert FakeClient.instances[0].token == token_env


def test_erro_sem_token_nao_vaza_token(monkeypatch, capsys):
    """Quando o token esta ausente, a mensagem de erro nao deve vazar credenciais."""
    monkeypatch.delenv("DAVIOS_TELEGRAM_TOKEN", raising=False)

    def fake_load_dotenv(path, override=False):
        pass  # nao carrega nada do .env

    import os as _os
    monkeypatch.setattr(run_telegram, "load_dotenv", fake_load_dotenv)
    rc = run_telegram.run(["--check"])
    assert rc == 2
    err = capsys.readouterr().err
    # Menciona o nome da variavel, mas nao o valor de nenhum token.
    assert "DAVIOS_TELEGRAM_TOKEN" in err
    assert VALID_TOKEN not in err
    assert "123456789" not in err


def test_load_dotenv_chamado_com_caminho_da_raiz(monkeypatch):
    """load_dotenv deve ser chamado com o caminho do .env na raiz do projeto."""
    monkeypatch.delenv("DAVIOS_TELEGRAM_TOKEN", raising=False)

    chamado_com = {}

    def fake_load_dotenv(path, override=False):
        chamado_com["path"] = str(path)
        chamado_com["override"] = override

    monkeypatch.setattr(run_telegram, "load_dotenv", fake_load_dotenv)
    monkeypatch.setattr(run_telegram, "TelegramClient", FakeClient)
    rc = run_telegram.run(["--check"])
    assert rc == 2  # token ainda ausente (fake nao define nada)
    assert "override" in chamado_com
    assert chamado_com["override"] is False
    assert chamado_com["path"].endswith(".env")


# --------------------------------------------------------------------------- #
# Ciclo de vida do bot: opt-in, polling ativo, falha visivel e rota de resposta #
# --------------------------------------------------------------------------- #

import signal as signal_mod  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402


def _load_config_clone(monkeypatch, **overrides):
    """Clona a config real com canal habilitado e whitelist fixa (4242)."""
    from config.davios_config import DaviosConfig

    cfg = DaviosConfig.load()
    cfg.telegram_enabled = True
    cfg.telegram_allowed_user_ids = [4242]
    cfg.owner_user_id = 4242
    for key, value in overrides.items():
        setattr(cfg, key, value)
    monkeypatch.setattr(run_telegram.DaviosConfig, "load", lambda *a, **k: cfg)
    return cfg


class _LifecycleClient(FakeClient):
    """FakeClient com long polling, envio e roteiro de updates."""

    script: list = []
    sent: list = []
    polls: list = []

    def get_updates(self, offset=None, timeout=30, allowed_updates=None):
        _LifecycleClient.polls.append(offset)
        if _LifecycleClient.script:
            return _LifecycleClient.script.pop(0)
        time.sleep(0.02)
        return []

    def send_message(self, chat_id, text):
        _LifecycleClient.sent.append((chat_id, text))
        return {"message_id": 1}

    def send_chat_action(self, chat_id, action="typing"):
        return True


@pytest.fixture(autouse=True)
def _reset_lifecycle_client():
    _LifecycleClient.script = []
    _LifecycleClient.sent = []
    _LifecycleClient.polls = []
    yield


class _SignalCapture:
    """Substitui signal.signal e guarda os handlers instalados por run()."""

    def __init__(self):
        self.handlers = {}

    def __call__(self, signum, handler):
        self.handlers[signum] = handler
        return None


def _run_direct(monkeypatch, argv=None):
    capture = _SignalCapture()
    monkeypatch.setattr(run_telegram.signal, "signal", capture)
    return run_telegram.run(argv or []), capture


def _run_in_thread(monkeypatch):
    capture = _SignalCapture()
    monkeypatch.setattr(run_telegram.signal, "signal", capture)
    state = {}

    def _target():
        state["rc"] = run_telegram.run([])

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    return thread, state, capture


def _wait_until(predicate, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _stop_and_join(capture, thread):
    handler = capture.handlers.get(signal_mod.SIGINT)
    assert handler is not None, "run() nao instalou handler de SIGINT"
    handler(signal_mod.SIGINT, None)
    thread.join(10.0)
    assert not thread.is_alive(), "run() nao encerrou apos o sinal"


def test_run_com_canal_desligado_retorna_tres(monkeypatch, capsys):
    _load_config_clone(monkeypatch, telegram_enabled=False)
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(run_telegram, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(run_telegram, "TelegramClient", FakeClient)

    rc = run_telegram.run([])

    assert rc == 3
    err = capsys.readouterr().err
    assert "telegram_enabled=false" in err
    # Nem o cliente foi criado: nada de polling.
    assert FakeClient.instances == []


def test_run_check_continua_com_canal_desligado(monkeypatch):
    _load_config_clone(monkeypatch, telegram_enabled=False)
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(run_telegram, "TelegramClient", FakeClient)

    rc = run_telegram.run(["--check"])

    assert rc == 0
    assert FakeClient.instances


def test_config_real_tem_canal_e_campo_do_provedor():
    # Regressao da causa raiz: a chave "llama_server_exe" do JSON era filtrada
    # (campo inexistente na dataclass) -> AttributeError em _find_server_exe
    # derrubava o boot. E garantia de que o bot nao volta a sair com o canal
    # desabilitado (telegram_enabled false/ausente = exit 3).
    from config.davios_config import DaviosConfig

    cfg = DaviosConfig.load()
    assert cfg.telegram_enabled is True
    assert hasattr(cfg, "llama_server_exe")
    assert cfg.llama_server_exe is None or isinstance(cfg.llama_server_exe, str)


def test_run_factory_quebrada_retorna_um_com_diagnostico(monkeypatch, capsys):
    """Falha de boot do engine vira exit 1 + diagnostico em console (nao zumbi)."""
    _load_config_clone(monkeypatch)
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(run_telegram, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(run_telegram, "TelegramClient", _LifecycleClient)

    def broken_factory():
        raise RuntimeError("segredo interno do provedor")

    monkeypatch.setattr(
        run_telegram, "build_engine_factory", lambda cfg: broken_factory
    )

    rc, _capture = _run_direct(monkeypatch)

    assert rc == 1
    err = capsys.readouterr().err
    assert "engine_factory falhou" in err
    assert "RuntimeError" in err
    # Diagnostico sem vazar a mensagem interna nem o token.
    assert "segredo interno do provedor" not in err
    assert VALID_TOKEN not in err


def test_run_permanece_ativo_ate_sinal_e_encerra_com_zero(monkeypatch, capsys):
    _load_config_clone(monkeypatch)
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(run_telegram, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(run_telegram, "TelegramClient", _LifecycleClient)
    monkeypatch.setattr(
        run_telegram,
        "build_engine_factory",
        lambda cfg: lambda: types.SimpleNamespace(
            process=lambda text, sender_id=None: types.SimpleNamespace(response="ok")
        ),
    )

    thread, state, capture = _run_in_thread(monkeypatch)

    # Polling realmente ativo: descarte de backlog + chamada do loop principal.
    assert _wait_until(lambda: len(_LifecycleClient.polls) >= 2), "polling nunca comecou"
    _stop_and_join(capture, thread)

    assert state.get("rc") == 0
    out = capsys.readouterr().out
    assert "polling ativo" in out
    assert "DaviOS (Telegram) encerrado." in out
    # O token nunca aparece na saida.
    assert VALID_TOKEN not in out


def _scripted_engine_factory(processed):
    def engine_factory():
        def _process(text, sender_id=None):
            processed.append((text, sender_id))
            return types.SimpleNamespace(response="resposta-ok")

        return types.SimpleNamespace(process=_process)

    return engine_factory


def _update(update_id, sender_id, text):
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


def test_run_mensagem_autorizada_recebe_resposta_no_chat(monkeypatch):
    _load_config_clone(monkeypatch)  # whitelist: [4242]
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(run_telegram, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(run_telegram, "TelegramClient", _LifecycleClient)

    processed = []
    monkeypatch.setattr(
        run_telegram,
        "build_engine_factory",
        lambda cfg: _scripted_engine_factory(processed),
    )
    # 1ª chamada e o descarte de backlog (fica vazia); a 2ª entrega a mensagem.
    _LifecycleClient.script = [[], [_update(10, 4242, "ola davios")]]

    thread, state, capture = _run_in_thread(monkeypatch)

    assert _wait_until(lambda: bool(_LifecycleClient.sent)), "resposta nunca enviada"
    _stop_and_join(capture, thread)

    assert state.get("rc") == 0
    assert processed == [("ola davios", 4242)]
    chat_id, text = _LifecycleClient.sent[0]
    assert chat_id == 4242
    assert "resposta-ok" in text


def test_run_mensagem_nao_autorizada_e_ignorada(monkeypatch):
    _load_config_clone(monkeypatch)  # whitelist: [4242]
    monkeypatch.setenv("DAVIOS_TELEGRAM_TOKEN", VALID_TOKEN)
    monkeypatch.setattr(run_telegram, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(run_telegram, "TelegramClient", _LifecycleClient)

    processed = []
    monkeypatch.setattr(
        run_telegram,
        "build_engine_factory",
        lambda cfg: _scripted_engine_factory(processed),
    )
    _LifecycleClient.script = [[], [_update(11, 9999, "invasor")]]

    thread, state, capture = _run_in_thread(monkeypatch)

    # Espera o update ser consumido pelo poller (whitelist bloqueia o resto).
    assert _wait_until(lambda: not _LifecycleClient.script), "update nunca consumido"
    time.sleep(0.3)  # margem para um eventual envio indevido
    _stop_and_join(capture, thread)

    assert state.get("rc") == 0
    assert processed == []
    assert _LifecycleClient.sent == []

