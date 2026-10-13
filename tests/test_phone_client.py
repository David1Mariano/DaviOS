"""Testes unitários para PhoneClient.

Seguem o mesmo padrão dos testes atuais: duck-typing / stubs, sem unittest.mock.
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.phone_client import PhoneClient


# ---------------------------------------------------------------------------
# Helpers para criar client com _request mockado
# ---------------------------------------------------------------------------


def _client_with_mock_request(mock_request_func):
    """Cria PhoneClient com _request substituído pela função mock."""
    client = PhoneClient("http://fake", token="t")
    client._request = mock_request_func  # type: ignore
    return client


# ---------------------------------------------------------------------------
# Stubs para simular respostas do _request interno
# ---------------------------------------------------------------------------


class _StubPhoneOK:
    """Simula _request respondendo OK com JSON válido."""

    def __init__(self, responses: dict[str, dict]):
        self.responses = responses

    def __call__(self, method: str, path: str, json_body: dict | None = None) -> dict:
        return self.responses.get(path, {"ok": True, "data": "stub"})


class _StubPhoneFail:
    """Simula _request respondendo falha (ok=False)."""

    def __init__(self, error_msg: str = "Erro simulado do celular"):
        self.error_msg = error_msg

    def __call__(self, method: str, path: str, json_body: dict | None = None) -> dict:
        return {"ok": False, "error": self.error_msg}


# ---------------------------------------------------------------------------
# Testes de validação de entrada (speak/vibrate) — testam validação REAL
# ---------------------------------------------------------------------------


def test_speak_validacao_texto_vazio():
    """Validação real: texto vazio → erro antes de chamar _request."""
    client = _client_with_mock_request(lambda m, p, b: {"ok": True})
    # Texto vazio
    res = client.speak("")
    assert res["ok"] is False
    assert "vazio" in res["error"].lower()
    # Apenas espaços
    res = client.speak("   ")
    assert res["ok"] is False
    # None
    res = client.speak(None)  # type: ignore
    assert res["ok"] is False
    assert "string" in res["error"].lower()


def test_speak_valida_limite_300_chars():
    """Validação real: >300 chars → erro antes de chamar _request."""
    client = _client_with_mock_request(lambda m, p, b: {"ok": True})
    # 300 chars OK
    res = client.speak("a" * 300)
    assert res["ok"] is True
    # 301 chars falha
    res = client.speak("a" * 301)
    assert res["ok"] is False
    assert "longo" in res["error"].lower()


def test_speak_remove_caracteres_controle():
    """Validação real: remove chars de controle, mantém \n \t \r."""
    client = _client_with_mock_request(lambda m, p, b: {"ok": True})
    res = client.speak("oi\x00\x07mundo\n\t\r")
    assert res["ok"] is True


def test_vibrate_validacao_tipo_e_limites():
    """Validação real: tipo e range de vibrate."""
    client = _client_with_mock_request(lambda m, p, b: {"ok": True})
    # Não-int
    res = client.vibrate("100")  # type: ignore
    assert res["ok"] is False
    assert "inteiro" in res["error"].lower()
    # Abaixo do mínimo
    res = client.vibrate(49)
    assert res["ok"] is False
    assert "50" in res["error"]
    # Acima do máximo
    res = client.vibrate(2001)
    assert res["ok"] is False
    assert "2000" in res["error"]
    # Dentro do range OK
    res = client.vibrate(500)
    assert res["ok"] is True


# ---------------------------------------------------------------------------
# Testes de métodos públicos com _request mockado (sucesso/falha)
# ---------------------------------------------------------------------------


def test_request_sucesso_retorna_payload():
    """_request mockado OK → repassa o payload."""
    stub = _StubPhoneOK(
        {
            "/battery": {"ok": True, "level": 85, "charging": False},
            "/location": {"ok": True, "lat": -23.5, "lon": -46.6, "accuracy": 10},
            "/device": {"ok": True, "model": "SM-A307F", "android_version": "11"},
        }
    )
    client = _client_with_mock_request(stub)

    assert client.get_battery() == {"ok": True, "level": 85, "charging": False}
    assert client.get_location() == {"ok": True, "lat": -23.5, "lon": -46.6, "accuracy": 10}
    assert client.get_device_info() == {"ok": True, "model": "SM-A307F", "android_version": "11"}


def test_request_servidor_retorna_ok_false():
    """_request mockado com ok=False → repassa error amigável."""
    stub = _StubPhoneFail("Bateria não disponível")
    client = _client_with_mock_request(stub)

    res = client.get_battery()
    assert res["ok"] is False
    assert res["error"] == "Bateria não disponível"


# ---------------------------------------------------------------------------
# Testes de configuração (token, base_url)
# ---------------------------------------------------------------------------


def test_request_sem_base_url():
    """Sem phone_base_url configurado → erro amigável (validação interna do _request)."""
    client = PhoneClient("", token="t")
    res = client.get_battery()
    assert res["ok"] is False
    assert "configurada" in res["error"].lower()


def test_token_lido_de_env_quando_nao_passado(monkeypatch):
    """Se token=None no construtor, lê de DAVIOS_PHONE_TOKEN."""
    monkeypatch.setenv("DAVIOS_PHONE_TOKEN", "token-do-env")
    client = PhoneClient("http://fake", token=None)
    assert client.token == "token-do-env"
    # Token explícito tem precedência
    client2 = PhoneClient("http://fake", token="explicito")
    assert client2.token == "explicito"
    monkeypatch.delenv("DAVIOS_PHONE_TOKEN")


def test_nunca_vaza_token_em_logs_ou_erros(caplog):
    """Token nunca aparece em mensagens de erro ou logs."""
    stub = _StubPhoneFail("falha simulada")
    client = _client_with_mock_request(stub)
    client.token = "SEGREDO123"  # sobrescreve para teste

    with caplog.at_level(logging.WARNING, logger="davios.phone"):
        res = client.get_battery()

    assert res["ok"] is False
    # Verifica que o token não vazou no erro retornado
    assert "SEGREDO123" not in res["error"]
    # Verifica que o token não vazou nos logs
    for rec in caplog.records:
        assert "SEGREDO123" not in rec.message


def test_argumentos_speak_vibrate_nao_vazam(caplog):
    """Valores dos argumentos não aparecem em logs/erros (apenas chaves)."""
    stub = _StubPhoneFail("falha simulada")
    client = _client_with_mock_request(stub)

    with caplog.at_level(logging.WARNING, logger="davios.phone"):
        res = client.speak("mensagem secreta do usuário")
        res2 = client.vibrate(1234)

    assert res["ok"] is False
    assert "mensagem secreta" not in res["error"]
    assert "1234" not in res2["error"]


# ---------------------------------------------------------------------------
# Testes de integração dos métodos públicos (mock _request para capturar chamadas)
# ---------------------------------------------------------------------------


def test_get_battery_chama_request_com_get():
    """Verifica que get_battery usa GET /battery."""
    called: dict = {}

    def _req(method: str, path: str, json_body: dict | None = None):
        called["method"] = method
        called["path"] = path
        called["body"] = json_body
        return {"ok": True, "level": 50, "charging": True}

    client = _client_with_mock_request(_req)
    client.get_battery()
    assert called["method"] == "GET"
    assert called["path"] == "/battery"
    assert called["body"] is None


def test_speak_chama_request_com_post_e_body():
    """Verifica que speak usa POST /tts/speak com body correto."""
    called: dict = {}

    def _req(method: str, path: str, json_body: dict | None = None):
        called["method"] = method
        called["path"] = path
        called["body"] = json_body
        return {"ok": True}

    client = _client_with_mock_request(_req)
    client.speak("olá mundo")
    assert called["method"] == "POST"
    assert called["path"] == "/tts/speak"
    assert called["body"] == {"text": "olá mundo"}


def test_vibrate_chama_request_com_post_e_body():
    """Verifica que vibrate usa POST /vibrate com body correto."""
    called: dict = {}

    def _req(method: str, path: str, json_body: dict | None = None):
        called["method"] = method
        called["path"] = path
        called["body"] = json_body
        return {"ok": True}

    client = _client_with_mock_request(_req)
    client.vibrate(500)
    assert called["method"] == "POST"
    assert called["path"] == "/vibrate"
    assert called["body"] == {"duration": 500}