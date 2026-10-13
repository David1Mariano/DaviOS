"""Testes do _request REAL do PhoneClient usando http.server local.

Sem unittest.mock, sem patchar urllib. Usa ThreadingHTTPServer em porta efêmera 127.0.0.1.
"""

import json
import logging
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.phone_client import PhoneClient


# ---------------------------------------------------------------------------
# Handlers HTTP para simular diferentes cenários
# ---------------------------------------------------------------------------


class _OKHandler(BaseHTTPRequestHandler):
    """Responde 200 com JSON ok=True."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "level": 85, "charging": False}).encode())

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "spoken": True}).encode())

    def log_message(self, format, *args):
        pass  # silencia logs do servidor


class _FailHandler(BaseHTTPRequestHandler):
    """Responde 200 com JSON ok=False + error."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": False, "error": "Falha simulada no servidor"}).encode())

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": False, "error": "Falha simulada no servidor"}).encode())

    def log_message(self, format, *args):
        pass


class _HTTPErrorHandler(BaseHTTPRequestHandler):
    """Responde com código HTTP de erro (401, 500, etc.)."""

    def __init__(self, *args, status_code=401, **kwargs):
        self._status_code = status_code
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.send_response(self._status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"detail": "Erro HTTP"}).encode())

    def do_POST(self):
        self.send_response(self._status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"detail": "Erro HTTP"}).encode())

    def log_message(self, format, *args):
        pass


class _NotJSONHandler(BaseHTTPRequestHandler):
    """Responde com corpo que não é JSON."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"isto nao e json")

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"tambem nao e json")

    def log_message(self, format, *args):
        pass


class _NotDictHandler(BaseHTTPRequestHandler):
    """Responde com JSON que não é objeto (array, string, number)."""

    def __init__(self, *args, payload=None, **kwargs):
        self._payload = payload
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self._payload).encode())

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self._payload).encode())

    def log_message(self, format, *args):
        pass


class _LargeResponseHandler(BaseHTTPRequestHandler):
    """Responde com corpo > 64 KB."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        # 70 KB de dados
        large_data = "x" * (70 * 1024)
        self.wfile.write(json.dumps({"ok": True, "data": large_data}).encode())

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        large_data = "x" * (70 * 1024)
        self.wfile.write(json.dumps({"ok": True, "data": large_data}).encode())

    def log_message(self, format, *args):
        pass


class _SlowHandler(BaseHTTPRequestHandler):
    """Responde após delay para testar timeout."""

    def do_GET(self):
        time.sleep(2)  # maior que timeout padrão de 1s nos testes
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode())

    def do_POST(self):
        time.sleep(2)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode())

    def log_message(self, format, *args):
        pass


# ---------------------------------------------------------------------------
# Fixture: servidor HTTP local em thread
# ---------------------------------------------------------------------------


class _TestServer:
    def __init__(self, handler_class, **handler_kwargs):
        self.server = HTTPServer(("127.0.0.1", 0), lambda *a, **kw: handler_class(*a, **handler_kwargs, **kw))
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        # Aguarda servidor ficar pronto
        time.sleep(0.1)

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)


@pytest.fixture
def ok_server():
    s = _TestServer(_OKHandler)
    yield s
    s.shutdown()


@pytest.fixture
def fail_server():
    s = _TestServer(_FailHandler)
    yield s
    s.shutdown()


@pytest.fixture
def http_401_server():
    s = _TestServer(_HTTPErrorHandler, status_code=401)
    yield s
    s.shutdown()


@pytest.fixture
def http_500_server():
    s = _TestServer(_HTTPErrorHandler, status_code=500)
    yield s
    s.shutdown()


@pytest.fixture
def not_json_server():
    s = _TestServer(_NotJSONHandler)
    yield s
    s.shutdown()


@pytest.fixture
def not_dict_server():
    s = _TestServer(_NotDictHandler, payload=["array", "not", "object"])
    yield s
    s.shutdown()


@pytest.fixture
def large_response_server():
    s = _TestServer(_LargeResponseHandler)
    yield s
    s.shutdown()


@pytest.fixture
def timeout_server():
    s = _TestServer(_SlowHandler)
    yield s
    s.shutdown()


# ---------------------------------------------------------------------------
# Testes do _request REAL
# ---------------------------------------------------------------------------


def test_request_real_sucesso_get_battery(ok_server):
    """_request real: GET /battery retorna payload ok=True."""
    client = PhoneClient(ok_server.base_url, token="test-token", timeout=5.0)
    res = client.get_battery()
    assert res["ok"] is True
    assert res["level"] == 85
    assert res["charging"] is False


def test_request_real_sucesso_speak(ok_server):
    """_request real: POST /tts/speak com body correto."""
    client = PhoneClient(ok_server.base_url, token="test-token", timeout=5.0)
    res = client.speak("Olá mundo")
    assert res["ok"] is True
    assert res["spoken"] is True


def test_request_real_conexao_recusada():
    """_request real: conexão recusada (servidor não roda) → erro amigável."""
    client = PhoneClient("http://127.0.0.1:9999", token="SEGREDO123", timeout=1.0)
    res = client.get_battery()
    assert res["ok"] is False
    assert "conectar" in res["error"].lower() or "rede" in res["error"].lower() or "tempo" in res["error"].lower()
    # Token não vaza
    assert "SEGREDO123" not in res["error"]


def test_request_real_timeout(timeout_server):
    """_request real: timeout → erro amigável."""
    client = PhoneClient(timeout_server.base_url, token="test-token", timeout=0.5)
    res = client.get_battery()
    assert res["ok"] is False
    assert "tempo esgotado" in res["error"].lower() or "timeout" in res["error"].lower()
    # Token não vaza
    assert "test-token" not in res["error"]


def test_request_real_http_401(http_401_server, caplog):
    """_request real: HTTP 401 → erro amigável (código pode aparecer no erro ao usuário, mas token não)."""
    client = PhoneClient(http_401_server.base_url, token="test-token", timeout=5.0)
    with caplog.at_level(logging.WARNING, logger="davios.phone"):
        res = client.get_battery()
    assert res["ok"] is False
    assert "erro" in res["error"].lower()
    # Token não vaza no erro nem nos logs
    assert "test-token" not in res["error"]
    for rec in caplog.records:
        assert "test-token" not in rec.message


def test_request_real_http_500(http_500_server, caplog):
    """_request real: HTTP 500 → erro amigável (código pode aparecer no erro ao usuário, mas token não)."""
    client = PhoneClient(http_500_server.base_url, token="test-token", timeout=5.0)
    with caplog.at_level(logging.WARNING, logger="davios.phone"):
        res = client.get_battery()
    assert res["ok"] is False
    assert "erro" in res["error"].lower()
    assert "test-token" not in res["error"]
    for rec in caplog.records:
        assert "test-token" not in rec.message


def test_request_real_nao_json(not_json_server, caplog):
    """_request real: resposta não-JSON → erro amigável."""
    client = PhoneClient(not_json_server.base_url, token="test-token", timeout=5.0)
    with caplog.at_level(logging.WARNING, logger="davios.phone"):
        res = client.get_battery()
    assert res["ok"] is False
    assert "inválida" in res["error"].lower() or "inválido" in res["error"].lower()
    assert "test-token" not in res["error"]
    for rec in caplog.records:
        assert "test-token" not in rec.message


def test_request_real_json_nao_objeto(not_dict_server, caplog):
    """_request real: JSON que não é objeto (array) → erro amigável."""
    client = PhoneClient(not_dict_server.base_url, token="test-token", timeout=5.0)
    with caplog.at_level(logging.WARNING, logger="davios.phone"):
        res = client.get_battery()
    assert res["ok"] is False
    assert "inválida" in res["error"].lower() or "inválido" in res["error"].lower()
    assert "test-token" not in res["error"]
    for rec in caplog.records:
        assert "test-token" not in rec.message


def test_request_real_resposta_grande(large_response_server):
    """_request real: resposta > 64 KB → erro amigável (limite de leitura)."""
    client = PhoneClient(large_response_server.base_url, token="test-token", timeout=5.0)
    res = client.get_battery()
    assert res["ok"] is False
    assert "64 kb" in res["error"].lower() or "64kb" in res["error"].lower() or "excede" in res["error"].lower()
    assert "test-token" not in res["error"]


def test_base_url_scheme_invalido():
    """base_url com esquema inválido (ftp://, ws://, etc.) → erro amigável."""
    for bad_url in ["ftp://127.0.0.1:8080", "ws://127.0.0.1:8080", "file:///tmp"]:
        client = PhoneClient(bad_url, token="t", timeout=1.0)
        res = client.get_battery()
        assert res["ok"] is False
        assert "http" in res["error"].lower() or "https" in res["error"].lower()
    # string vazia cai na validação de "não configurada"
    client = PhoneClient("", token="t", timeout=1.0)
    res = client.get_battery()
    assert res["ok"] is False
    assert "configurada" in res["error"].lower()


def test_token_nao_vaza_em_nenhum_caso(caplog):
    """Token nunca aparece em erros nem logs em nenhum cenário de falha."""
    client = PhoneClient("http://127.0.0.1:9999", token="SEGREDO123", timeout=0.1)
    with caplog.at_level(logging.WARNING, logger="davios.phone"):
        res = client.get_battery()

    assert res["ok"] is False
    assert "SEGREDO123" not in res["error"]
    for rec in caplog.records:
        assert "SEGREDO123" not in rec.message