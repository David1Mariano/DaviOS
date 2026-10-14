"""Testes do phone_server.py sem unittest.mock.

Usa ThreadingHTTPServer em porta efêmera com runner falso.
"""
import json
import logging
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phone_server.phone_server import (
    PhoneRequestHandler,
    default_runner,
    TIMEOUT_LOCATION,
    TIMEOUT_DEFAULT,
)

# ---------------------------------------------------------------------------
# Runner falso para testes
# ---------------------------------------------------------------------------


class FakeRunner:
    """Runner que pode ser configurado para retornar diferentes resultados."""

    def __init__(self):
        self.callback = None

    def set_callback(self, callback):
        self.callback = callback

    def __call__(self, cmd, timeout):
        if self.callback:
            return self.callback(cmd, timeout)
        # padrão: sucesso com saída vazia
        return 0, b"", b""


# ---------------------------------------------------------------------------
# Fixture: servidor HTTP local em thread
# ---------------------------------------------------------------------------


class _TestServer:
    def __init__(self, handler_class, token="test-token", **handler_kwargs):
        # Define a variável de ambiente para o token apenas se fornecido
        if token is not None:
            os.environ["DAVIOS_PHONE_TOKEN"] = token
        self.token = token
        self.server = HTTPServer(
            ("127.0.0.1", 0), lambda *a, **kw: handler_class(*a, **handler_kwargs, **kw)
        )
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)  # aguarda servidor ficar pronto

    def shutdown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
        # Remove a variável de ambiente apenas se nós a definimos
        if self.token is not None and os.environ.get("DAVIOS_PHONE_TOKEN") == self.token:
            del os.environ["DAVIOS_PHONE_TOKEN"]


@pytest.fixture
def server():
    s = _TestServer(PhoneRequestHandler)
    yield s
    s.shutdown()


@pytest.fixture
def no_token_server():
    # Não definimos o token
    s = _TestServer(PhoneRequestHandler, token=None)
    yield s
    s.shutdown()


@pytest.fixture
def wrong_token_server():
    # Definimos um token válido no ambiente, mas usaremos um token errado na requisição
    s = _TestServer(PhoneRequestHandler, token="correct-token")
    yield s
    s.shutdown()


# ---------------------------------------------------------------------------
# Fixture para fornecer um runner falso reutilizável
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_runner():
    return FakeRunner()


# ---------------------------------------------------------------------------
# Helpers para fazer requisições
# ---------------------------------------------------------------------------


def _make_request(method, url, token, body=None, headers=None):
    """Faz uma requisição HTTP usando urllib (para evitar dependências externas)."""
    import urllib.request
    import urllib.error

    req_headers = {}
    if token:
        req_headers["Authorization"] = f"Bearer {token}"
    if headers:
        req_headers.update(headers)

    data = None
    if body is not None:
        if isinstance(body, dict):
            data = json.dumps(body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        else:
            data = body

    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.getcode()
            resp_headers = dict(resp.getheaders())
            resp_body = resp.read()
            return status, resp_headers, resp_body
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()
    except urllib.error.URLError as e:
        # retorna status 0 para indicar erro de rede
        return 0, {}, str(e).encode()


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


def test_server_nao_inicia_sem_token():
    """Se DAVIOS_PHONE_TOKEN não estiver definido, o servidor não inicia."""
    # Remove a variável de ambiente se existir
    old_token = os.environ.get("DAVIOS_PHONE_TOKEN")
    if "DAVIOS_PHONE_TOKEN" in os.environ:
        del os.environ["DAVIOS_PHONE_TOKEN"]
    try:
        # Importa e tenta iniciar o servidor (mas não escuta porta real)
        from phone_server.phone_server import run_server

        # Simulamos que a porta está ocupada para evitar bloqueio, mas vamos verificar a mensagem
        # Como não podemos realmente iniciar o servidor em uma porta livre sem bloquear,
        # vamos verificar que a função retorna código de erro quando token falta.
        # No entanto, a função run_server apenas imprime e retorna 1.
        # Vamos capturar o saída.
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result = run_server()
        output = f.getvalue()
        assert result == 1
        assert "DAVIOS_PHONE_TOKEN" in output
    finally:
        if old_token is not None:
            os.environ["DAVIOS_PHONE_TOKEN"] = old_token


def test_retorna_401_sem_token(no_token_server):
    """Sem token no header, retorna 401 com corpo vazio."""
    status, _, body = _make_request(
        "GET", f"{no_token_server.base_url}/battery", token=None
    )
    assert status == 401
    # corpo vazio
    assert len(body) == 0


def test_retorna_401_token_errado(wrong_token_server):
    """Token errado retorna 401 com corpo vazio."""
    status, _, body = _make_request(
        "GET", f"{wrong_token_server.base_url}/battery", token="wrong-token"
    )
    assert status == 401
    # corpo vazio
    assert len(body) == 0


def test_battery_sucesso(server, fake_runner):
    """Endpoint /battery com sucesso."""
    # Configura o runner falso para retornar saída válida
    def battery_runner(cmd, timeout):
        assert cmd == ["termux-battery-status"]
        return 0, json.dumps({"percentage": 85, "status": "CHARGING"}).encode(), b""

    fake_runner.set_callback(battery_runner)
    PhoneRequestHandler.runner = fake_runner

    status, headers, body = _make_request(
        "GET", f"{server.base_url}/battery", token=server.token
    )
    assert status == 200
    assert headers.get("Content-Type") == "application/json"
    data = json.loads(body)
    assert data["ok"] is True
    assert data["level"] == 85
    assert data["charging"] is True


def test_battery_comando_falha(server, fake_runner):
    """Quando o comando termux-battery-status falha, retorna erro amigável."""
    def fail_runner(cmd, timeout):
        return 1, b"", b"algum erro"

    fake_runner.set_callback(fail_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "GET", f"{server.base_url}/battery", token=server.token
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Falha ao obter bateria" in data["error"]


def test_battery_saida_nao_json(server, fake_runner):
    """Quando a saída não é JSON, retorna erro amigável."""
    def non_json_runner(cmd, timeout):
        return 0, b"isto nao e json", b""

    fake_runner.set_callback(non_json_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "GET", f"{server.base_url}/battery", token=server.token
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Resposta inválida do comando de bateria" in data["error"]


def test_battery_saida_vazia(server, fake_runner):
    """Quando a saída é vazia, trata como não JSON."""
    def empty_runner(cmd, timeout):
        return 0, b"", b""

    fake_runner.set_callback(empty_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "GET", f"{server.base_url}/battery", token=server.token
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Resposta inválida do comando de bateria" in data["error"]


def test_location_sucesso(server, fake_runner):
    """Endpoint /location com sucesso."""
    def location_runner(cmd, timeout):
        assert cmd == ["termux-location", "-p", "network", "-r", "once"]
        return 0, json.dumps({"latitude": -12.34, "longitude": -56.78, "accuracy": 10.0}).encode(), b""

    fake_runner.set_callback(location_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "GET", f"{server.base_url}/location", token=server.token
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert abs(data["lat"] - (-12.34)) < 0.001
    assert abs(data["lon"] - (-56.78)) < 0.001
    assert abs(data["accuracy"] - 10.0) < 0.001


def test_location_saida_nao_json_playstore(server, fake_runner):
    """Simula o comportamento do Termux da Play Store: saída não JSON."""
    def playstore_runner(cmd, timeout):
        # O comando sai com código 0 e imprime essa mensagem
        return 0, b"Termux:API is not yet available on Google Play", b""

    fake_runner.set_callback(playstore_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "GET", f"{server.base_url}/location", token=server.token
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    # Mensagem curta em português
    assert data["error"] == "Localização indisponível"


def test_location_comando_timeout(server, fake_runner):
    """Quando o comando demora demais, retorna erro de timeout."""
    def timeout_runner(cmd, timeout):
        # simula timeout do comando
        return -1, b"", b"timeout"

    fake_runner.set_callback(timeout_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "GET", f"{server.base_url}/location", token=server.token
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    # O servidor traduz o timeout do comando como erro genérico?
    # No nosso código, não temos um tratamento específico para timeout do comando;
    # o runner retorna (-1, b"", b"timeout") e então o código != 0, então cai no erro genérico.
    # Vamos verificar que o erro é genérico.
    assert "Falha ao obter" in data["error"] or "Localização indisponível" in data["error"]


def test_device_sucesso(server, fake_runner):
    """Endpoint /device com sucesso."""
    def device_runner(cmd, timeout):
        assert cmd == ["getprop", "ro.product.model", "ro.build.version.release"]
        # Simula duas linhas de saída
        return 0, b"Galaxy A30S\n11\n", b""

    fake_runner.set_callback(device_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "GET", f"{server.base_url}/device", token=server.token
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert data["model"] == "Galaxy A30S"
    assert data["android_version"] == "11"


def test_device_saida_nao_json(server, fake_runner):
    """Quando a saída do getprop não é como esperado (não duas linhas)."""
    def bad_device_runner(cmd, timeout):
        return 0, b"soh uma linha\n", b""

    fake_runner.set_callback(bad_device_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "GET", f"{server.base_url}/device", token=server.token
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Falha ao obter informações do dispositivo" in data["error"]


def test_speak_sucesso(server, fake_runner):
    """Endpoint /tts/speak com sucesso."""
    def speak_runner(cmd, timeout):
        # cmd deve ser ["termux-tts-speak", <texto>]
        assert cmd[0] == "termux-tts-speak"
        assert isinstance(cmd[1], str)
        # Sucesso: saída vazia e código 0
        return 0, b"", b""

    fake_runner.set_callback(speak_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/tts/speak",
        token=server.token,
        body={"text": "Olá mundo"},
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert data["spoken"] is True


def test_speak_texto_vazio(server, fake_runner):
    """Texto vazio após limpeza retorna erro."""
    fake_runner.set_callback(lambda cmd, timeout: (0, b"", b""))  # dummy, não usado
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/tts/speak",
        token=server.token,
        body={"text": "   \n\t  "},
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Texto vazio após limpeza" in data["error"]


def test_speak_texto_longo(server, fake_runner):
    """Texto maior que 300 caracteres retorna erro."""
    long_text = "x" * 301
    fake_runner.set_callback(lambda cmd, timeout: (0, b"", b""))  # dummy
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/tts/speak",
        token=server.token,
        body={"text": long_text},
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Texto muito longo" in data["error"]


def test_speak_texto_com_controle(server, fake_runner):
    """Texto com caracteres de controle (exceto \n,\t,\r) deve ser limpo."""
    def speak_runner(cmd, timeout):
        # Verifica que o caractere de controle foi removido
        assert cmd[0] == "termux-tts-speak"
        # O texto original tinha um \x01, que deve ser removido
        assert cmd[1] == "olá"  # porque \x01 foi removido e depois strip
        return 0, b"", b""

    fake_runner.set_callback(speak_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/tts/speak",
        token=server.token,
        body={"text": "\x01olá\x02"},
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert data["spoken"] is True


def test_speak_texto_com_aspas_e_pontovigula(server, fake_runner):
    """Texto com aspas e ponto e vírgula deve ser passado como argumento literal."""
    def speak_runner(cmd, timeout):
        assert cmd[0] == "termux-tts-speak"
        # O texto deve ser passado exatamente como fornecido (após validação)
        assert cmd[1] == 'texto "com" aspas; e ponto;e vírgula'
        return 0, b"", b""

    fake_runner.set_callback(speak_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/tts/speak",
        token=server.token,
        body={"text": 'texto "com" aspas; e ponto;e vírgula'},
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert data["spoken"] is True


def test_vibrate_sucesso(server, fake_runner):
    """Endpoint /vibrate com sucesso."""
    def vibrate_runner(cmd, timeout):
        assert cmd[0] == "termux-vibrate"
        assert cmd[1] == "100"
        return 0, b"", b""

    fake_runner.set_callback(vibrate_runner)
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/vibrate",
        token=server.token,
        body={"duration": 100},
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is True
    assert data["vibrated"] is True


def test_vibrate_fora_limite_minimo(server, fake_runner):
    """Duração menor que 50 ms retorna erro."""
    fake_runner.set_callback(lambda cmd, timeout: (0, b"", b""))  # dummy
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/vibrate",
        token=server.token,
        body={"duration": 40},
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Duração deve ser entre 50 e 2000 ms" in data["error"]


def test_vibrate_fora_limite_maximo(server, fake_runner):
    """Duração maior que 2000 ms retorna erro."""
    fake_runner.set_callback(lambda cmd, timeout: (0, b"", b""))  # dummy
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/vibrate",
        token=server.token,
        body={"duration": 2500},
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Duração deve ser entre 50 e 2000 ms" in data["error"]


def test_vibrate_nao_inteiro(server, fake_runner):
    """Duração não inteira retorna erro."""
    fake_runner.set_callback(lambda cmd, timeout: (0, b"", b""))  # dummy
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/vibrate",
        token=server.token,
        body={"duration": 100.5},
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Duração deve ser inteiro (ms)" in data["error"]


def test_corpo_grande(server, fake_runner):
    """Corpo de requisição maior que 4 KB retorna erro."""
    # corpo de 5 KB
    big_body = {"text": "x" * 5000}
    fake_runner.set_callback(lambda cmd, timeout: (0, b"", b""))  # dummy
    PhoneRequestHandler.runner = fake_runner

    status, _, body = _make_request(
        "POST",
        f"{server.base_url}/tts/speak",
        token=server.token,
        body=big_body,
    )
    assert status == 200
    data = json.loads(body)
    assert data["ok"] is False
    assert "Corpo muito grande" in data["error"]


def test_rota_inexistente(server):
    """Rota inexistente retorna 404."""
    status, _, body = _make_request(
        "GET", f"{server.base_url}/foo", token=server.token
    )
    assert status == 404


def test_metodo_errado(server):
    """Método errado (ex: PUT em /battery) retorna 405."""
    status, _, body = _make_request(
        "PUT", f"{server.base_url}/battery", token=server.token
    )
    assert status == 405


def test_token_nao_vaza_em_logs(server, caplog):
    """Token não aparece nos logs em nenhum cenário."""
    # Fazemos uma requisição que irá gerar log (qualquer uma)
    status, _, body = _make_request(
        "GET", f"{server.base_url}/battery", token=server.token
    )
    # Verifica que o token não está em nenhuma mensagem de log
    for record in caplog.records:
        assert server.token not in record.getMessage()


def test_contrato_com_phone_client_real(server, fake_runner):
    """Teste de contrato: subir o servidor com runner falso e chamar com o PhoneClient REAL."""
    from brain.phone_client import PhoneClient

    # Configuramos o runner falso para simular respostas reais do Termux
    def battery_runner(cmd, timeout):
        assert cmd == ["termux-battery-status"]
        return 0, json.dumps({"percentage": 85, "status": "CHARGING"}).encode(), b""

    def location_runner(cmd, timeout):
        assert cmd == ["termux-location", "-p", "network", "-r", "once"]
        return 0, json.dumps({"latitude": -12.34, "longitude": -56.78, "accuracy": 10.0}).encode(), b""

    def device_runner(cmd, timeout):
        assert cmd == ["getprop", "ro.product.model", "ro.build.version.release"]
        return 0, b"Galaxy A30S\n11\n", b""

    def speak_runner(cmd, timeout):
        assert cmd[0] == "termux-tts-speak"
        assert isinstance(cmd[1], str)
        return 0, b"", b""

    def vibrate_runner(cmd, timeout):
        assert cmd[0] == "termux-vibrate"
        assert cmd[1] == "200"
        return 0, b"", b""

    # Definimos o runner do handler para um que pode alternar com base no comando
    def runner_factory(cmd, timeout):
        if cmd == ["termux-battery-status"]:
            return battery_runner(cmd, timeout)
        if cmd == ["termux-location", "-p", "network", "-r", "once"]:
            return location_runner(cmd, timeout)
        if cmd == ["getprop", "ro.product.model", "ro.build.version.release"]:
            return device_runner(cmd, timeout)
        if cmd[0] == "termux-tts-speak":
            return speak_runner(cmd, timeout)
        if cmd[0] == "termux-vibrate":
            return vibrate_runner(cmd, timeout)
        return 0, b"", b""

    fake_runner.set_callback(runner_factory)
    PhoneRequestHandler.runner = fake_runner

    # Agora criamos o PhoneClient real e chamamos cada método
    client = PhoneClient(server.base_url, token=server.token, timeout=5.0)

    # get_battery
    res = client.get_battery()
    assert res["ok"] is True
    assert res["level"] == 85
    assert res["charging"] is True

    # get_location
    res = client.get_location()
    assert res["ok"] is True
    assert abs(res["lat"] - (-12.34)) < 0.001
    assert abs(res["lon"] - (-56.78)) < 0.001
    assert abs(res["accuracy"] - 10.0) < 0.001

    # get_device_info
    res = client.get_device_info()
    assert res["ok"] is True
    assert res["model"] == "Galaxy A30S"
    assert res["android_version"] == "11"

    # speak
    res = client.speak("Olá mundo")
    assert res["ok"] is True
    assert res["spoken"] is True

    # vibrate
    res = client.vibrate(200)
    assert res["ok"] is True
    assert res["vibrated"] is True