#!/usr/bin/env python3
"""Servidor HTTP simples para expor APIs do Termux:API.

Variáveis de ambiente:
- DAVIOS_PHONE_TOKEN: token Bearer obrigatório. Se ausente, o servidor não inicia.
- DAVIOS_PHONE_HOST: host para bind (padrão: 127.0.0.1)
- DAVIOS_PHONE_PORT: porta para bind (padrão: 8765)

Endpoints (todos requerem header Authorization: Bearer <token>):
- GET /battery
- GET /location
- GET /device
- POST /tts/speak
- POST /vibrate

Respostas JSON padronizadas:
  Sucesso: {"ok": true, ...dados específicos...}
  Erro:  {"ok": false, "error": "mensagem curta em português"}

Nunca vaza token, stack trace ou caminhos internos.
"""
import json
import os
import subprocess
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import time
from typing import Callable, List, Optional, Tuple, Dict
import hmac  # importado aqui para uso em _check_token
import logging  # para logging de erros inesperados

# Timeouts dos comandos (em segundos)
TIMEOUT_LOCATION = 8  # termux-location pode demorar
TIMEOUT_DEFAULT = 5   # demais comandos

# Allowlist de comandos e seus argumentos fixos
COMMANDS = {
    "battery": ["termux-battery-status"],
    "location": ["termux-location", "-p", "network", "-r", "once"],
    "device": ["getprop", "ro.product.model", "ro.build.version.release"],
    # Para speak e vibrate, os argumentos são montados dinamicamente
}

# Função de executor injetável (para testes)
RunnerType = Callable[[List[str], int], Tuple[int, bytes, bytes]]


def default_runner(cmd: List[str], timeout: int) -> Tuple[int, bytes, bytes]:
    """Executa o comando com subprocess.run e retorna (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=False,  # retorna bytes
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, b"", b"timeout"
    except Exception as e:
        return -1, b"", str(e).encode()


def _json_response(
    data: dict, status: int = 200
) -> Tuple[int, dict[str, str], bytes]:
    """Prepara resposta JSON."""
    body = json.dumps(data).encode("utf-8")
    return status, {"Content-Type": "application/json"}, body


def _error_response(msg: str, status: int = 200) -> Tuple[int, dict[str, str], bytes]:
    """Prepara resposta de erro padronizada."""
    return _json_response({"ok": False, "error": msg}, status)


def _check_token(headers) -> bool:
    """Verifica se o header Authorization contém o token correto."""
    token_from_env = os.environ.get("DAVIOS_PHONE_TOKEN")
    if not token_from_env:
        # Se o token não está definido no ambiente, nenhum token é válido.
        return False
    auth = headers.get("Authorization")
    if not auth:
        return False
    # Espera "Bearer <token>"
    parts = auth.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    token = parts[1]
    return hmac.compare_digest(token, token_from_env)


class PhoneRequestHandler(BaseHTTPRequestHandler):
    """Handler para as requisições HTTP do servidor de telefone."""

    # Injetável para testes
    runner: RunnerType = default_runner

    def __init__(self, *args, commands: Optional[Dict[str, List[str]]] = None, runner: Optional[RunnerType] = None, **kwargs):
        # Se commands for None, usa a allowlist padrão
        self.commands = commands if commands is not None else COMMANDS
        # Se runner for None, usa o atributo de classe (que pode ter sido sobrescrito por testes)
        self.runner = runner if runner is not None else type(self).runner
        super().__init__(*args, **kwargs)

    # Endpoints permitidos e seus métodos
    _endpoints = {
        "/battery": {"GET"},
        "/location": {"GET"},
        "/device": {"GET"},
        "/tts/speak": {"POST"},
        "/vibrate": {"POST"},
    }

    def do_GET(self):
        self._handle_request("GET")

    def do_POST(self):
        self._handle_request("POST")

    # Para métodos não implementados, retornamos 405
    def do_PUT(self):
        self._handle_request("PUT")

    def do_DELETE(self):
        self._handle_request("DELETE")

    def do_HEAD(self):
        self._handle_request("HEAD")

    def do_OPTIONS(self):
        self._handle_request("OPTIONS")

    def do_PATCH(self):
        self._handle_request("PATCH")

    def _handle_request(self, method: str):
        try:
            # Verifica token
            if not _check_token(self.headers):
                # Sem token ou token inválido: retorna 401 com corpo vazio
                self.send_response(401)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            # Rota e método
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            # Verifica se o caminho é um endpoint conhecido
            if path not in self._endpoints:
                self.send_error(404)
                return

            # Verifica se o método é permitido para esse endpoint
            if method not in self._endpoints[path]:
                self.send_error(405)
                return

            # Agora podemos tratar o método e caminho específicos
            if method == "GET" and path == "/battery":
                self._handle_battery()
            elif method == "GET" and path == "/location":
                self._handle_location()
            elif method == "GET" and path == "/device":
                self._handle_device()
            elif method == "POST" and path == "/tts/speak":
                self._handle_speak()
            elif method == "POST" and path == "/vibrate":
                self._handle_vibrate()
            else:
                # Não deve chegar aqui por causa das verificações acima
                self.send_error(500)
        except Exception as e:
            # Log apenas o tipo da exceção (sem token nem corpo)
            logging.warning("[PHONE] Erro inesperado em %s: %s", path, type(e).__name__)
            self._send_json_error("Erro interno")

    def _handle_battery(self):
        code, stdout, stderr = self.runner(self.commands["battery"], TIMEOUT_DEFAULT)
        if code != 0:
            self._send_json_error("Falha ao obter bateria")
            return
        try:
            data = json.loads(stdout.decode("utf-8"))
            percentage = int(data.get("percentage", 0))
            status = data.get("status", "")
            charging = status in ("CHARGING", "FULL")
            self._send_json_ok({"level": percentage, "charging": charging})
        except (json.JSONDecodeError, KeyError, ValueError):
            self._send_json_error("Resposta inválida do comando de bateria")

    def _handle_location(self):
        code, stdout, stderr = self.runner(self.commands["location"], TIMEOUT_LOCATION)
        if code != 0:
            self._send_json_error("Localização indisponível")
            return
        try:
            data = json.loads(stdout.decode("utf-8"))
            lat = float(data.get("latitude", 0.0))
            lon = float(data.get("longitude", 0.0))
            acc = float(data.get("accuracy", 0.0))
            self._send_json_ok({"lat": lat, "lon": lon, "accuracy": acc})
        except (json.JSONDecodeError, KeyError, ValueError):
            self._send_json_error("Localização indisponível")

    def _handle_device(self):
        code, stdout, stderr = self.runner(self.commands["device"], TIMEOUT_DEFAULT)
        if code != 0:
            self._send_json_error("Falha ao obter informações do dispositivo")
            return
        try:
            # stdout contém duas linhas: model e version
            lines = stdout.decode("utf-8").strip().splitlines()
            if len(lines) < 2:
                raise ValueError("Saída inesperada")
            model = lines[0].strip()
            android_version = lines[1].strip()
            self._send_json_ok({"model": model, "android_version": android_version})
        except (json.JSONDecodeError, KeyError, ValueError):
            self._send_json_error("Falha ao obter informações do dispositivo")

    def _handle_speak(self):
        # Lê e limita o corpo a 4 KB
        length = self.headers.get("Content-Length")
        try:
            length = int(length) if length else 0
        except ValueError:
            length = 0
        if length > 4 * 1024:
            self._send_json_error("Corpo muito grande")
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw.decode("utf-8"))
            text = data.get("text", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            self._send_json_error("Requisição inválida")
            return

        # Validações do servidor (não confia apenas no cliente)
        if not isinstance(text, str):
            self._send_json_error("Texto para fala deve ser string")
            return
        # Remove caracteres de controle (exceto \n, \t, \r) conforme PhoneClient
        import re
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        cleaned = cleaned.strip()
        if not cleaned:
            self._send_json_error("Texto vazio após limpeza")
            return
        if len(cleaned) > 300:
            self._send_json_error("Texto muito longo (máx. 300 caracteres)")
            return

        # Executa o comando com o texto limpo como argumento
        cmd = ["termux-tts-speak", cleaned]
        code, stdout, stderr = self.runner(cmd, TIMEOUT_DEFAULT)
        if code != 0:
            # Mesmo que o comando falhe, tratamos como erro genérico
            self._send_json_error("Falha ao executar fala")
            return
        # Para termux-tts-speak, sucesso é saída vazia e código 0
        self._send_json_ok({"spoken": True})

    def _handle_vibrate(self):
        length = self.headers.get("Content-Length")
        try:
            length = int(length) if length else 0
        except ValueError:
            length = 0
        if length > 4 * 1024:
            self._send_json_error("Corpo muito grande")
            return
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            data = json.loads(raw.decode("utf-8"))
            ms = data.get("duration", 0)
        except (json.JSONDecodeError, KeyError, ValueError):
            self._send_json_error("Requisição inválida")
            return

        # Validações do servidor
        if not isinstance(ms, int):
            self._send_json_error("Duração deve ser inteiro (ms)")
            return
        if ms < 50 or ms > 2000:
            self._send_json_error("Duração deve ser entre 50 e 2000 ms")
            return

        cmd = ["termux-vibrate", str(ms)]
        code, stdout, stderr = self.runner(cmd, TIMEOUT_DEFAULT)
        if code != 0:
            self._send_json_error("Falha ao executar vibração")
            return
        self._send_json_ok({"vibrated": True})

    def _send_json_ok(self, data: dict):
        status, headers, body = _json_response({"ok": True, **data})
        self._send_response(status, headers, body)

    def _send_json_error(self, msg: str):
        status, headers, body = _error_response(msg)
        self._send_response(status, headers, body)

    def _send_response(
        self, status: int, headers: dict[str, str], body: bytes
    ):
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Não logar token nem corpo de requisição, apenas método, path, status
        # O log padrão do BaseHTTPRequestHandler já não inclui o corpo, apenas a linha de requisição.
        # A linha de requisição contém o método, o caminho e a versão do HTTP, mas não o token.
        # Portanto, podemos usar o log padrão.
        super().log_message(format, *args)


def run_server():
    """Inicia o servidor HTTP."""
    # Verifica se o token está definido
    token = os.environ.get("DAVIOS_PHONE_TOKEN")
    if not token:
        print("Erro: variável de ambiente DAVIOS_PHONE_TOKEN não definida")
        print("Defina o token e reinicie o servidor.")
        return 1

    host = os.environ.get("DAVIOS_PHONE_HOST", "127.0.0.1")
    try:
        port = int(os.environ.get("DAVIOS_PHONE_PORT", "8765"))
    except ValueError:
        port = 8765

    server = HTTPServer((host, port), PhoneRequestHandler)
    print(f"Servidor iniciado em http://{host}:{port}")
    print("Pressione Ctrl+C para parar.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor interrompido.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    exit(run_server())