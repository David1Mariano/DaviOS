"""AgentProtocol do DaviOS.

Protocolo simple JSON sobre socket local para orquestar multiples agentes.

Formato de mensaje (JSON, una linea por frame):
    {"type": "...", "payload": {...}, "request_id": "..."}

Seguridad:
- Solo escucha en 127.0.0.1 (loopback);
- Maximo tamano de frame (evita DoS por memoria);
- Timeout de conexion e inactividad;
- Handler por tipo; tipos no registrados -> error.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger("davios.agents")

MAX_FRAME_BYTES = 1_000_000  # 1 MB maximo por frame
CONNECTION_TIMEOUT = 30.0
MAX_CONNECTIONS = 8


@dataclass
class AgentResponse:
    """Respuesta estandar de un handler de agente."""

    ok: bool
    payload: dict[str, Any] = None
    error: str = ""


class AgentHandler:
    """Registra handlers para tipos de mensaje y procesa frames."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], AgentResponse]] = {}

    def register(
        self,
        msg_type: str,
        handler: Callable[[dict[str, Any]], AgentResponse],
    ) -> None:
        self._handlers[msg_type] = handler

    def handle(self, frame: dict[str, Any]) -> dict[str, Any]:
        msg_type = frame.get("type")
        request_id = frame.get("request_id", "")
        payload = frame.get("payload", {})
        handler = self._handlers.get(msg_type)
        if handler is None:
            return {
                "type": "error",
                "request_id": request_id,
                "payload": {"message": f"Tipo de mensaje desconocido: {msg_type}"},
            }
        try:
            result = handler(payload)
        except Exception as e:
            logger.warning("[AGENT] handler error: %s", e)
            return {
                "type": "error",
                "request_id": request_id,
                "payload": {"message": f"Error interno del handler: {type(e).__name__}"},
            }
        if result.ok:
            return {
                "type": "ok",
                "request_id": request_id,
                "payload": result.payload or {},
            }
        return {
            "type": "error",
            "request_id": request_id,
            "payload": {"message": result.error or "Error desconocido"},
        }


class AgentFrameError(Exception):
    """Frame invalido (no JSON, demasiado grande, o sin type)."""


def decode_frame(raw: bytes) -> dict[str, Any]:
    """Decodifica un frame completo; lanza AgentFrameError si es invalido."""
    if len(raw) > MAX_FRAME_BYTES:
        raise AgentFrameError("Frame excede el tamano maximo.")
    try:
        frame = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise AgentFrameError(f"JSON invalido: {e}")
    if not isinstance(frame, dict) or "type" not in frame:
        raise AgentFrameError("Frame debe tener campo 'type'.")
    return frame


def encode_frame(frame: dict[str, Any]) -> bytes:
    """Serializa un frame a JSON + newline (framing simple)."""
    return (json.dumps(frame, ensure_ascii=False) + "\n").encode("utf-8")


class AgentServer:
    """Servidor socket JSON en loopback para orquestar agentes externos."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        enabled: bool = False,
        handler: Optional[AgentHandler] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.enabled = enabled
        self.handler = handler or AgentHandler()
        self._server_socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> bool:
        """Levanta el servidor en un hilo de fondo. True si inicia OK."""
        if not self.enabled:
            logger.info("[AGENT] server deshabilitado.")
            return False
        if self._running:
            return True
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((self.host, self.port))
            sock.listen(MAX_CONNECTIONS)
            sock.settimeout(CONNECTION_TIMEOUT)
        except OSError as e:
            logger.warning("[AGENT] no se pudo levantar server: %s", e)
            return False

        self._server_socket = sock
        self._running = True
        self._thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="davios-agent-server"
        )
        self._thread.start()
        logger.info("[AGENT] server activo en %s:%d", self.host, self.port)
        return True

    def stop(self) -> None:
        self._running = False
        if self._server_socket is not None:
            try:
                self._server_socket.close()
            except OSError:
                pass
            self._server_socket = None

    def _accept_loop(self) -> None:
        while self._running and self._server_socket is not None:
            try:
                conn, _addr = self._server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True,
                name="davios-agent-conn",
            ).start()

    def _handle_connection(self, conn: socket.socket) -> None:
        conn.settimeout(CONNECTION_TIMEOUT)
        try:
            with conn:
                data = conn.recv(MAX_FRAME_BYTES)
                if not data:
                    return
                frame = decode_frame(data.rstrip(b"\n"))
                response = self.handler.handle(frame)
                conn.sendall(encode_frame(response))
        except AgentFrameError as e:
            try:
                conn.sendall(
                    encode_frame(
                        {"type": "error", "request_id": "", "payload": {"message": str(e)}}
                    )
                )
            except OSError:
                pass
        except OSError as e:
            logger.warning("[AGENT] error de conexion: %s", e)
        finally:
            try:
                conn.close()
            except OSError:
                pass