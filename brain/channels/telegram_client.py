"""Cliente HTTP mínimo da API do Telegram (long polling + envio).

Decisões vinculantes honradas aqui:
  D1  transporte HTTP direto (urllib), sem dependências externas.
  D5  o token vive SOMENTE dentro do cliente; nunca em repr/str/log, e a URL
      (que contém o token) NUNCA é logada.
  D6  mensagens enviadas como TEXTO PURO (sem parse_mode).

O cliente é testável sem rede: ``transport`` é um callable injetável
``(url, body_bytes, timeout) -> bytes``. Por padrão usa urllib.request.urlopen.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from brain.channels.base import IncomingMessage

logger = logging.getLogger("davios.channels.telegram_client")

API_BASE = "https://api.telegram.org/bot"
# Timeout das chamadas de ENVIO (não long-polling).
SEND_TIMEOUT_SECONDS = 30.0
# Folga somada ao timeout de long polling para o urlopen não cortar antes do
# servidor responder (timeout do urlopen = timeout de long polling + 10).
LONG_POLL_GRACE_SECONDS = 10.0

REDACTED = "***"

# Validação de formato do token do Telegram: <bot_id>:<segredo>. Não confirma
# que o token é real — só barra lixo óbvio antes de any I/O. [0-9] (não \d,
# que aceitaria dígitos Unicode) e fullmatch (ancora a string inteira, então um
# "\n" ao final não é tolerado).
_TOKEN_RE = re.compile(r"[0-9]{5,}:[A-Za-z0-9_-]{20,}")

Transport = Callable[[str, bytes, float], bytes]


class TelegramAPIError(Exception):
    """Erro de uma chamada à API do Telegram.

    A mensagem NUNCA contém token nem URL (o cliente redige antes de criar o
    erro). ``error_code`` e ``retry_after`` são preenchidos quando o Telegram
    os informa (ex.: HTTP 429 com ``parameters.retry_after``).
    """

    def __init__(self, message, error_code=None, retry_after=None):
        super().__init__(message)
        self.error_code = error_code
        self.retry_after = retry_after


def redact(text, token):
    """Substitui TODAS as ocorrências de ``token`` por ``"***"`` em ``text``."""
    safe = text if isinstance(text, str) else str(text)
    if not token:
        return safe
    return safe.replace(token, REDACTED)



class TelegramClient:
    """Camada fina sobre a API HTTP do Telegram, orientada a injeção.

    Somente esta classe conhece o token; ele nunca escapa para repr/str/log e
    a URL (que o embute) jamais é registrada.
    """

    def __init__(self, token: str, transport: Optional[Transport] = None) -> None:
        # D5: o token só existe aqui dentro. Strip tolera "\n"/espaços ao colar
        # de uma var de ambiente. A mensagem de erro é GENÉRICA e NUNCA contém
        # o token (nem o valor rejeitado).
        if not isinstance(token, str):
            raise ValueError("token do Telegram em formato invalido")
        cleaned = token.strip()
        if not _TOKEN_RE.fullmatch(cleaned):
            raise ValueError("token do Telegram em formato invalido")
        self._token = cleaned
        self._transport = transport

    def __repr__(self) -> str:
        # Nunca expor o token aqui.
        return "TelegramClient(token=%r)" % REDACTED

    __str__ = __repr__

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _url(self, method: str) -> str:
        # Contém o token: NUNCA passe para logging.
        return "%s%s/%s" % (API_BASE, self._token, method)

    def _default_transport(
        self,
        url: str,
        body_bytes: bytes,
        timeout: float,
        content_type: str = "application/json",
    ) -> bytes:
        request = urllib.request.Request(
            url,
            data=body_bytes,
            headers={"Content-Type": content_type},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def _api_error(self, data: Any) -> TelegramAPIError:
        error_code = None
        retry_after = None
        description = ""
        if isinstance(data, dict):
            error_code = data.get("error_code")
            parameters = data.get("parameters")
            if isinstance(parameters, dict):
                retry_after = parameters.get("retry_after")
            description = data.get("description") or ""
        message = redact(
            description or "Erro desconhecido da API do Telegram.", self._token
        )
        return TelegramAPIError(message, error_code=error_code, retry_after=retry_after)

    def _error_from_http(self, exc: "urllib.error.HTTPError") -> TelegramAPIError:
        error_code = getattr(exc, "code", None)
        retry_after = None
        description = ""
        try:
            data = json.loads(exc.read().decode("utf-8"))
        except Exception:  # noqa: BLE001 - corpo pode não ser JSON
            data = None
        if isinstance(data, dict):
            if isinstance(data.get("error_code"), int):
                error_code = data["error_code"]
            parameters = data.get("parameters")
            if isinstance(parameters, dict):
                retry_after = parameters.get("retry_after")
            description = data.get("description") or ""
        if not description:
            description = "HTTP %s" % error_code
        return TelegramAPIError(
            redact(description, self._token),
            error_code=error_code,
            retry_after=retry_after,
        )

    def _decode(self, raw: bytes) -> Any:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise TelegramAPIError("Resposta do Telegram nao e JSON valido.") from None
        if not isinstance(data, dict) or not data.get("ok", False):
            raise self._api_error(data)
        return data.get("result")

    def _call(
        self,
        method: str,
        payload: dict,
        transport_timeout: float,
        success_level: int = logging.INFO,
    ) -> Any:
        url = self._url(method)  # NUNCA registrar `url` (contém o token).
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        transport = self._transport or self._default_transport
        try:
            raw = transport(url, body, transport_timeout)
        except urllib.error.HTTPError as exc:
            error = self._error_from_http(exc)
            logger.warning("[TG] %s http_error error_code=%s", method, error.error_code)
            raise error from None
        except urllib.error.URLError as exc:
            reason = redact(str(exc.reason), self._token)
            logger.warning(
                "[TG] %s falha de rede error_type=%s", method, type(exc).__name__
            )
            raise TelegramAPIError(
                "Falha de rede ao chamar o Telegram: %s" % reason
            ) from None
        except (OSError, http.client.HTTPException, ValueError) as exc:
            # Cobre timeout na leitura (socket.timeout/TimeoutError), conexão
            # resetada (ConnectionResetError), IncompleteRead (HTTPException) e
            # InvalidURL (ValueError cuja mensagem inclui a URL com o token).
            detail = redact(str(exc), self._token)
            logger.warning("[TG] %s erro de transporte %s", method, type(exc).__name__)
            raise TelegramAPIError(
                "%s: %s" % (type(exc).__name__, detail)
            ) from None
        # Long polling é barulhento: registra em DEBUG. Envios ficam em INFO.
        logger.log(success_level, "[TG] %s ok", method)
        return self._decode(raw)


    # ------------------------------------------------------------------ #
    # API pública
    # ------------------------------------------------------------------ #
    def get_updates(
        self,
        offset: Optional[int] = None,
        timeout: int = 30,
        allowed_updates: Optional[list] = None,
    ) -> list:
        """Long polling de updates. Timeout do urlopen = ``timeout`` + 10."""
        if allowed_updates is None:
            allowed_updates = ["message"]
        payload: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": allowed_updates,
        }
        if offset is not None:
            payload["offset"] = offset
        result = self._call(
            "getUpdates",
            payload,
            timeout + LONG_POLL_GRACE_SECONDS,
            success_level=logging.DEBUG,
        )
        return result or []

    def send_message(self, chat_id: int, text: str) -> Any:
        """Envia texto puro (D6: sem parse_mode)."""
        payload = {"chat_id": chat_id, "text": text}
        return self._call("sendMessage", payload, SEND_TIMEOUT_SECONDS)

    def send_chat_action(self, chat_id: int, action: str = "typing") -> Any:
        payload = {"chat_id": chat_id, "action": action}
        return self._call("sendChatAction", payload, SEND_TIMEOUT_SECONDS)

    def send_voice(self, chat_id: int, voice_path: str, timeout: float = SEND_TIMEOUT_SECONDS) -> Any:
        """Envia um arquivo de áudio como mensagem de voz (``sendVoice``).

        O Telegram espera ``sendVoice`` via ``multipart/form-data`` com os
        campos ``chat_id`` e ``voice`` (arquivo). O arquivo deve ser um áudio
        válido (recomendado OGG Opus). Não exibe o token nem a URL em lugar
        nenhum — ela só trafega no transporte injetado.
        """
        boundary = "----DaviOSVoiceBoundary----"
        with open(voice_path, "rb") as f:
            voice_bytes = f.read()

        parts: list[bytes] = [
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="chat_id"\r\n\r\n',
            str(chat_id).encode("utf-8"),
            b"\r\n",
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="voice"; filename="voice.ogg"\r\n',
            b"Content-Type: audio/ogg\r\n\r\n",
            voice_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        body = b"".join(parts)
        content_type = "multipart/form-data; boundary=%s" % boundary
        return self._call_raw("sendVoice", body, content_type, timeout)

    def _call_raw(
        self,
        method: str,
        body_bytes: bytes,
        content_type: str,
        timeout: float,
    ) -> Any:
        """Chama um método da API com corpo bytes bruto (ex.: multipart).
        A URL (que contém o token) nunca é registrada; erros vêm redigidos.
        """
        url = self._url(method)
        try:
            if self._transport is None:
                raw = self._default_transport(
                    url, body_bytes, timeout, content_type=content_type
                )
            else:
                raw = self._transport(url, body_bytes, timeout)
        except urllib.error.HTTPError as exc:
            error = self._error_from_http(exc)
            logger.warning(
                "[TG] %s http_error error_code=%s",
                method,
                error.error_code,
            )
            raise error from None
        except urllib.error.URLError as exc:
            reason = redact(str(exc.reason), self._token)
            logger.warning(
                "[TG] %s falha de rede error_type=%s",
                method,
                type(exc).__name__,
            )
            raise TelegramAPIError(
                "Falha de rede ao chamar o Telegram: %s" % reason
            ) from None
        except (OSError, http.client.HTTPException, ValueError) as exc:
            detail = redact(str(exc), self._token)
            logger.warning("[TG] %s erro de transporte %s", method, type(exc).__name__)
            raise TelegramAPIError(
                "%s: %s" % (type(exc).__name__, detail)
            ) from None
        logger.log(logging.INFO, "[TG] %s ok", method)
        return self._decode(raw)

    def get_me(self) -> Any:
        """Valida o token contra ``getMe`` (usado por ``run_telegram --check``).

        Reusa ``_call``: a URL (que embute o token) NUNCA é registrada e qualquer
        erro já vem redigido (sem token). Devolve o dict do bot (id, username…).
        Levanta ``TelegramAPIError`` se o token for inválido ou a rede falhar.
        """
        return self._call("getMe", {}, SEND_TIMEOUT_SECONDS)


# --------------------------------------------------------------------------- #
# Funções puras
# --------------------------------------------------------------------------- #
def chunk_text(text: str, limit: int = 4000) -> list:
    """Divide ``text`` em blocos de no máximo ``limit`` caracteres.

    Regras:
      - texto vazio/``None``/só espaços -> lista vazia;
      - nenhum bloco VAZIO e nenhum bloco SÓ-WHITESPACE (``c.strip()`` vazio);
      - nenhum bloco acima de ``limit``;
      - preferência de quebra: parágrafo (\\n\\n) > linha (\\n) > espaço, e só
        então corte no meio de uma palavra.
    """
    if text is None:
        return []
    if not text.strip():
        return []
    if limit <= 0:
        raise ValueError("limit deve ser positivo")

    chunks: list = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        window = remaining[:limit]
        cut = _best_break(window)
        if cut <= 0:
            # Sem quebra "limpa" na janela: corte seco no limite.
            cut = limit
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
        if not remaining:
            break
    # Descarta blocos vazios e blocos que sejam só whitespace (ex.: excesso de
    # "\n" ou espaços gerado pela quebra depois do separador).
    return [c for c in chunks if c and c.strip()]


def _best_break(window: str) -> int:
    """Índice (exclusivo) da melhor quebra dentro de ``window`` (<= limit).

    Retorna 0 se não houver parágrafo/linha/espaço quebrável (aí corta seco).
    A quebra é feita DEPOIS do separador, preservando o conteúdo: com isso
    ``"".join(chunk_text(text)) == text`` para texto sem espaços à esquerda
    perdidos (a quebra nunca descarta caracteres).
    """
    idx = window.rfind("\n\n")  # parágrafo
    if idx != -1:
        return idx + 2
    idx = window.rfind("\n")  # linha
    if idx != -1:
        return idx + 1
    idx = window.rfind(" ")  # espaço
    if idx != -1:
        return idx + 1
    return 0


def extract_messages(updates: list):
    """Extrai mensagens válidas e calcula o próximo offset (D7).

    Aplica D7: só ``chat.type == "private"`` com campo ``message`` contendo
    ``text`` e um ``from``. Grupos, canais, ``edited_message`` e updates sem
    texto são IGNORADOS, mas ainda ASSIM avançam o offset.

    Retorna ``(list[IncomingMessage], next_offset)`` onde ``next_offset`` é o
    maior ``update_id`` + 1, ou ``None`` se ``updates`` estiver vazio.
    """
    messages: list = []
    max_update_id = None
    for update in updates or []:
        if not isinstance(update, dict):
            continue
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            if max_update_id is None or update_id > max_update_id:
                max_update_id = update_id
        message = update.get("message")
        if not isinstance(message, dict):
            continue  # edited_message, channel_post, etc. -> ignora (offset já avançou)
        chat = message.get("chat")
        if not isinstance(chat, dict) or chat.get("type") != "private":
            continue  # D7: só chats privados
        text = message.get("text")
        if not isinstance(text, str) or not text:
            continue  # sem texto (sticker, foto, etc.) -> ignora
        sender = message.get("from")
        if not isinstance(sender, dict) or "id" not in sender:
            continue  # sem remetente identificável -> ignora
        chat_id = chat.get("id")
        sender_id = sender.get("id")
        if not isinstance(chat_id, int) or not isinstance(sender_id, int):
            continue
        messages.append(
            IncomingMessage(sender_id=sender_id, chat_id=chat_id, text=text)
        )

    next_offset = max_update_id + 1 if max_update_id is not None else None
    return messages, next_offset
