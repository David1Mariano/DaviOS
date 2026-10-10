"""Testes do cliente Telegram e das funções puras do canal.

SEM rede real: uma fixture autouse faz urllib.request.urlopen levantar erro se
alguém tentar chamá-lo de verdade. Todo o tráfego é injetado via um
``transport`` falso.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

import brain.channels.telegram_client as tc
from brain.channels.base import IncomingMessage
from brain.channels.telegram_client import (
    TelegramAPIError,
    TelegramClient,
    chunk_text,
    extract_messages,
    redact,
)

TOKEN = "123456:ABCDEF-secret-token"


@pytest.fixture(autouse=True)
def _no_real_network(monkeypatch):
    """Proíbe qualquer urlopen real durante os testes deste módulo."""

    def _boom(*args, **kwargs):  # pragma: no cover - não deve rodar
        raise AssertionError("urllib.request.urlopen real não deve ser chamado")

    monkeypatch.setattr(tc.urllib.request, "urlopen", _boom)


class _RecordingTransport:
    """Transport falso: devolve uma resposta e registra as chamadas."""

    def __init__(self, response=b'{"ok": true, "result": []}'):
        self.response = response
        self.calls = []

    def __call__(self, url, body_bytes, timeout):
        self.calls.append({"url": url, "body": body_bytes, "timeout": timeout})
        return self.response


def _client(transport=None, token=TOKEN):
    return TelegramClient(token, transport=transport)


# --------------------------------------------------------------------------- #
# redact
# --------------------------------------------------------------------------- #
class TestRedact:
    def test_replaces_all_occurrences(self):
        assert redact("a TOKEN b TOKEN", "TOKEN") == "a *** b ***"

    def test_no_token_returns_text(self):
        assert redact("sem segredo", "") == "sem segredo"

    def test_non_string_is_stringified(self):
        assert redact(123, "x") == "123"


# --------------------------------------------------------------------------- #
# URL + payload + timeout no transport falso
# --------------------------------------------------------------------------- #
class TestTransportWiring:
    def test_get_updates_url_and_payload(self):
        tr = _RecordingTransport(b'{"ok": true, "result": []}')
        _client(tr).get_updates(offset=100, timeout=30)
        call = tr.calls[0]
        assert call["url"].endswith("/bot%s/getUpdates" % TOKEN)
        assert TOKEN in call["url"]
        body = json.loads(call["body"].decode("utf-8"))
        assert body == {"timeout": 30, "allowed_updates": ["message"], "offset": 100}

    def test_get_updates_timeout_is_long_poll_plus_ten(self):
        tr = _RecordingTransport(b'{"ok": true, "result": []}')
        _client(tr).get_updates(timeout=30)
        assert tr.calls[0]["timeout"] == 40

    def test_get_updates_without_offset_omits_offset(self):
        tr = _RecordingTransport(b'{"ok": true, "result": []}')
        _client(tr).get_updates()
        body = json.loads(tr.calls[0]["body"].decode("utf-8"))
        assert "offset" not in body

    def test_send_message_url_payload_and_no_parse_mode(self):
        tr = _RecordingTransport(b'{"ok": true, "result": {}}')
        _client(tr).send_message(chat_id=555, text="oi")
        call = tr.calls[0]
        assert call["url"].endswith("/bot%s/sendMessage" % TOKEN)
        body = json.loads(call["body"].decode("utf-8"))
        assert body == {"chat_id": 555, "text": "oi"}
        assert "parse_mode" not in body  # D6: texto puro

    def test_send_message_uses_send_timeout(self):
        tr = _RecordingTransport(b'{"ok": true, "result": {}}')
        _client(tr).send_message(chat_id=1, text="x")
        assert tr.calls[0]["timeout"] == 30

    def test_send_chat_action_payload(self):
        tr = _RecordingTransport(b'{"ok": true, "result": true}')
        _client(tr).send_chat_action(chat_id=9, action="typing")
        assert tr.calls[0]["url"].endswith("/bot%s/sendChatAction" % TOKEN)
        body = json.loads(tr.calls[0]["body"].decode("utf-8"))
        assert body == {"chat_id": 9, "action": "typing"}


# --------------------------------------------------------------------------- #
# UTF-8 (acentos e emoji) no corpo
# --------------------------------------------------------------------------- #
class TestUtf8Body:
    def test_accents_and_emoji_are_utf8_not_escaped(self):
        tr = _RecordingTransport(b'{"ok": true, "result": {}}')
        _client(tr).send_message(chat_id=1, text="Olá, café ☕🚀!")
        raw = tr.calls[0]["body"]
        # ensure_ascii=False -> bytes UTF-8 reais, não \uXXXX.
        assert "café".encode("utf-8") in raw
        assert "🚀".encode("utf-8") in raw
        assert b"\\u" not in raw
        body = json.loads(raw.decode("utf-8"))
        assert body["text"] == "Olá, café ☕🚀!"


# --------------------------------------------------------------------------- #
# Erros: ok:false, HTTPError, URLError -> TelegramAPIError redigido
# --------------------------------------------------------------------------- #
class TestErrorHandling:
    def test_ok_false_raises_with_error_code_and_retry_after(self):
        payload = {
            "ok": False,
            "error_code": 429,
            "description": "Too Many Requests: retry later",
            "parameters": {"retry_after": 7},
        }
        tr = _RecordingTransport(json.dumps(payload).encode("utf-8"))
        with pytest.raises(TelegramAPIError) as exc:
            _client(tr).get_updates()
        assert exc.value.error_code == 429
        assert exc.value.retry_after == 7

    def test_ok_false_without_parameters_has_no_retry_after(self):
        payload = {"ok": False, "error_code": 400, "description": "Bad Request"}
        tr = _RecordingTransport(json.dumps(payload).encode("utf-8"))
        with pytest.raises(TelegramAPIError) as exc:
            _client(tr).send_message(1, "x")
        assert exc.value.error_code == 400
        assert exc.value.retry_after is None

    def test_http_error_is_redacted_and_carries_code(self):
        body = json.dumps(
            {
                "ok": False,
                "error_code": 403,
                "description": "Forbidden: token %s leaked" % TOKEN,
            }
        ).encode("utf-8")

        def _raise_http(url, body_bytes, timeout):
            raise urllib.error.HTTPError(url, 403, "Forbidden", {}, io.BytesIO(body))

        with pytest.raises(TelegramAPIError) as exc:
            _client(_raise_http).get_updates()
        assert exc.value.error_code == 403
        assert TOKEN not in str(exc.value)
        assert "***" in str(exc.value)

    def test_http_error_non_json_body_still_raises(self):
        def _raise_http(url, body_bytes, timeout):
            raise urllib.error.HTTPError(
                url, 500, "Server Error", {}, io.BytesIO(b"<html>oops</html>")
            )

        with pytest.raises(TelegramAPIError) as exc:
            _client(_raise_http).get_updates()
        assert exc.value.error_code == 500

    def test_url_error_is_redacted(self):
        def _raise_url(url, body_bytes, timeout):
            raise urllib.error.URLError("connection refused for %s" % TOKEN)

        with pytest.raises(TelegramAPIError) as exc:
            _client(_raise_url).get_updates()
        assert TOKEN not in str(exc.value)

    def test_invalid_json_response_raises(self):
        tr = _RecordingTransport(b"not json at all")
        with pytest.raises(TelegramAPIError):
            _client(tr).get_updates()

    def test_get_updates_returns_result_list(self):
        updates = [{"update_id": 1, "message": {}}]
        tr = _RecordingTransport(
            json.dumps({"ok": True, "result": updates}).encode("utf-8")
        )
        assert _client(tr).get_updates() == updates

    def test_get_updates_empty_result_returns_empty_list(self):
        tr = _RecordingTransport(b'{"ok": true, "result": []}')
        assert _client(tr).get_updates() == []


# --------------------------------------------------------------------------- #
# Sigilo do token: repr/str/log
# --------------------------------------------------------------------------- #
class TestTokenSecrecy:
    def test_repr_and_str_never_contain_token(self):
        client = _client()
        assert TOKEN not in repr(client)
        assert TOKEN not in str(client)
        assert "***" in repr(client)

    def test_token_never_in_logs_even_on_error(self, caplog):
        import logging

        payload = {"ok": False, "error_code": 429, "description": "boom %s" % TOKEN}
        tr = _RecordingTransport(json.dumps(payload).encode("utf-8"))
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(TelegramAPIError):
                _client(tr).get_updates()
        assert TOKEN not in caplog.text

    def test_token_never_in_logs_on_success(self, caplog):
        import logging

        tr = _RecordingTransport(b'{"ok": true, "result": []}')
        with caplog.at_level(logging.DEBUG):
            _client(tr).get_updates()
        assert TOKEN not in caplog.text

    def test_url_with_token_never_logged(self, caplog):
        import logging

        tr = _RecordingTransport(b'{"ok": true, "result": []}')
        with caplog.at_level(logging.DEBUG):
            _client(tr).get_updates()
        # A URL contém o token; garantir que ela (e o token) não vazaram.
        assert ("bot" + TOKEN) not in caplog.text



# --------------------------------------------------------------------------- #
# chunk_text
# --------------------------------------------------------------------------- #
class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("oi", 4000) == ["oi"]

    def test_empty_and_whitespace_return_empty_list(self):
        assert chunk_text("", 4000) == []
        assert chunk_text("   \n\t  ", 4000) == []
        assert chunk_text(None, 4000) == []

    def test_exact_limit_single_chunk(self):
        text = "a" * 4000
        assert chunk_text(text, 4000) == [text]

    def test_no_chunk_exceeds_limit(self):
        text = ("palavra " * 2000).strip()  # ~14000 chars
        chunks = chunk_text(text, 4000)
        assert all(len(c) <= 4000 for c in chunks)
        assert len(chunks) >= 2

    def test_no_empty_chunks(self):
        text = (" palavra " * 3000)
        chunks = chunk_text(text, 4000)
        assert all(c for c in chunks)

    def test_long_without_spaces_is_hard_cut(self):
        text = "x" * 10000
        chunks = chunk_text(text, 4000)
        assert all(len(c) <= 4000 for c in chunks)
        assert all(c for c in chunks)
        assert "".join(chunks) == text

    def test_prefers_paragraph_break(self):
        para = "A" * 3990 + "\n\n" + "B" * 5000
        chunks = chunk_text(para, 4000)
        # Quebra no "\n\n" (depois do separador): chunk0 contém o parágrafo inteiro.
        assert chunks[0].endswith("\n\n")
        assert all(len(c) <= 4000 for c in chunks)
        # Lossless: nada é perdido.
        assert "".join(chunks) == para

    def test_prefers_line_break_over_space(self):
        text = ("linha com espacos\n" * 500)
        chunks = chunk_text(text, 200)
        assert all(len(c) <= 200 for c in chunks)
        assert chunks[0].endswith("\n")

    def test_reassembly_is_lossless(self):
        text = " ".join(["token%d" % i for i in range(3000)])
        chunks = chunk_text(text, 500)
        assert all(len(c) <= 500 for c in chunks)
        # Lossless: concatenar os blocos devolve exatamente o texto original.
        assert "".join(chunks) == text

    def test_invalid_limit_raises(self):
        with pytest.raises(ValueError):
            chunk_text("abc", 0)



# --------------------------------------------------------------------------- #
# extract_messages (D7)
# --------------------------------------------------------------------------- #
def _private_msg(update_id, sender, chat, text):
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": sender},
            "chat": {"id": chat, "type": "private"},
            "text": text,
        },
    }


class TestExtractMessages:
    def test_empty_list(self):
        assert extract_messages([]) == ([], None)

    def test_single_private_message(self):
        ups = [_private_msg(10, sender=111, chat=111, text="oi")]
        msgs, offset = extract_messages(ups)
        assert offset == 11
        assert msgs == [IncomingMessage(sender_id=111, chat_id=111, text="oi")]

    def test_group_is_ignored(self):
        ups = [
            {
                "update_id": 1,
                "message": {
                    "from": {"id": 111},
                    "chat": {"id": -999, "type": "group"},
                    "text": "oi grupo",
                },
            }
        ]
        msgs, offset = extract_messages(ups)
        assert msgs == []
        assert offset == 2  # offset avança mesmo ignorando

    def test_channel_is_ignored(self):
        ups = [
            {
                "update_id": 5,
                "message": {
                    "from": {"id": 1},
                    "chat": {"id": -100, "type": "channel"},
                    "text": "post",
                },
            }
        ]
        msgs, offset = extract_messages(ups)
        assert msgs == []
        assert offset == 6

    def test_message_without_text_is_ignored(self):
        ups = [
            {
                "update_id": 3,
                "message": {
                    "from": {"id": 111},
                    "chat": {"id": 111, "type": "private"},
                },
            }
        ]
        msgs, offset = extract_messages(ups)
        assert msgs == []
        assert offset == 4

    def test_edited_message_is_ignored(self):
        ups = [
            {
                "update_id": 7,
                "edited_message": {
                    "from": {"id": 111},
                    "chat": {"id": 111, "type": "private"},
                    "text": "editado",
                },
            }
        ]
        msgs, offset = extract_messages(ups)
        assert msgs == []
        assert offset == 8

    def test_message_without_from_is_ignored(self):
        ups = [
            {
                "update_id": 9,
                "message": {
                    "chat": {"id": 111, "type": "private"},
                    "text": "sem remetente",
                },
            }
        ]
        msgs, offset = extract_messages(ups)
        assert msgs == []
        assert offset == 10

    def test_offset_is_max_update_id_plus_one(self):
        ups = [
            _private_msg(100, 1, 1, "a"),
            _private_msg(50, 2, 2, "b"),
            _private_msg(300, 3, 3, "c"),
        ]
        msgs, offset = extract_messages(ups)
        assert offset == 301
        assert len(msgs) == 3

    def test_offset_advances_over_ignored_updates(self):
        ups = [
            _private_msg(10, 1, 1, "ok"),
            {
                "update_id": 20,
                "message": {
                    "from": {"id": 2},
                    "chat": {"id": 2, "type": "group"},
                    "text": "grupo",
                },
            },
        ]
        msgs, offset = extract_messages(ups)
        assert len(msgs) == 1
        assert offset == 21

    def test_mixed_bag(self):
        ups = [
            _private_msg(1, 111, 111, "oi"),
            {
                "update_id": 2,
                "message": {
                    "from": {"id": 222},
                    "chat": {"id": 222, "type": "supergroup"},
                    "text": "grupo",
                },
            },
            _private_msg(3, 333, 333, "tchau"),
        ]
        msgs, offset = extract_messages(ups)
        assert [m.text for m in msgs] == ["oi", "tchau"]
        assert offset == 4

