"""Testes do cliente Telegram e das funções puras do canal.

SEM rede real: uma fixture autouse faz urllib.request.urlopen levantar erro se
alguém tentar chamá-lo de verdade. Todo o tráfego é injetado via um
``transport`` falso.
"""

from __future__ import annotations

import http.client
import io
import json
import os
import tempfile
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

TOKEN = "123456:ABCDEFghij-KLMNOP_qrstuvw01"  # >=20 chars após ":" (válido p/ regex)


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
# Transporte padrão respeita Content-Type recebido
# --------------------------------------------------------------------------- #
class TestDefaultTransportContentType:
    def test_default_transport_uses_given_content_type(self, monkeypatch):
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"ok": true}'

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr(
            tc.urllib.request, "urlopen", fake_urlopen
        )
        client = _client()
        body = b"--boundary123\r\nexample\r\n"
        content_type = "multipart/form-data; boundary=boundary123"
        result = client._default_transport(
            "https://example.invalid/test",
            body,
            7,
            content_type=content_type,
        )
        request = captured["request"]
        assert request.get_method() == "POST"
        assert request.get_header("Content-type") == content_type
        assert request.data == body
        assert captured["timeout"] == 7
        assert result == b'{"ok": true}'


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
# sendVoice (mensagens de voz)
# --------------------------------------------------------------------------- #
class TestSendVoice:
    def _make_ogg_file(self):
        fd, path = tempfile.mkstemp(suffix=".ogg")
        with os.fdopen(fd, "wb") as f:
            f.write(b"fake-ogg-opus-content")
        return path

    def test_send_voice_url_is_sendvoice(self):
        tr = _RecordingTransport(b'{"ok": true, "result": {}}')
        path = self._make_ogg_file()
        try:
            _client(tr).send_voice(chat_id=555, voice_path=path)
        finally:
            os.unlink(path)
        call = tr.calls[0]
        assert call["url"].endswith("/bot%s/sendVoice" % TOKEN)
        assert TOKEN in call["url"]

    def test_send_voice_multipart_contains_chat_and_voice(self):
        tr = _RecordingTransport(b'{"ok": true, "result": {}}')
        path = self._make_ogg_file()
        try:
            _client(tr).send_voice(chat_id=777, voice_path=path)
        finally:
            os.unlink(path)
        call = tr.calls[0]
        body = call["body"].decode("utf-8", errors="replace")
        boundary = None
        for line in body.split("\r\n"):
            if line.startswith("--"):
                boundary = line[2:]
                break
        assert boundary is not None
        assert boundary in body
        assert 'name="chat_id"' in body
        assert "777" in body
        assert 'name="voice"; filename="voice.ogg"' in body
        assert "Content-Type: audio/ogg" in body
        assert b"fake-ogg-opus-content" in call["body"]

    def test_send_voice_boundary_matches_content_type(self, monkeypatch):
        client = _client()
        captured = {}

        def fake_call_raw(method, body_bytes, content_type, timeout=None):
            captured["method"] = method
            captured["body"] = body_bytes
            captured["content_type"] = content_type
            captured["timeout"] = timeout
            return {}

        monkeypatch.setattr(client, "_call_raw", fake_call_raw)
        path = self._make_ogg_file()
        try:
            client.send_voice(chat_id=777, voice_path=path)
        finally:
            os.unlink(path)
        content_type = captured["content_type"]
        assert captured["method"] == "sendVoice"
        assert content_type.startswith("multipart/form-data; boundary=")
        boundary = content_type.split("boundary=", 1)[1].encode("ascii")
        body = captured["body"]
        assert body.startswith(b"--" + boundary + b"\r\n")
        assert b"\r\n--" + boundary + b"--\r\n" in body

    def test_send_voice_uses_send_timeout(self):
        tr = _RecordingTransport(b'{"ok": true, "result": {}}')
        path = self._make_ogg_file()
        try:
            _client(tr).send_voice(chat_id=1, voice_path=path)
        finally:
            os.unlink(path)
        assert tr.calls[0]["timeout"] == 30

    def test_send_voice_transport_error_raises_telegramapierror(self):
        class _FailingTransport:
            def __call__(self, url, body_bytes, timeout):
                raise OSError("rede indisponivel")

        path = self._make_ogg_file()
        try:
            with pytest.raises(TelegramAPIError):
                _client(_FailingTransport()).send_voice(chat_id=1, voice_path=path)
        finally:
            os.unlink(path)


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


# --------------------------------------------------------------------------- #
# Validação do token no construtor (T3b 2a)
# --------------------------------------------------------------------------- #
class TestTokenValidation:
    def test_token_with_newline_and_trailing_spaces_is_accepted_after_strip(self):
        raw = "  \n123456:ABCDEFghij-KLMNOP_qrstuvw01\n  "
        client = TelegramClient(raw)
        assert TOKEN not in repr(client)
        # Não levanta; token interno é o valor com strip.
        assert client._token == TOKEN

    def test_malformed_token_raises_valueerror(self):
        with pytest.raises(ValueError):
            TelegramClient("nao-e-token")

    def test_token_with_inner_newline_rejected_by_fullmatch(self):
        # fullmatch ancora a string inteira: newline interno (após o strip das
        # bordas) não é tolerado, ao contrário de um ^...$ com \d.
        with pytest.raises(ValueError):
            TelegramClient("123456:ABCDEFghij-KLMNOP_qr\nstuvw01")

    def test_malformed_token_error_message_is_generic_without_token(self):
        bad = "123:short-and-wrong-format-value"
        with pytest.raises(ValueError) as exc:
            TelegramClient(bad)
        assert bad not in str(exc.value)
        assert str(exc.value) == "token do Telegram em formato invalido"

    def test_non_string_token_raises_valueerror(self):
        with pytest.raises(ValueError):
            TelegramClient(123456)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            TelegramClient(None)  # type: ignore[arg-type]

    def test_token_too_short_secret_rejected(self):
        # < 20 chars depois dos dois-pontos
        with pytest.raises(ValueError):
            TelegramClient("123456:short")

    def test_error_message_never_echoes_token_value(self):
        secretish = "999999:" + "X" * 30 + "unique-marker-xyz"
        # secretish é válido; para torná-lo inválido sem expor, corrompemos o fim
        invalid = secretish[:-1] + "!"  # '!' não permitido
        with pytest.raises(ValueError) as exc:
            TelegramClient(invalid)
        assert invalid not in str(exc.value)
        assert "unique-marker-xyz" not in str(exc.value)

    def test_token_valido_bot_id_minimo_5_digitos(self):
        # Limite inferior: bot_id com exatamente 5 dígitos é aceito.
        client = TelegramClient("12345:" + "A" * 20)
        assert client._token == "12345:" + "A" * 20

    def test_token_valido_segredo_minimo_20_chars(self):
        # Limite inferior: segredo com exatamente 20 caracteres é aceito.
        client = TelegramClient("123456:" + "B" * 20)
        assert client._token == "123456:" + "B" * 20

    def test_token_valido_com_underline_e_hifen(self):
        # O segredo pode conter underline e hífen além de alfanuméricos.
        client = TelegramClient("123456:" + "A_b-C_d-E_f-G_h-I_j-K_l-M_n")
        assert client._token == "123456:" + "A_b-C_d-E_f-G_h-I_j-K_l-M_n"

    def test_token_invalido_bot_id_4_digitos_rejeitado(self):
        # Menos de 5 dígitos no bot_id é rejeitado pelo regex.
        with pytest.raises(ValueError):
            TelegramClient("1234:" + "A" * 20)

    def test_token_invalido_segredo_19_chars_rejeitado(self):
        # Menos de 20 caracteres no segredo é rejeitado pelo regex.
        with pytest.raises(ValueError):
            TelegramClient("123456:" + "A" * 19)

    def test_token_invalido_bot_id_nao_numerico_rejeitado(self):
        # bot_id com letras é rejeitado (só [0-9] permitido).
        with pytest.raises(ValueError):
            TelegramClient("abcde:" + "A" * 20)

    def test_token_invalido_caractere_ponto_no_segredo_rejeitado(self):
        # '.' não é permitido no segredo (só [A-Za-z0-9_-]).
        with pytest.raises(ValueError):
            TelegramClient("123456:" + "A" * 19 + ".")

    def test_token_invalido_espaco_no_meio_rejeitado(self):
        # Espaço no meio do token não é tolerado (fullmatch ancorra tudo).
        with pytest.raises(ValueError):
            TelegramClient("123456:ABCDEF ghij-KLMNOP_qrst")

    def test_token_com_varios_whitespace_bordas_aceito_apos_strip(self):
        # strip() remove \r, \n, \r\n, tab e espaços nas bordas.
        for raw in ["\r\n123456:ABCDEFghij-KLMNOP_qrstuvw01\r\n",
                    "\t123456:ABCDEFghij-KLMNOP_qrstuvw01\t",
                    "  \r\n\t 123456:ABCDEFghij-KLMNOP_qrstuvw01  \t\n"]:
            client = TelegramClient(raw)
            assert client._token == "123456:ABCDEFghij-KLMNOP_qrstuvw01"


# --------------------------------------------------------------------------- #
# Erros de transporte de baixo nível -> TelegramAPIError redigido (T3b 2b)
# --------------------------------------------------------------------------- #
class TestTransportLowLevelErrors:
    def _raiser(self, exc):
        def _transport(url, body_bytes, timeout):
            raise exc

        return _transport

    def test_invalid_url_with_token_is_redacted(self):
        # http.client.InvalidURL (HTTPException) cuja msg inclui a URL/token.
        exc = http.client.InvalidURL("bad url for %s" % TOKEN)
        with pytest.raises(TelegramAPIError) as raised:
            _client(self._raiser(exc)).get_updates()
        assert TOKEN not in str(raised.value)
        assert "InvalidURL" in str(raised.value)

    def test_timeout_error_becomes_api_error(self):
        with pytest.raises(TelegramAPIError) as raised:
            _client(self._raiser(TimeoutError("timed out %s" % TOKEN))).get_updates()
        assert TOKEN not in str(raised.value)
        assert "TimeoutError" in str(raised.value)

    def test_connection_reset_becomes_api_error(self):
        with pytest.raises(TelegramAPIError) as raised:
            _client(
                self._raiser(ConnectionResetError("reset %s" % TOKEN))
            ).send_message(1, "x")
        assert TOKEN not in str(raised.value)
        assert "ConnectionResetError" in str(raised.value)

    def test_incomplete_read_becomes_api_error(self):
        with pytest.raises(TelegramAPIError) as raised:
            _client(self._raiser(http.client.IncompleteRead(b""))).get_updates()
        assert "IncompleteRead" in str(raised.value)

    def test_low_level_errors_do_not_leak_token_in_logs(self, caplog):
        import logging

        exc = http.client.InvalidURL("url %s invalida" % TOKEN)
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(TelegramAPIError):
                _client(self._raiser(exc)).get_updates()
        assert TOKEN not in caplog.text
        assert TOKEN not in repr(_client())



# --------------------------------------------------------------------------- #
# chunk_text: blocos só-whitespace descartados (T3b 2d)
# --------------------------------------------------------------------------- #
class TestChunkTextWhitespaceOnly:
    def test_trailing_spaces_produce_no_whitespace_only_blocks(self):
        text = "a" + (" " * 9000)
        chunks = chunk_text(text, 4000)
        assert all(c.strip() for c in chunks)  # nenhum bloco vazio/só espaço
        assert all(len(c) <= 4000 for c in chunks)
        # O primeiro bloco começa com "a"; os de espaço puro foram descartados.
        assert chunks[0].startswith("a")

    def test_newline_runs_produce_no_whitespace_only_blocks(self):
        text = "a" + ("\n" * 5000) + "b"
        chunks = chunk_text(text, 4000)
        assert all(c.strip() for c in chunks)  # nenhum bloco só "\n"
        assert all(len(c) <= 4000 for c in chunks)
        # "a" e "b" (o conteúdo real) sobrevivem em algum bloco.
        joined = "".join(chunks)
        assert "a" in joined and "b" in joined

    def test_all_whitespace_input_returns_empty(self):
        assert chunk_text(" " * 10000, 4000) == []
        assert chunk_text("\n" * 10000, 4000) == []

    def test_short_text_still_single_chunk(self):
        assert chunk_text("a" + " " * 100, 4000) == ["a" + " " * 100]


