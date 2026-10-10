"""Testes do contrato de canais (brain/channels) e dos campos de config.

Sem rede e sem llama-server: apenas funções puras, dataclasses e carga de
config a partir de JSON em diretório temporário.
"""

from __future__ import annotations

import json

import pytest

from brain.channels import Channel, IncomingMessage, is_authorized
from brain.channels.base import Channel as BaseChannel
from config.davios_config import DaviosConfig


# --------------------------------------------------------------------------- #
# is_authorized (função pura de whitelist)
# --------------------------------------------------------------------------- #
class TestIsAuthorized:
    def test_authorized_when_id_in_list(self):
        assert is_authorized(123, [1, 123, 456]) is True

    def test_not_authorized_when_id_absent(self):
        assert is_authorized(999, [1, 123, 456]) is False

    def test_empty_list_authorizes_nobody(self):
        # REGRA: lista vazia NUNCA libera tudo por omissão.
        assert is_authorized(123, []) is False

    def test_none_list_authorizes_nobody(self):
        assert is_authorized(123, None) is False

    def test_single_authorized(self):
        assert is_authorized(42, [42]) is True

    def test_wrong_type_string_id_still_matches_int(self):
        # Tolerante a tipo: "123" (str) casa com 123 (int).
        assert is_authorized(123, ["123"]) is True
        assert is_authorized("123", [123]) is True

    def test_wrong_type_uncoercible_id_does_not_authorize(self):
        # id de tipo errado e inconversível -> não autoriza.
        assert is_authorized("abc", [123]) is False
        assert is_authorized(123, [None, "xyz"]) is False

    def test_invalid_entries_are_ignored_not_fatal(self):
        # Elemento inválido na whitelist é ignorado; id válido ainda casa.
        assert is_authorized(123, [None, "xyz", 123]) is True


# --------------------------------------------------------------------------- #
# IncomingMessage (dataclass)
# --------------------------------------------------------------------------- #
class TestIncomingMessage:
    def test_fields(self):
        msg = IncomingMessage(sender_id=10, chat_id=20, text="oi")
        assert msg.sender_id == 10
        assert msg.chat_id == 20
        assert msg.text == "oi"

    def test_is_frozen(self):
        msg = IncomingMessage(sender_id=1, chat_id=2, text="x")
        with pytest.raises(Exception):
            msg.text = "outro"  # type: ignore[misc]



# --------------------------------------------------------------------------- #
# Contrato de Channel (classe abstrata)
# --------------------------------------------------------------------------- #
class _MinimalChannel(BaseChannel):
    """Subclasse mínima apenas para exercitar o contrato."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class TestChannelContract:
    def test_cannot_instantiate_abstract_channel(self):
        with pytest.raises(TypeError):
            Channel()  # type: ignore[abstract]

    def test_default_name(self):
        assert BaseChannel.name == "channel"

    def test_subclass_can_override_name(self):
        class Named(BaseChannel):
            name = "telegram"

            def start(self) -> None:  # pragma: no cover - não chamado
                ...

            def stop(self) -> None:  # pragma: no cover - não chamado
                ...

        assert Named().name == "telegram"

    def test_minimal_subclass_start_stop(self):
        ch = _MinimalChannel()
        assert ch.started is False and ch.stopped is False
        ch.start()
        ch.stop()
        assert ch.started is True and ch.stopped is True

    def test_incomplete_subclass_cannot_instantiate(self):
        class MissingStop(BaseChannel):
            def start(self) -> None:  # pragma: no cover
                ...

        with pytest.raises(TypeError):
            MissingStop()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# Campos de config do Telegram (defaults + carga por JSON)
# --------------------------------------------------------------------------- #
class TestTelegramConfigDefaults:
    def test_defaults_are_off_and_empty(self):
        cfg = DaviosConfig()
        assert cfg.telegram_enabled is False
        assert cfg.telegram_allowed_user_ids == []
        assert cfg.telegram_tools_enabled is False

    def test_default_list_is_independent_per_instance(self):
        # field(default_factory=list): mutar uma instância não afeta outra.
        a = DaviosConfig()
        b = DaviosConfig()
        a.telegram_allowed_user_ids.append(1)
        assert b.telegram_allowed_user_ids == []

    def test_no_token_field_in_config(self):
        # D5: token NÃO pode existir como campo do DaviosConfig.
        cfg = DaviosConfig()
        assert not hasattr(cfg, "telegram_token")
        assert not hasattr(cfg, "telegram_bot_token")


class TestTelegramConfigLoadFromJson:
    def _write(self, tmp_path, data: dict):
        p = tmp_path / "davios.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        return p

    def test_load_reads_telegram_fields(self, tmp_path):
        path = self._write(
            tmp_path,
            {
                "telegram_enabled": True,
                "telegram_allowed_user_ids": [111, 222],
                "telegram_tools_enabled": True,
            },
        )
        cfg = DaviosConfig.load(str(path))
        assert cfg.telegram_enabled is True
        assert cfg.telegram_allowed_user_ids == [111, 222]
        assert cfg.telegram_tools_enabled is True

    def test_load_without_telegram_fields_keeps_defaults(self, tmp_path):
        # Regressão zero: JSON sem os novos campos continua com os defaults.
        path = self._write(tmp_path, {"debug": True})
        cfg = DaviosConfig.load(str(path))
        assert cfg.debug is True
        assert cfg.telegram_enabled is False
        assert cfg.telegram_allowed_user_ids == []
        assert cfg.telegram_tools_enabled is False

    def test_load_ignores_unknown_fields(self, tmp_path):
        # Campos desconhecidos continuam sendo filtrados (comportamento atual).
        path = self._write(tmp_path, {"campo_inexistente": 1})
        cfg = DaviosConfig.load(str(path))
        assert not hasattr(cfg, "campo_inexistente")
