"""Testes para os modulos da Parte B - Parte 1: ActionManager + WebAccess."""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.action_manager import ActionManager, AllowedCommand
from brain.web_access import WebAccess


# ======================================================================
# ActionManager
# ======================================================================


class TestActionManagerDisabled:
    def test_disabled_rejects_all(self):
        mgr = ActionManager(enabled=False)
        result = mgr.execute("datetime")
        assert result.ok is False
        assert "deshabilitad" in result.message.lower()

    def test_disabled_lists_commands(self):
        mgr = ActionManager(enabled=False)
        cmds = mgr.list_commands()
        assert len(cmds) > 0


class TestActionManagerAllowlist:
    def test_unknown_command_rejected(self):
        mgr = ActionManager(enabled=True)
        result = mgr.execute("rm_rf_root")
        assert result.ok is False

    def test_is_allowed_knows_registered(self):
        mgr = ActionManager(enabled=True)
        assert mgr.is_allowed("datetime") is True
        assert mgr.is_allowed("nonexistent") is False


class TestActionManagerExecution:
    def test_echo_works(self):
        mgr = ActionManager(enabled=True)
        result = mgr.execute("echo", "oi mundo")
        assert result.ok is True
        assert "oi mundo" in result.stdout

    def test_datetime_works(self):
        mgr = ActionManager(enabled=True)
        result = mgr.execute("datetime")
        assert result.ok is True
        assert result.stdout != ""


class TestActionManagerDestructive:
    def test_destructive_without_confirm_rejects(self):
        cmd = AllowedCommand(
            name="del_file",
            argv=["cmd", "/c", "del", "{args}"],
            description="Deleta arquivo",
            destructive=True,
        )
        mgr = ActionManager(enabled=True, extra_commands=[cmd])
        result = mgr.execute("del_file", "teste.txt")
        assert result.ok is False
        assert result.requires_confirmation is True

    def test_destructive_with_confirm_yes_executes(self):
        cmd = AllowedCommand(
            name="del_file",
            argv=["cmd", "/c", "echo", "deletado:{args}"],
            description="Deleta arquivo (mock)",
            destructive=True,
        )
        mgr = ActionManager(enabled=True, extra_commands=[cmd], confirm=lambda _: True)
        result = mgr.execute("del_file", "teste.txt")
        assert result.ok is True
        assert "deletado:teste.txt" in result.stdout

    def test_destructive_with_confirm_no_cancels(self):
        cmd = AllowedCommand(
            name="del_file",
            argv=["cmd", "/c", "echo", "nunca"],
            description="Deleta arquivo (mock)",
            destructive=True,
        )
        mgr = ActionManager(enabled=True, extra_commands=[cmd], confirm=lambda _: False)
        result = mgr.execute("del_file", "teste.txt")
        assert result.ok is False
        assert "cancelada" in result.message.lower()


class TestActionManagerArgValidation:
    def test_args_too_long_rejected(self):
        cmd = AllowedCommand(
            name="print",
            argv=["cmd", "/c", "echo", "{args}"],
            description="Print",
            max_args_length=5,
        )
        mgr = ActionManager(enabled=True, extra_commands=[cmd])
        result = mgr.execute("print", "a" * 20)
        assert result.ok is False

    def test_empty_args_ok(self):
        mgr = ActionManager(enabled=True)
        result = mgr.execute("datetime", "")
        assert result.ok is True


# ======================================================================
# WebAccess
# ======================================================================


class TestWebAccessDisabled:
    def test_disabled_returns_error(self):
        web = WebAccess(enabled=False)
        result = web.fetch("https://example.com")
        assert result.ok is False
        assert "deshabilitado" in result.error.lower()

    def test_enabled_false_is_default(self):
        web = WebAccess()
        assert web.enabled is False


class TestWebAccessValidation:
    def test_non_http_scheme_rejected(self):
        web = WebAccess(enabled=True)
        result = web.fetch("ftp://example.com")
        assert result.ok is False

    def test_no_hostname_rejected(self):
        web = WebAccess(enabled=True)
        result = web.fetch("https://")
        assert result.ok is False


class TestWebAccessSanitization:
    def test_sanitize_removes_html(self):
        raw = "<html><body><p>Hello world</p><script>alert(1)</script></body></html>"
        result = WebAccess._sanitize_text(raw)
        assert "<" not in result
        assert "Hello world" in result
        assert "alert" not in result

    def test_sanitize_decodes_entities(self):
        raw = "<p>Ol&aacute; mundo &amp; tudo bem</p>"
        result = WebAccess._sanitize_text(raw)
        assert "Olá" in result

    def test_extract_title(self):
        raw = "<html><head><title>My Page Title</title></head><body>Content</body></html>"
        title = WebAccess._extract_title(raw)
        assert title == "My Page Title"

    def test_extract_title_missing(self):
        raw = "<html><body>No title here</body></html>"
        title = WebAccess._extract_title(raw)
        assert title == ""
