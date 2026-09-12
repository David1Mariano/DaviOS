import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.action_manager import ActionManager, AllowedCommand, _read_file_handler
from brain.tool_adapters import register_default_tools
from brain.tool_registry import ToolRegistry, ToolRouter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestReadFileTool:
    def test_read_file_registered_in_registry(self):
        reg = ToolRegistry()
        registered = register_default_tools(reg)
        assert "read_file" in registered
        router = ToolRouter(registry=reg)
        result = router.route("read_file")
        assert result.found is True
        assert result.tool.name == "read_file"
        assert "le" in result.tool.description.lower() or "arquivo" in result.tool.description.lower()

    def test_read_file_not_found(self):
        am = ActionManager(enabled=True)
        result = am.execute("read_file", args="caminho/inexistente/xyz.txt")
        assert result.ok is False
        msg = result.message.lower()
        assert "nao encontrado" in msg or "invalido" in msg or "acesso negado" in msg

    def test_read_file_outside_allowed_directory(self):
        am = ActionManager(enabled=True)
        result = am.execute("read_file", args="../fora_do_projeto.txt")
        assert result.ok is False
        msg = result.message.lower()
        assert "acesso negado" in msg or "fora" in msg or "invalido" in msg

    def test_read_file_truncated_large_file(self):
        test_file = PROJECT_ROOT / "tmp_test_grande.txt"
        test_file.write_text("A" * 300, encoding="utf-8")
        try:
            result = _read_file_handler(str(test_file), max_bytes=100)
            assert "truncado" in result.lower()
            assert "300" in result
        finally:
            test_file.unlink(missing_ok=True)

    def test_read_file_non_utf8_encoding(self):
        test_file = PROJECT_ROOT / "tmp_test_latin1.txt"
        test_file.write_bytes("cafe resume".encode("latin-1"))
        try:
            result = _read_file_handler(str(test_file))
            assert len(result) > 0
        finally:
            test_file.unlink(missing_ok=True)

    def test_read_file_small_text_file(self):
        test_file = PROJECT_ROOT / "tmp_test_small.txt"
        test_file.write_text("Ola, mundo!\\nLinha 2.", encoding="utf-8")
        try:
            result = _read_file_handler(str(test_file))
            assert "Ola, mundo!" in result
            assert "Linha 2." in result
        finally:
            test_file.unlink(missing_ok=True)
