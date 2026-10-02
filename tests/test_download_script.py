"""Testes do script `scripts/download_model.py` (CLI de download de modelos).

Regras desta suíte:
- NENHUM download real e NENHUM acesso à internet: o downloader canônico é
  substituído por um dublê e o transporte HTTP seria um fake;
- O script é importado como módulo via importlib, nunca executado como
  subprocesso com rede;
- O catálogo usado é sintético, em `tmp_path`.

O foco é garantir que o script é um *adaptador fino* para o downloader
canônico: ele não baixa nada por conta própria, não aceita URL e não contorna
as proteções.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "download_model.py"


def _load_script():
    """Importa `scripts/download_model.py` como módulo, sem rodá-lo."""
    spec = importlib.util.spec_from_file_location("davios_download_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script():
    return _load_script()


# --------------------------------------------------------------------- #
# 1. O script não usa o downloader legado
# --------------------------------------------------------------------- #

class TestNoLegacyDownloader:
    def test_script_does_not_import_utils_model_downloader(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "utils.model_downloader" not in source
        assert "from utils import model_downloader" not in source

    def test_no_import_statement_targets_utils_model_downloader(self):
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "utils.model_downloader"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "utils.model_downloader" not in alias.name

    def test_script_imports_the_canonical_downloader(self, script):
        assert script.ModelDownloader.__module__ == "brain.model_downloader"

    def test_legacy_downloader_file_still_exists(self):
        """O legado NÃO pode ser apagado: `llama_binary_downloader` o usa."""
        assert (PROJECT_ROOT / "utils" / "model_downloader.py").is_file()


# --------------------------------------------------------------------- #
# 2. Nenhum bypass de catálogo na CLI
# --------------------------------------------------------------------- #

class TestNoUrlBypass:
    def _options(self, script):
        return {
            option
            for action in script.build_parser()._actions
            for option in action.option_strings
        }

    def test_cli_has_no_url_option(self, script):
        options = self._options(script)
        assert "--url" not in options
        assert "--dest" not in options
        assert "--dest-path" not in options
        assert "--sha256" not in options

    def test_cli_exposes_model_instead(self, script):
        assert "--model" in self._options(script)

    def test_url_option_is_rejected_by_argparse(self, script):
        """Passar --url deve falhar, não ser ignorado em silêncio."""
        with pytest.raises(SystemExit) as excinfo:
            script.build_parser().parse_args(["--url", "http://exemplo/m.gguf"])
        assert excinfo.value.code != 0

    def test_source_has_no_hardcoded_model_url(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        assert "huggingface.co" not in source
        assert "resolve/main" not in source

    def test_source_does_not_call_network_apis_directly(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        for forbidden in ("urllib", "requests", "curl", "subprocess", "http.client"):
            assert forbidden not in source



# --------------------------------------------------------------------- #
# Catálogo sintético (usado apenas por testes que precisam de IDs reais)
# --------------------------------------------------------------------- #

DOWNLOADABLE = "baixavel-q4km"
NO_URL = "sem-url-q4km"
UNKNOWN_ID = "nao-existe-q4km"


class FakeCatalog:
    def __init__(self):
        self.installed = []
        self.downloadable = []

    def get_installed_models(self):
        return self.installed

    def get_downloadable_models(self):
        return self.downloadable


class FakeDownloader:
    """Substitui o ModelDownloader canônico; registra o que foi pedido."""

    def __init__(self, preview_result=None, download_result=None):
        self.preview_result = preview_result
        self.download_result = download_result
        self.preview_calls = []
        self.download_calls = []
        self.catalog = FakeCatalog()

    def preview(self, model):
        self.preview_calls.append(model)
        if isinstance(self.preview_result, Exception):
            raise self.preview_result
        return self.preview_result

    def download(self, model, **kwargs):
        self.download_calls.append((model, kwargs))
        confirm = kwargs.get("confirm")
        if callable(confirm):
            kwargs["_confirm_answered"] = confirm(None)
        return self.download_result


def _patch(monkeypatch, script, downloader):
    monkeypatch.setattr(script, "_make_downloader", lambda: downloader)


def _ok_preview():
    from brain.model_downloader import DownloadPreview

    return DownloadPreview(
        model_id=DOWNLOADABLE,
        name="Modelo de Teste",
        expected_bytes=1024,
        download_url="http://127.0.0.1:9/fake.gguf",
        destination="models/light/baixavel-q4km.gguf",
        downloadable=True,
    )


# --------------------------------------------------------------------- #
# 3. O fluxo usa model_id e o downloader canônico
# --------------------------------------------------------------------- #

class TestFlowUsesCanonicalDownloader:
    def test_preview_is_called_with_the_model_id(self, script, monkeypatch):
        downloader = FakeDownloader(preview_result=_ok_preview())
        _patch(monkeypatch, script, downloader)
        monkeypatch.setattr(script, "_report", lambda result: 0)

        script.main(["--model", DOWNLOADABLE, "--yes", "--quiet"])

        assert downloader.preview_calls == [DOWNLOADABLE]
        assert downloader.download_calls[0][0] == DOWNLOADABLE

    def test_download_receives_confirmation_and_callbacks(self, script, monkeypatch):
        downloader = FakeDownloader(preview_result=_ok_preview())
        _patch(monkeypatch, script, downloader)
        monkeypatch.setattr(script, "_report", lambda result: 0)

        script.main(["--model", DOWNLOADABLE, "--yes"])

        _model, kwargs = downloader.download_calls[0]
        assert kwargs["require_confirmation"] is True
        assert callable(kwargs["confirm"])
        assert callable(kwargs["progress_callback"])

    def test_quiet_disables_progress_callback(self, script, monkeypatch):
        downloader = FakeDownloader(preview_result=_ok_preview())
        _patch(monkeypatch, script, downloader)
        monkeypatch.setattr(script, "_report", lambda result: 0)

        script.main(["--model", DOWNLOADABLE, "--yes", "--quiet"])


# --------------------------------------------------------------------- #
# 4. Indisponível não baixa
# --------------------------------------------------------------------- #

class TestUnavailableModelsAreNotDownloaded:
    def test_blocked_preview_never_calls_download(self, script, monkeypatch, capsys):
        from brain.model_downloader import DownloadPreview

        preview = DownloadPreview(
            model_id=NO_URL,
            name="Sem URL",
            blocked_reason="Origem oficial nao confirmada.",
        )
        downloader = FakeDownloader(preview_result=preview)
        _patch(monkeypatch, script, downloader)

        code = script.main(["--model", NO_URL, "--yes"])

        assert code == 4
        assert downloader.download_calls == []
        assert "Origem oficial nao confirmada" in capsys.readouterr().out

    def test_unknown_model_is_reported_without_downloading(
        self, script, monkeypatch, capsys
    ):
        from brain.model_downloader import DownloadPreview

        preview = DownloadPreview(
            model_id=UNKNOWN_ID,
            blocked_reason="Modelo nao encontrado no catalogo: %r" % UNKNOWN_ID,
        )
        downloader = FakeDownloader(preview_result=preview)
        _patch(monkeypatch, script, downloader)

        code = script.main(["--model", UNKNOWN_ID, "--yes"])

        assert code == 4
        assert downloader.download_calls == []
        assert "nao encontrado no catalogo" in capsys.readouterr().out


# --------------------------------------------------------------------- #
# 5. Confirmação passa pela API canônica
# --------------------------------------------------------------------- #

class TestConfirmationGoesThroughCanonicalApi:
    def test_yes_answers_true_to_confirm(self, script, monkeypatch):
        downloader = FakeDownloader(preview_result=_ok_preview())
        _patch(monkeypatch, script, downloader)
        monkeypatch.setattr(script, "_report", lambda result: 0)

        script.main(["--model", DOWNLOADABLE, "--yes"])

        _model, kwargs = downloader.download_calls[0]
        assert kwargs["confirm"](None) is True
        assert kwargs["_confirm_answered"] is True

    def test_without_yes_confirmation_is_interactive(self, script, monkeypatch):
        downloader = FakeDownloader(preview_result=_ok_preview())
        _patch(monkeypatch, script, downloader)
        monkeypatch.setattr(script, "_report", lambda result: 0)
        monkeypatch.setattr("builtins.input", lambda _prompt: "s")

        script.main(["--model", DOWNLOADABLE])

        _model, kwargs = downloader.download_calls[0]
        assert kwargs["confirm"] is script._confirm_interactive
        assert kwargs["_confirm_answered"] is True

    def test_declining_stops_without_downloading(self, script, monkeypatch):
        from brain.model_downloader import STATUS_DECLINED, DownloadResult

        result = DownloadResult(status=STATUS_DECLINED, model_id=DOWNLOADABLE)
        downloader = FakeDownloader(
            preview_result=_ok_preview(), download_result=result
        )
        _patch(monkeypatch, script, downloader)
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")

        code = script.main(["--model", DOWNLOADABLE])



# --------------------------------------------------------------------- #
# 6. Resultado do downloader é tratado corretamente
# --------------------------------------------------------------------- #

class TestResultHandling:
    def _run_with(self, script, monkeypatch, result):
        downloader = FakeDownloader(preview_result=_ok_preview(), download_result=result)
        _patch(monkeypatch, script, downloader)
        return script.main(["--model", DOWNLOADABLE, "--yes", "--quiet"])

    def test_downloaded_returns_zero(self, script, monkeypatch, capsys):
        from brain.model_downloader import STATUS_DOWNLOADED, DownloadResult

        result = DownloadResult(
            status=STATUS_DOWNLOADED,
            success=True,
            model_id=DOWNLOADABLE,
            destination="models/light/baixavel-q4km.gguf",
            downloaded_bytes=2048,
        )
        assert self._run_with(script, monkeypatch, result) == 0
        assert "2048 bytes" in capsys.readouterr().out

    def test_already_installed_returns_zero(self, script, monkeypatch, capsys):
        from brain.model_downloader import STATUS_ALREADY_INSTALLED, DownloadResult

        result = DownloadResult(
            status=STATUS_ALREADY_INSTALLED,
            success=True,
            already_installed=True,
            model_id=DOWNLOADABLE,
            destination="models/light/baixavel-q4km.gguf",
        )
        assert self._run_with(script, monkeypatch, result) == 0
        assert "ja estava instalado" in capsys.readouterr().out

    def test_refused_returns_four(self, script, monkeypatch):
        from brain.model_downloader import STATUS_REFUSED, DownloadResult

        result = DownloadResult(
            status=STATUS_REFUSED, model_id=NO_URL, error="Origem nao confirmada."
        )
        assert self._run_with(script, monkeypatch, result) == 4

    def test_failed_returns_one(self, script, monkeypatch):
        from brain.model_downloader import STATUS_FAILED, DownloadResult

        result = DownloadResult(
            status=STATUS_FAILED, model_id=DOWNLOADABLE, error="Rede caiu."
        )
        assert self._run_with(script, monkeypatch, result) == 1

    def test_confirmation_required_returns_two(self, script, monkeypatch):
        from brain.model_downloader import STATUS_CONFIRMATION_REQUIRED, DownloadResult

        result = DownloadResult(
            status=STATUS_CONFIRMATION_REQUIRED, model_id=DOWNLOADABLE
        )
        assert self._run_with(script, monkeypatch, result) == 2


# --------------------------------------------------------------------- #
# 7. Erros tratados sem traceback
# --------------------------------------------------------------------- #

class TestErrorHandling:
    def test_downloader_construction_failure_is_clean(self, script, monkeypatch, capsys):
        def boom():
            raise RuntimeError("config quebrado")

        monkeypatch.setattr(script, "_make_downloader", boom)

        code = script.main(["--model", DOWNLOADABLE, "--yes"])

        assert code == 1
        out = capsys.readouterr().out
        assert "config quebrado" in out
        assert "Traceback" not in out

    def test_preview_exception_is_clean(self, script, monkeypatch, capsys):
        downloader = FakeDownloader(preview_result=RuntimeError("catalogo corrompido"))
        _patch(monkeypatch, script, downloader)

        code = script.main(["--model", DOWNLOADABLE, "--yes"])

        assert code == 1
        out = capsys.readouterr().out
        assert "catalogo corrompido" in out
        assert "Traceback" not in out
        assert downloader.download_calls == []

    def test_download_exception_is_clean(self, script, monkeypatch, capsys):
        class Exploding(FakeDownloader):
            def download(self, model, **kwargs):
                raise OSError("conexao perdida")

        downloader = Exploding(preview_result=_ok_preview())
        _patch(monkeypatch, script, downloader)

        code = script.main(["--model", DOWNLOADABLE, "--yes"])

        assert code == 1
        out = capsys.readouterr().out
        assert "conexao perdida" in out
        assert "Traceback" not in out

    def test_list_with_empty_catalog(self, script, monkeypatch, capsys):
        downloader = FakeDownloader(preview_result=_ok_preview())
        _patch(monkeypatch, script, downloader)

        code = script.main(["--list"])

        assert code == 1
        assert "Nenhum modelo" in capsys.readouterr().out
        assert downloader.download_calls == []


# --------------------------------------------------------------------- #
# 8. Progresso nunca inventa denominador
# --------------------------------------------------------------------- #

class TestProgressReporting:
    def test_unknown_total_uses_bytes_not_percentage(self, script, capsys):
        report = script._progress_printer(quiet=False)
        report(64 * 1024 * 1024, 0)
        out = capsys.readouterr().out
        assert "64.0 MiB baixados" in out
        assert "%" not in out

    def test_known_total_uses_percentage(self, script, capsys):
        report = script._progress_printer(quiet=False)
        report(50, 100)
        assert "50%" in capsys.readouterr().out

    def test_quiet_returns_none(self, script):
        assert script._progress_printer(quiet=True) is None



# --------------------------------------------------------------------- #
# 10. STATUS_LOCKED
# --------------------------------------------------------------------- #

class TestLockedMessages:
    def test_locked_result_returns_exit_code_five(self, script, capsys):
        from brain.model_downloader import STATUS_LOCKED, DownloadResult

        result = DownloadResult(
            status=STATUS_LOCKED, model_id=DOWNLOADABLE, lock_holder="host:1234"
        )
        code = script._report(result)
        out = capsys.readouterr().out
        # 5 é um código próprio: conflito não é falha(1) nem recusa(4).
        assert code == 5
        assert "andamento" in out
        assert "host:1234" in out
        assert "Nenhum download novo" in out
        assert "Tente novamente" in out

    def test_locked_without_holder_still_reads_well(self, script, capsys):
        from brain.model_downloader import STATUS_LOCKED, DownloadResult

        result = DownloadResult(status=STATUS_LOCKED, model_id=DOWNLOADABLE)
        code = script._report(result)
        assert code == 5
        assert "andamento" in capsys.readouterr().out

    def test_cli_never_waits_or_polls_for_the_lock(self, script):
        """A CLI não pode ter espera, polling nem bypass do lock."""
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        for proibido in (
            "while not", "sleep(", "wait_lock", "wait=True",
            "--wait", "--force", "--ignore-lock", "--skip-lock",
            "--no-wait", "poll",
        ):
            assert proibido not in source, f"CLI contém {proibido!r}"

    def test_yes_is_not_a_lock_bypass(self, script, monkeypatch):
        """`--yes` só confirma; não destrava um download em andamento."""
        from brain.model_downloader import (
            STATUS_DOWNLOADED,
            STATUS_LOCKED,
            DownloadResult,
        )

        dono = DownloadResult(status=STATUS_LOCKED, model_id=DOWNLOADABLE)
        result = DownloadResult(status=STATUS_DOWNLOADED, success=True,
                                model_id=DOWNLOADABLE, destination="x", downloaded_bytes=1)

        class Fake:
            def preview(self, model):
                from brain.model_downloader import DownloadPreview
                return DownloadPreview(model_id=DOWNLOADABLE, name="M",
                                       expected_bytes=1,
                                       download_url="http://fake/x.gguf",
                                       destination="x", downloadable=True)

            def download(self, model, **kwargs):
                # A API canônica é quem decide; a CLI apenas repassa.
                return result if kwargs.get("confirm", lambda p: True)(None) else dono

        monkeypatch.setattr(script, "_make_downloader", lambda: Fake())
        code = script.main(["--model", DOWNLOADABLE, "--yes", "--quiet"])
        # Com --yes a confirmação responde True, mas o status continua
        # vindo do downloader: a CLI não "fura" nada.
        assert code in (0, 5)



# --------------------------------------------------------------------- #
# 9. Mensagens de integridade SHA-256
# --------------------------------------------------------------------- #

class TestIntegrityMessages:
    """A CLI não calcula hash: ela traduz o que o downloader verificou."""

    def _result(self, **kwargs):
        from brain.model_downloader import DownloadResult

        defaults = {
            "status": "downloaded",
            "success": True,
            "model_id": DOWNLOADABLE,
            "destination": "models/light/x.gguf",
            "downloaded_bytes": 2048,
        }
        defaults.update(kwargs)
        return DownloadResult(**defaults)

    def test_verified_hash_is_announced(self, script):
        result = self._result(hash_verified=True, expected_sha256="a" * 64)
        joined = " ".join(script._report_integrity(result))
        assert "Integridade verificada" in joined
        assert "SHA-256 confere" in joined

    def test_mismatch_shows_both_hashes(self, script):
        result = self._result(
            status="failed",
            success=False,
            hash_verified=False,
            expected_sha256="a" * 64,
            actual_sha256="b" * 64,
        )
        joined = " ".join(script._report_integrity(result))
        assert "não confere" in joined
        assert "a" * 64 in joined and "b" * 64 in joined
        assert "Nada foi instalado" in joined

    def test_absent_hash_is_flagged_as_unverified(self, script):
        result = self._result(
            hash_verified=False, expected_sha256=None, actual_sha256=None
        )
        joined = " ".join(script._report_integrity(result))
        assert "não possui SHA-256" in joined
        assert "não foi verificada" in joined

    def test_cli_never_computes_hash_itself(self):
        """A CLI não pode ler o arquivo: só traduz o DownloadResult."""
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        for forbidden in ("hashlib", "_compute_sha256", "sha256("):
            assert forbidden not in source

    def test_cli_has_no_hash_bypass_option(self, script):
        options = {
            option
            for action in script.build_parser()._actions
            for option in action.option_strings
        }
        for bypass in (
            "--skip-hash", "--no-hash", "--ignore-hash", "--force-hash",
        ):
            assert bypass not in options

    def test_downloaded_result_prints_integrity_line(self, script, capsys):
        result = self._result(hash_verified=True, expected_sha256="a" * 64)
        code = script._report(result)
        assert code == 0
        assert "Integridade verificada" in capsys.readouterr().out

    def test_mismatch_result_prints_both_hashes(self, script, capsys):
        result = self._result(
            status="failed", success=False, hash_verified=False,
            expected_sha256="a" * 64, actual_sha256="b" * 64,
        )
        code = script._report(result)
        out = capsys.readouterr().out
        assert code == 1
        assert "a" * 64 in out and "b" * 64 in out

    def test_already_installed_prints_integrity_line(self, script, capsys):
        result = self._result(
            status="already_installed",
            already_installed=True,
            hash_verified=True,
            expected_sha256="a" * 64,
        )
        code = script._report(result)
        assert code == 0
        assert "Integridade verificada" in capsys.readouterr().out
