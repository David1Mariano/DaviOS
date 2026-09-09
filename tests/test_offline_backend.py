"""Testes do backend standalone (llama.cpp), downloader e offline-first.

Nenhum teste faz request real a Internet: usamos fakes/mocks.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from brain.llm_provider import LLMProvider, LLMRequest, LLMResponse
from brain.providers.local_llama_cpp_provider import LocalLlamaCppProvider
from utils.model_downloader import (
    DownloadLock,
    DownloadResult,
    ModelDownloader,
)
from utils.llama_binary_downloader import (
    EXPECTED_BINARIES,
    LlamaBinaryDownloader,
    LlamaBinaryInfo,
    get_llama_bin_dir,
)


class FakeStandaloneProvider(LLMProvider):
    """Fake do provider standalone: responde com texto fixo."""

    name = "fake_llama_cpp"

    def __init__(self, available=True, backend="cpu"):
        self._available = available
        self._backend = backend
        self.calls = 0

    def initialize(self) -> bool:
        return self._available

    def is_available(self) -> bool:
        return self._available

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text="Resposta offline do backend.",
            provider=self.name,
            model="Qwen3-4B-Q4_K_M",
            backend=f"llama_cpp:{self._backend}",
        )

    def unload(self) -> None:
        self._available = False


class TestLlamaCppStandaloneProvider:
    def test_name(self):
        assert LocalLlamaCppProvider.name == "local_llama_cpp"

    def test_backend_ausente_reports_false(self, tmp_path, monkeypatch):
        provider = LocalLlamaCppProvider()
        # Garante que nenhum servidor externo seja detectado
        monkeypatch.setattr(provider, "_check_existing_server", lambda: False)
        assert provider.initialize() is False
        assert provider.is_available() is False

    def test_modelo_ausente_false(self, tmp_path, monkeypatch):
        from config.davios_config import DaviosConfig
        from brain.model_manager import ModelSelection

        config = DaviosConfig.load()
        selection = ModelSelection()
        provider = LocalLlamaCppProvider(config=config, selection=selection)
        monkeypatch.setattr(provider, "_check_existing_server", lambda: False)
        assert provider.initialize() is False

    def test_fake_provider_returns_text(self):
        provider = FakeStandaloneProvider()
        assert provider.initialize() is True
        resp = provider.generate(LLMRequest(prompt="oi"))
        assert resp.text == "Resposta offline do backend."
        assert resp.backend == "llama_cpp:cpu"
        assert provider.calls == 1

    def test_fake_provider_unload(self):
        provider = FakeStandaloneProvider()
        provider.unload()
        assert provider.is_available() is False

    def test_health_check_shape(self):
        provider = LocalLlamaCppProvider()
        info = provider.health_check()
        assert "available" in info
        assert "backend" in info
        assert "server_port" in info
        assert "base_url" in info


class TestLlamaBinaryDownloader:
    def test_get_llama_bin_dir(self, tmp_path):
        d = get_llama_bin_dir(tmp_path)
        assert str(d).endswith("llama.cpp")
        assert (tmp_path / "bin" / "llama.cpp") == d

    def test_is_installed_false_when_missing(self, tmp_path):
        downloader = LlamaBinaryDownloader(bin_dir=tmp_path / "nope")
        assert downloader.is_installed() is False

    def test_is_installed_true_when_server_exists(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "llama-server.exe").write_bytes(b"fake")
        downloader = LlamaBinaryDownloader(bin_dir=bin_dir)
        assert downloader.is_installed() is True

    def test_scan_binaries_detects_executables(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "llama-server.exe").write_bytes(b"a")
        (bin_dir / "llama-cli.exe").write_bytes(b"b")
        downloader = LlamaBinaryDownloader(bin_dir=bin_dir)
        info = downloader._scan_binaries()
        assert info.server_exe is not None
        assert info.cli_exe is not None
        assert info.is_ready is True

    def test_expected_binaries_defined(self):
        assert "llama-server.exe" in EXPECTED_BINARIES
        assert "llama-cli.exe" in EXPECTED_BINARIES


class TestModelDownloaderOffline:
    """Testes sem Internet: usam monkeypatch para simular falha de rede."""

    @staticmethod
    def _make_file(path: Path, size: int) -> Path:
        path.write_bytes(b"\0" * size)
        return path

    @staticmethod
    def _patch_curl(monkeypatch):
        """Faz `subprocess.run` do downloader falhar como se curl nao conectasse."""

        def fake_run(cmd, *args, **kwargs):
            raise subprocess.TimeoutExpired(cmd, timeout=kwargs.get("timeout"))

        import utils.model_downloader as md

        monkeypatch.setattr(md.subprocess, "run", fake_run)

    def test_downloader_detects_existing_complete(self, tmp_path):
        dl = ModelDownloader()
        dest = tmp_path / "model.gguf"
        # Usa SHA256 para verificacao exata (independente de tolerancia)
        content = b"x" * 5000
        dest.write_bytes(content)
        sha = dl._compute_sha256(dest)
        result = dl.download(
            url="http://127.0.0.1:1/nonexistent.gguf",
            dest_path=dest,
            expected_size=5000,
            sha256=sha,
        )
        assert result.success is True
        assert result.attempts == 0  # reutilizou, nao baixou

    def test_downloader_detects_corrupted_size(self, tmp_path, monkeypatch):
        self._patch_curl(monkeypatch)
        dl = ModelDownloader(max_retries=1)
        dest = tmp_path / "model.gguf"
        # Arquivo bem menor que o esperado (diff > tolerancia de 1024 bytes)
        self._make_file(dest, 50_000)
        result = dl.download(
            url="http://127.0.0.1:1/nonexistent.gguf",
            dest_path=dest,
            expected_size=100_000,
        )
        # Tamanho errado -> tenta baixar -> falha (simulada)
        assert result.success is False
        assert result.error != ""

    def test_download_fails_offline_gracefully(self, tmp_path, monkeypatch):
        self._patch_curl(monkeypatch)
        dl = ModelDownloader(max_retries=1)
        dest = tmp_path / "model.gguf"
        result = dl.download(
            url="http://127.0.0.1:1/never.gguf",
            dest_path=dest,
        )
        assert result.success is False
        assert result.error != ""

    def test_lock_prevents_duplicate_download(self, tmp_path):
        lock_path = tmp_path / "model.gguf.lock"
        with DownloadLock(lock_path):
            with pytest.raises(FileExistsError):
                with DownloadLock(lock_path):
                    pass  # nao deveria adquirir

    def test_lock_stale_is_reclaimed(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "model.gguf.lock"
        lock_path.write_text("99999999")  # PID inexistente = stale

        # Mock os.kill para simular processo morto (Windows pode ter comportamento diferente)
        import utils.model_downloader as md

        original_kill = md.os.kill

        def fake_kill(pid, sig):
            if pid == 99999999:
                raise ProcessLookupError("PID nao existe")
            return original_kill(pid, sig)

        monkeypatch.setattr(md.os, "kill", fake_kill)
        with DownloadLock(lock_path):
            assert lock_path.exists()

    def test_sha256_verification(self, tmp_path):
        dl = ModelDownloader()
        dest = tmp_path / "model.gguf"
        dest.write_bytes(b"dados")
        sha = dl._compute_sha256(dest)
        assert dl._verify_complete(dest, expected_size=5, sha256=sha) is True
        assert dl._verify_complete(dest, expected_size=5, sha256="0" * 64) is False

    def test_compute_sha256_stable(self, tmp_path):
        dl = ModelDownloader()
        dest = tmp_path / "model.gguf"
        dest.write_bytes(b"abc")
        assert len(dl._compute_sha256(dest)) == 64


class TestOfflineFirst:
    def test_fake_provider_works_without_network(self):
        provider = FakeStandaloneProvider()
        assert provider.is_available()
        resp = provider.generate(LLMRequest(prompt="qual o sentido da vida?"))
        assert resp.text != ""

    def test_config_offline_default(self):
        from config.davios_config import DaviosConfig

        config = DaviosConfig.load()
        assert config.offline_mode is True

    def test_backend_preference_standalone(self):
        from config.davios_config import DaviosConfig

        config = DaviosConfig.load()
        assert config.backend == "llama_cpp_standalone"