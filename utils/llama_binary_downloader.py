@dataclass
class LlamaBinaryInfo:
    """Informacoes sobre o binario llama.cpp instalado."""

    bin_dir: Path
    server_exe: Optional[Path] = None
    cli_exe: Optional[Path] = None
    backend: str = "unknown"

    @property
    def is_ready(self) -> bool:
        return self.server_exe is not None and self.server_exe.exists()


def get_llama_bin_dir(project_root: Optional[Path] = None) -> Path:
    """Retorna o diretorio onde o llama.cpp deve ser instalado."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    return project_root / "bin" / "llama.cpp"


class LlamaBinaryDownloader:
    """Baixa e extrai o binario llama.cpp pre-compilado."""

    DEFAULT_VERSION = LLAMA_CPP_VERSION

    def __init__(self, bin_dir: Optional[Path] = None):
        self.bin_dir = bin_dir or get_llama_bin_dir()
        self.downloader = ModelDownloader(
            max_retries=5,
            base_backoff=2.0,
            max_backoff=30.0,
        )

    def is_installed(self) -> bool:
        """Verifica se os binarios do llama.cpp ja estao instalados."""
        server = self.bin_dir / "llama-server.exe"
        return server.exists()

    def install(self, prefer: str = "vulkan") -> LlamaBinaryInfo:
        """Baixa e instala os binarios do llama.cpp.

        Args:
            prefer: "vulkan" (prioriza Vulkan para AMD) ou "cpu" (prioriza CPU).
        """
        if self.is_installed():
            logger.info("llama.cpp ja instalado em %s", self.bin_dir)
            return self._scan_binaries()

        if prefer == "vulkan":
            urls = [VULKAN_DOWNLOAD_URL, CPU_DOWNLOAD_URL]
            backend = "vulkan"
        else:
            urls = [CPU_DOWNLOAD_URL, VULKAN_DOWNLOAD_URL]
            backend = "cpu"

        self.bin_dir.mkdir(parents=True, exist_ok=True)

        for url in urls:
            logger.info("Baixando llama.cpp de %s", url)
            fname = url.rsplit("/", 1)[-1]
            tmp_dir = Path(tempfile.mkdtemp(prefix="llama_dl_"))
            zip_path = tmp_dir / fname

            result = self.downloader.download(
                url=url,
                dest_path=zip_path,
                headers={"User-Agent": "DaviOS/1.0"},
            )

            if not result.success or not zip_path.exists():
                logger.warning("Download falhou: %s", result.error)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue

            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.bin_dir)
                logger.info("llama.cpp extraido em %s", self.bin_dir)
            except Exception as e:
                logger.error("Falha ao extrair: %s", e)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            info = self._scan_binaries()
            if info.is_ready:
                info.backend = backend
                return info

        raise RuntimeError(
            "Nao foi possivel baixar o llama.cpp. "
            "Baixe manualmente de: https://github.com/ggml-org/llama.cpp/releases "
            f"e extraia em: {self.bin_dir}"
        )

    def _scan_binaries(self) -> LlamaBinaryInfo:
        """Escaneia o diretorio bin por binarios conhecidos."""
        info = LlamaBinaryInfo(bin_dir=self.bin_dir)

        server = self.bin_dir / "llama-server.exe"
        if server.exists():
            info.server_exe = server
        else:
            info.server_exe = None

        cli = self.bin_dir / "llama-cli.exe"
        if cli.exists():
            info.cli_exe = cli
        else:
            info.cli_exe = None

        return info

    def get_info(self) -> LlamaBinaryInfo:
        """Retorna informacoes sobre o binario ja instalado (sem download)."""
        info = self._scan_binaries()
        if info.server_exe and info.server_exe.exists():
            try:
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                result = subprocess.run(
                    [str(info.server_exe), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=creationflags,
                )
                output = result.stdout + result.stderr
                if "vulkan" in output.lower():
                    info.backend = "vulkan"
                elif "cpu" in output.lower() or result.returncode == 0:
                    info.backend = "cpu"
                else:
                    info.backend = "unknown"
            except Exception:
                info.backend = "unknown"
        return info
"""Downloader e extrator do binario llama.cpp standalone (Windows x64).

Nao depende de llama-cpp-python. O binario pre-compilado e baixado da pagina
de releases do llama.cpp (ggml-org/llama.cpp) e extraido localmente.

Builds suportados:
  - Vulkan para AMD (RX 6600 etc.)
  - CPU-only (fallback universal)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.model_downloader import ModelDownloader

logger = logging.getLogger("davios.llama_downloader")

# Versao/build de referencia do llama.cpp
LLAMA_CPP_VERSION = "b10844"

# URLs de download (ordenadas por prioridade)
VULKAN_DOWNLOAD_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b10844/llama-b10844-bin-win-vulkan-x64.zip"
)
CPU_DOWNLOAD_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b10844/llama-b10844-bin-win-cpu-x64.zip"
)

# Arquivos binarios esperados dentro do zip
EXPECTED_BINARIES = ["llama-server.exe", "llama-cli.exe"]
