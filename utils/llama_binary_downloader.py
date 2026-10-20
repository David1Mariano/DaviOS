"""Downloader e extrator do binario llama.cpp standalone (Windows x64).

Nao depende de llama-cpp-python. O binario pre-compilado e baixado da pagina
de releases do llama.cpp (ggml-org/llama.cpp) e extraido localmente.

Builds suportados:
  - Vulkan para AMD (RX 6600 etc.)
  - CPU-only (fallback universal)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.model_downloader import ModelDownloader

logger = logging.getLogger("davios.llama_downloader")

# Versao/build de referencia do llama.cpp
LLAMA_CPP_VERSION = "b10853"

# URLs de download (ordenadas por prioridade)
VULKAN_DOWNLOAD_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b10853/llama-b10853-bin-win-vulkan-x64.zip"
)
CPU_DOWNLOAD_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b10853/llama-b10853-bin-win-cpu-x64.zip"
)

# Tamanhos esperados (bytes) - para verificacao
VULKAN_ZIP_SIZE = 35_710_570
CPU_ZIP_SIZE = 18_417_116

# Arquivos binarios esperados dentro do zip
EXPECTED_BINARIES = ["llama-server.exe", "llama-cli.exe"]


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


def get_llama_bin_candidates(project_root: Optional[Path] = None) -> list[Path]:
    """Lista ordenada de diretorios onde o llama.cpp pode JA estar instalado.

    Ordem de preferencia:

    1. ``<raiz do projeto>/bin/llama.cpp`` — a instalacao deste projeto;
    2. ``<raiz do projeto>/../DaviOS/bin/llama.cpp`` — a instalacao IRMA do
       DaviOS. O DaviOS e o DaviOS-telegram sao projetos separados que
       convivem no mesmo disco: o binario costuma ser baixado uma vez na
       instalacao principal e o projeto telegram nao o repete. Sem esta
        busca, o provider cairia em "llama-server.exe nao encontrado" mesmo
       com o binario presente no disco;
    3. ``<raiz do projeto>/../DaviOS-telegram/bin/llama.cpp`` e demais
       pastas irmas com prefixo ``DaviOS`` — mesma razao do item 2.

    A lista NAO cria diretorios: apenas aponta caminhos possiveis. A
    verificacao de existencia fica a cargo de quem chama.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    root = Path(project_root).resolve()
    candidates: list[Path] = [root / "bin" / "llama.cpp"]
    try:
        parent = root.parent
    except (OSError, IndexError):
        parent = None
    if parent and parent.is_dir():
        for sibling in sorted(parent.iterdir()):
            if not sibling.is_dir() or sibling == root:
                continue
            if sibling.name.lower().startswith("davios"):
                candidates.append(sibling / "bin" / "llama.cpp")
    return candidates


def find_llama_server_exe(project_root: Optional[Path] = None) -> Optional[Path]:
    """Localiza ``llama-server.exe`` no proprio projeto ou em installacoes irmas.

    Devolve None se em nenhum lugar houver o executavel. Nao consulta o PATH
    (isso e responsabilidade do provider, que tambem aceita binarios
    instalados pelo sistema).
    """
    for bin_dir in get_llama_bin_candidates(project_root):
        candidate = bin_dir / "llama-server.exe"
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


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
        return (self.bin_dir / "llama-server.exe").exists()

    def install(self, prefer: str = "vulkan") -> LlamaBinaryInfo:
        """Baixa e instala os binarios do llama.cpp.

        Suporta retomada: se existir arquivo .part, continua do ponto onde parou.

        Args:
            prefer: "vulkan" (prioriza Vulkan para AMD) ou "cpu" (prioriza CPU).
        """
        if self.is_installed():
            logger.info("llama.cpp ja instalado em %s", self.bin_dir)
            return self._scan_binaries()

        if prefer == "vulkan":
            urls = [VULKAN_DOWNLOAD_URL, CPU_DOWNLOAD_URL]
            backend = "vulkan"
            expected_size = VULKAN_ZIP_SIZE
        else:
            urls = [CPU_DOWNLOAD_URL, VULKAN_DOWNLOAD_URL]
            backend = "cpu"
            expected_size = CPU_ZIP_SIZE

        self.bin_dir.mkdir(parents=True, exist_ok=True)

        for url in urls:
            logger.info("Baixando llama.cpp de %s", url)
            fname = url.rsplit("/", 1)[-1]
            zip_path = self.bin_dir / fname

            # Verifica se ja existe arquivo completo (mesmo sem .part)
            if zip_path.exists():
                actual_size = zip_path.stat().st_size
                if actual_size == expected_size:
                    logger.info("Arquivo ja completo: %s", fname)
                    return self._extract_and_verify(zip_path, backend)
                else:
                    logger.info("Arquivo incompleto (%d/%d), retomando...", actual_size, expected_size)

            result = self.downloader.download(
                url=url,
                dest_path=zip_path,
                expected_size=expected_size,
                headers={"User-Agent": "DaviOS/1.0"},
            )

            if not result.success or not zip_path.exists():
                logger.warning("Download falhou: %s", result.error)
                # Mantem arquivo .part para retomada futura
                continue

            info = self._extract_and_verify(zip_path, backend)
            if info.is_ready:
                return info

        # Se chegou aqui, nao conseguiu baixar
        # Verifica se existe .part para retomada
        for url in urls:
            fname = url.rsplit("/", 1)[-1]
            part_path = self.bin_dir / (fname + ".part")
            if part_path.exists():
                size_mb = part_path.stat().st_size / 1024 / 1024
                logger.info("Arquivo parcial salvo: %s (%.1f MB) - execute novamente para retomar", fname, size_mb)

        raise RuntimeError(
            "Nao foi possivel baixar o llama.cpp. "
            "Baixe manualmente de: https://github.com/ggml-org/llama.cpp/releases "
            f"e extraia em: {self.bin_dir}"
        )

    def _extract_and_verify(self, zip_path: Path, backend: str) -> LlamaBinaryInfo:
        """Extrai o zip e verifica os binarios."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.bin_dir)
            logger.info("llama.cpp extraido em %s", self.bin_dir)
        except Exception as e:
            logger.error("Falha ao extrair: %s", e)
            return LlamaBinaryInfo(bin_dir=self.bin_dir)

        info = self._scan_binaries()
        if info.is_ready:
            info.backend = backend
            # Remove o zip apos extracao bem-sucedida
            try:
                zip_path.unlink()
            except OSError:
                pass
        return info

    def _scan_binaries(self) -> LlamaBinaryInfo:
        """Escaneia o diretorio bin por binarios conhecidos."""
        info = LlamaBinaryInfo(bin_dir=self.bin_dir)
        server = self.bin_dir / "llama-server.exe"
        info.server_exe = server if server.exists() else None
        cli = self.bin_dir / "llama-cli.exe"
        info.cli_exe = cli if cli.exists() else None
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