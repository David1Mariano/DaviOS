"""Downloader robusto e offline-resiliente para modelos GGUF do DaviOS.

Requisitos atendidos:
  - Download retomável (HTTP Range)
  - Arquivo temporário .part (nunca sobrescrever completo)
  - Verificar tamanho
  - Verificar SHA256 quando disponível
  - Retry automático com backoff progressivo
  - Detectar conexão interrompida e continuar do ponto onde parou
  - Evitar downloads duplicados (lock file de processo)
  - Detectar arquivo já completo
  - Verificar integridade antes de considerar concluído

O downloader usa curl.exe (Windows) com opções adequadas para resume e retry.
NÃO inicia múltiplos downloads simultâneos do mesmo arquivo.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = __import__("logging").getLogger("davios.downloader")

# Tempo máximo para uma única tentativa de curl
_DEFAULT_CONNECT_TIMEOUT = 30
_DEFAULT_READ_TIMEOUT = 120


@dataclass
class DownloadResult:
    """Resultado de um download."""

    success: bool
    file_path: str
    size_bytes: int = 0
    sha256: Optional[str] = None
    error: str = ""
    attempts: int = 0
    resumed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "error": self.error,
            "attempts": self.attempts,
            "resumed": self.resumed,
        }


class DownloadLock:
    """Lock file simples para impedir downloads duplicados de um mesmo arquivo.

    Usa um arquivo .lock com o PID do processo. Se o processo que criou o
    lock não existir mais, o lock é considerado óbvio (stale) e é reclamado.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._acquired = False

    def acquire(self) -> bool:
        """Tenta adquirir o lock. Retorna False se outro processo já está baixando."""
        if self.lock_path.exists():
            try:
                pid_str = self.lock_path.read_text().strip()
                pid = int(pid_str)
                try:
                    os.kill(pid, 0)  # signal 0 = check existence
                    logger.info("Download já em andamento (PID %d).", pid)
                    return False
                except (OSError, ProcessLookupError):
                    logger.info("Lock stale (PID %d morto). Reclamando.", pid)
            except (ValueError, IOError):
                pass  # lock file corrompido

        try:
            fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self._acquired = True
            return True
        except FileExistsError:
            return False

    def release(self) -> None:
        if self._acquired:
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
            self._acquired = False

    def __enter__(self):
        if not self.acquire():
            raise FileExistsError(
                f"Download já em andamento para {self.lock_path.stem}. "
                "Aguarde o processo anterior terminar."
            )
        return self

        def __exit__(self, *args):
        self.release()


class ModelDownloader:
    """Downloader resumable para arquivos GGUF.

    Uso:
        dl = ModelDownloader()
        result = dl.download(
            url="https://huggingface.co/.../model.gguf",
            dest_path=Path("models/balanced/model.gguf"),
            expected_size=2_600_000_000,
            sha256="abc123...",
        )
    """

    def __init__(
        self,
        max_retries: int = 10,
        base_backoff: float = 2.0,
        max_backoff: float = 30.0,
        connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT,
        read_timeout: int = _DEFAULT_READ_TIMEOUT,
    ):
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

    # ------------------------------------------------------------------

    def download(
        self,
        url: str,
        dest_path: Path,
        expected_size: Optional[int] = None,
        sha256: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> DownloadResult:
        """Baixa um arquivo com resume automatico.

        Se o arquivo destino ja existe e e completo (tamanho + sha256 OK),
        retorna sucesso imediatamente sem download.
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Checar se arquivo ja esta completo
        if dest_path.exists() and dest_path.is_file():
            if self._verify_complete(dest_path, expected_size, sha256):
                logger.info("Download ja concluido: %s", dest_path)
                return DownloadResult(
                    success=True,
                    file_path=str(dest_path),
                    size_bytes=dest_path.stat().st_size,
                    sha256=self._compute_sha256(dest_path) if sha256 else None,
                    attempts=0,
                    resumed=False,
                )

        # 2. Lock para evitar downloads duplicados
        lock_path = dest_path.with_suffix(dest_path.suffix + ".lock")
        with DownloadLock(lock_path) as lock:
            part_path = dest_path.with_suffix(dest_path.suffix + ".part")

            total_bytes = 0
            resumed = False
            attempt = 0

            for attempt in range(1, self.max_retries + 1):
                result = self._curl_download(
                    url=url,
                    part_path=part_path,
                    expected_size=expected_size,
                    sha256=sha256,
                    headers=headers,
                    resume_bytes=total_bytes,
                )

                if result.success:
                    self._finalize(part_path, dest_path)
                    if self._verify_complete(dest_path, expected_size, sha256):
                        final_size = dest_path.stat().st_size
                        final_sha = self._compute_sha256(dest_path) if sha256 else None
                        return DownloadResult(
                            success=True,
                            file_path=str(dest_path),
                            size_bytes=final_size,
                            sha256=final_sha,
                            attempts=attempt,
                            resumed=resumed,
                            error="",
                        )
                    else:
                        logger.warning("Arquivo corrompido. Reinicando download.")
                        try:
                            dest_path.unlink()
                        except FileNotFoundError:
                            pass
                        part_path.unlink(missing_ok=True)
                        total_bytes = 0
                        resumed = False
                        continue

                # Download falhou — verifica se pode retomar
                if part_path.exists():
                    total_bytes = part_path.stat().st_size
                    if total_bytes > 0:
                        resumed = True
                        logger.info(
                            "Download interrompido em %d bytes (tentativa %d/%d).",
                            total_bytes, attempt, self.max_retries,
                        )
                else:
                    total_bytes = 0

                if attempt < self.max_retries:
                    backoff = min(self.base_backoff * (2 ** (attempt - 1)), self.max_backoff)
                    sleep_time = backoff + 0.1 * backoff * 0.5
                    logger.info("Retry em %.1fs...", sleep_time)
                    time.sleep(sleep_time)

                        return DownloadResult(
                success=False,
                file_path=str(dest_path),
                size_bytes=total_bytes,
                error=f"Falha apos {attempt} tentativas.",
                attempts=attempt,
                resumed=resumed,
            )

    # ------------------------------------------------------------------

    def _curl_download(
        self,
        url: str,
        part_path: Path,
        expected_size: Optional[int],
        sha256: Optional[str],
        headers: Optional[dict],
        resume_bytes: int,
    ) -> DownloadResult:
        """Um unico attempt de download via curl.exe."""

        resume = resume_bytes > 0 and part_path.exists()

        cmd = [
            "curl.exe",
            "--silent",
            "--show-error",
            "--location",
            "--fail",
            "--connect-timeout", str(self.connect_timeout),
            "--max-time", str(self.read_timeout + self.connect_timeout),
            "--retry", "0",
        ]

        if resume:
            cmd.extend(["--continue-at", str(resume_bytes)])

        if headers:
            for key, val in headers.items():
                cmd.extend(["-H", f"{key}: {val}"])

        if resume:
            cmd.extend(["-H", f"Range: bytes={resume_bytes}-"])

        cmd.extend(["-o", str(part_path)])
        cmd.append(url)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=self.read_timeout + self.connect_timeout + 10,
            )
        except subprocess.TimeoutExpired:
            return DownloadResult(success=False, file_path=str(part_path), error="Timeout")
        except FileNotFoundError:
            logger.error("curl.exe nao encontrado. Usando fallback urllib.")
            return self._urllib_download(url, part_path, resume, resume_bytes)

        if result.returncode == 0:
            return DownloadResult(success=True, file_path=str(part_path))
        else:
            stderr = result.stderr.strip()
            logger.warning("curl falhou (exit %d): %s", result.returncode, stderr[:200])
            return DownloadResult(success=False, file_path=str(part_path), error=stderr)

    def _urllib_download(self, url, part_path, resume, resume_bytes):
        """Fallback usando urllib se curl nao estiver disponivel."""
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "DaviOS/1.0"})
        try:
            if resume:
                req.add_header("Range", f"bytes={resume_bytes}-")
                mode = "ab"
            else:
                mode = "wb"

            with urllib.request.urlopen(req, timeout=self.read_timeout) as resp:
                with open(part_path, mode) as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
            return DownloadResult(success=True, file_path=str(part_path))
        except Exception as e:
            return DownloadResult(success=False, file_path=str(part_path), error=str(e))

    # ------------------------------------------------------------------

    def _finalize(self, part_path: Path, dest_path: Path) -> None:
        """Move .part para o destino final."""
        if part_path.exists():
            part_path.replace(dest_path)

    def _verify_complete(
        self,
        path: Path,
        expected_size: Optional[int],
        sha256: Optional[str],
    ) -> bool:
        """Verifica tamanho e SHA256 do arquivo."""
        if not path.exists() or not path.is_file():
            return False

        actual_size = path.stat().st_size

        if expected_size is not None:
            diff = abs(actual_size - expected_size)
            if diff > max(1024, expected_size * 0.001):
                logger.warning(
                    "Tamanho incompativel: esperado %d, obtido %d (diff %d)",
                    expected_size, actual_size, diff,
                )
                return False

        if sha256:
            computed = self._compute_sha256(path)
            if computed != sha256:
                logger.warning("SHA256 incompativel.")
                return False

        return True

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()