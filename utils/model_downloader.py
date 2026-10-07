"""Downloader robusto e offline-resiliente para modelos GGUF do DaviOS.

Requisitos atendidos:
  - Download retomavel (HTTP Range)
  - Arquivo temporario .part (nunca sobrescrever completo)
  - Verificar tamanho
  - Verificar SHA256 quando disponivel
  - Retry automatico com backoff progressivo
  - Detectar conexao interrompida e continuar do ponto onde parou
  - Evitar downloads duplicados (lock file de processo)
  - Verificar integridade antes de considerar concluido

Usa curl.exe (Windows) com opcoes de resume e retry.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("davios.downloader")

_DEFAULT_CONNECT_TIMEOUT = 30
_DEFAULT_READ_TIMEOUT = 120


_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

_win32_kernel = None


def _win32_api():
    """Retorna o kernel32 com protótipos Win32 e captura documentada de erro.

    `WinDLL(..., use_last_error=True)` habilita `ctypes.get_last_error()`,
    o mecanismo documentado para ler o último código de erro da chamada.
    `argtypes`/`restype` explícitos (HANDLE = ponteiro) evitam truncamento
    de handle em sistemas de 64 bits. Instanciado uma única vez (cache).
    """
    global _win32_kernel
    if _win32_kernel is None:
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        kernel.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        _win32_kernel = kernel
    return _win32_kernel


def _process_is_alive(pid: int) -> bool:
    """True se o processo com este PID existe AGORA neste host.

    No Windows, `os.kill(pid, 0)` NAO e seguro: signal 0 vale
    CTRL_C_EVENT (0) e a chamada gera um Ctrl+C no console
    compartilhado, interrompendo o processo chamador (e o shell
    hospedeiro). Usa-se `OpenProcess` via ctypes, que apenas consulta.
    No POSIX, `os.kill(pid, 0)` e o probe padrao (não envia sinal).

    No Windows, morte so e afirmada com `ERROR_INVALID_PARAMETER` (87),
    que indica que o PID não existe. `ERROR_ACCESS_DENIED` (5) prova que
    o processo EXISTE. Qualquer outro codigo de erro ou excecao segue a
    política conservadora de tratar como vivo, para não roubar o lock.
    """
    if pid <= 0:
        return False

    if os.name == "nt":
        try:
            import ctypes

            kernel = _win32_api()
            # PROCESS_QUERY_LIMITED_INFORMATION: o menos privilegiado
            # que ainda responde "existe?".
            handle = kernel.OpenProcess(
                _PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle:
                kernel.CloseHandle(handle)
                return True
            err = ctypes.get_last_error()
            if err == _ERROR_ACCESS_DENIED:
                return True  # existe, mas sem permissao para inspecionar
            if err == _ERROR_INVALID_PARAMETER:
                return False  # PID inexistente (codigo confiavel)
            # Codigo inesperado: nao prova morte; na duvida, vivo.
            logger.debug(
                "OpenProcess: erro %d inesperado; PID %d tratado como vivo.",
                err,
                pid,
            )
            return True
        except Exception:  # noqa: BLE001 - nunca derruba o download por isso
            return True  # na duvida, vivo (nao rouba o lock)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # existe, mas e de outro usuario
    except OSError:
        return False
    return True


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
    """Lock file para impedir downloads duplicados de um mesmo arquivo.

    Usa .lock com o PID do processo. Se o processo que criou o lock
    nao existir mais, o lock e considerado obsoleto (stale) e e reclaim.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._acquired = False

    def acquire(self) -> bool:
        """Tenta adquirir o lock. False se outro processo ja esta baixando."""
        if self.lock_path.exists():
            try:
                pid_str = self.lock_path.read_text().strip()
                pid = int(pid_str)
                if _process_is_alive(pid):
                    logger.info("Download ja em andamento (PID %d).", pid)
                    return False
                logger.info("Lock stale (PID %d morto). Reclamando.", pid)
                # Remove o lock stale antes de tentar criar novo
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    pass
            except (ValueError, IOError):
                # Conteudo invalido no lock: remove e tenta criar novo
                try:
                    self.lock_path.unlink()
                except FileNotFoundError:
                    pass

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
                f"Download ja em andamento para {self.lock_path.stem}. "
                "Aguarde o processo anterior terminar."
            )
        return self

    def __exit__(self, *args):
        self.release()


class ModelDownloader:
    """Downloader retomavel para arquivos GGUF.

    Uso:
        dl = ModelDownloader()
        result = dl.download(
            url="https://huggingface.co/.../model.gguf",
            dest_path=Path("models/balanced/model.gguf"),
            expected_size=2_600_000_000,
            sha256="...",
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

    def download(
        self,
        url: str,
        dest_path: Path,
        expected_size: Optional[int] = None,
        sha256: Optional[str] = None,
        headers: Optional[dict] = None,
    ) -> DownloadResult:
        """Baixa um arquivo com resume automatico.

        Se o arquivo destino ja existe e esta completo (tamanho + sha256 OK),
        retorna sucesso imediatamente sem download.
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

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

        lock_path = dest_path.with_suffix(dest_path.suffix + ".lock")
        with DownloadLock(lock_path):
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
                        return DownloadResult(
                            success=True,
                            file_path=str(dest_path),
                            size_bytes=dest_path.stat().st_size,
                            sha256=(
                                self._compute_sha256(dest_path) if sha256 else None
                            ),
                            attempts=attempt,
                            resumed=resumed,
                            error="",
                        )
                    else:
                        logger.warning("Arquivo corrompido. Reiniciando download.")
                        try:
                            dest_path.unlink()
                        except FileNotFoundError:
                            pass
                        part_path.unlink(missing_ok=True)
                        total_bytes = 0
                        resumed = False
                        continue

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
                    backoff = min(
                        self.base_backoff * (2 ** (attempt - 1)), self.max_backoff
                    )
                    sleep_time = backoff + (0.1 * backoff * 0.5)
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
                cmd,
                capture_output=True,
                text=True,
                timeout=self.read_timeout + self.connect_timeout + 10,
            )
        except subprocess.TimeoutExpired:
            return DownloadResult(
                success=False, file_path=str(part_path), error="Timeout"
            )
        except FileNotFoundError:
            logger.error("curl.exe nao encontrado. Usando fallback urllib.")
            return self._urllib_download(url, part_path, resume, resume_bytes)

        if result.returncode == 0:
            return DownloadResult(success=True, file_path=str(part_path))
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