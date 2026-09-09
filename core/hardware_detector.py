"""Detecção de hardware do DaviOS.

Reúne informações da máquina (OS, CPU, RAM, GPU, disco, backend de
inferência) em um ``HardwareProfile``. Toda detecção é tolerante a falhas:
o que não puder ser descoberto vira ``None``/"unknown", nunca uma exceção.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

UNKNOWN = "unknown"

try:  # pragma: no cover - ambiente dependente
    import psutil  # type: ignore

    PSUTIL_AVAILABLE = True
except Exception:  # pragma: no cover
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False


@dataclass
class HardwareProfile:
    """Retrato da máquina em execução."""

    os: str = UNKNOWN
    architecture: str = UNKNOWN
    cpu_name: str = UNKNOWN
    cpu_cores: Optional[int] = None
    ram_total_gb: Optional[float] = None
    ram_available_gb: Optional[float] = None
    gpu_name: str = UNKNOWN
    gpu_vram_gb: Optional[float] = None
    gpu_vendor: str = UNKNOWN
    inference_backend: str = "none"
    disk_free_gb: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "os": self.os,
            "architecture": self.architecture,
            "cpu_name": self.cpu_name,
            "cpu_cores": self.cpu_cores,
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "gpu_vendor": self.gpu_vendor,
            "inference_backend": self.inference_backend,
            "disk_free_gb": self.disk_free_gb,
        }

    def summary_lines(self) -> list[str]:
        """Linhas curtas para exibição no boot."""
        cpu = (
            f"CPU: {self.cpu_name} ({self.cpu_cores or '?'} cores)"
            if self.cpu_name != UNKNOWN
            else f"CPU: {self.cpu_cores or '?'} cores"
        )
        ram = f"RAM: {self._fmt(self.ram_total_gb)} GB"
        if self.ram_available_gb is not None:
            ram += f" ({self._fmt(self.ram_available_gb)} GB livres)"
        if self.gpu_name != UNKNOWN:
            vram = (
                f", {self._fmt(self.gpu_vram_gb)} GB VRAM"
                if self.gpu_vram_gb is not None
                else ""
            )
            gpu = f"GPU: {self.gpu_name}{vram}"
        else:
            gpu = "GPU: nao detectada (usando CPU)"
        return [cpu, ram, gpu]

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        if value is None:
            return "?"
        return f"{value:.1f}".rstrip("0").rstrip(".")


def _run_command(cmd: list[str], timeout: float = 5.0) -> Optional[str]:
    """Executa um comando auxiliar sem nunca propagar exceção."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception:
        return None
    return None


def detect_inference_backend() -> str:
    """Descobre se existe backend de inferencia local instalado.

    Prioriza o backend standalone do llama.cpp (llama-server.exe em
    bin/llama.cpp/). Se nao, verifica llama-cpp-python (in-process).
    Retorna "none" quando nada esta instalado — o DaviOS continua em
    modo regras.
    """
    # 1. llama.cpp standalone (bin/llama.cpp/llama-server.exe)
    try:
        from utils.llama_binary_downloader import get_llama_bin_dir

        if (get_llama_bin_dir() / "llama-server.exe").exists():
            return "llama_cpp_standalone"
    except Exception:
        pass

    # 2. llama-cpp-python (in-process)
    try:
        import llama_cpp  # type: ignore  # noqa: F401

        return "llama_cpp"
    except Exception:
        return "none"
class HardwareDetector:
    """Detecta hardware da máquina atual."""

    def __init__(self, enable_gpu_probe: bool = True):
        self.enable_gpu_probe = enable_gpu_probe

    def detect_os(self) -> str:
        try:
            system = platform.system() or UNKNOWN
            if system == "Windows":
                return f"Windows {platform.release()}"
            if system == "Linux":
                return f"Linux {platform.release()}"
            if system == "Darwin":
                return f"macOS {platform.release()}"
            return system
        except Exception:
            return UNKNOWN

    def detect_architecture(self) -> str:
        try:
            machine = platform.machine() or UNKNOWN
            bits = platform.architecture()[0] or ""
            return f"{machine} {bits}".strip()
        except Exception:
            return UNKNOWN

    def detect_cpu_name(self) -> str:
        identifier = os.environ.get("PROCESSOR_IDENTIFIER")
        if identifier:
            return identifier.split(",")[0].strip()
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except Exception:
                pass
        return UNKNOWN

    def detect_cpu_cores(self) -> Optional[int]:
        try:
            cores = os.cpu_count()
            return int(cores) if cores else None
        except Exception:
            return None

    def detect_ram(self) -> tuple[Optional[float], Optional[float]]:
        """Retorna (total_gb, available_gb)."""
        if PSUTIL_AVAILABLE:
            try:
                vm = psutil.virtual_memory()
                return (
                    round(vm.total / (1024**3), 2),
                    round(vm.available / (1024**3), 2),
                )
            except Exception:
                pass
        try:
            if platform.system() == "Windows":
                out = _run_command(
                    [
                        "powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_ComputerSystem)"
                        ".TotalPhysicalMemory / 1GB",
                    ]
                )
                if out:
                    return round(float(out.strip()), 2), None
        except Exception:
            pass
        return None, None

    def detect_gpu(self) -> tuple[str, Optional[float], str]:
        """Retorna (gpu_name, vram_gb, vendor). "unknown"/None se indisponível."""
        if not self.enable_gpu_probe:
            return UNKNOWN, None, UNKNOWN
        if platform.system() == "Windows":
            return self._detect_gpu_windows()
        return self._detect_gpu_linux()

    def _detect_gpu_windows(self) -> tuple[str, Optional[float], str]:
        ps_script = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name, AdapterRAM | ConvertTo-Json"
        )
        out = _run_command(["powershell", "-NoProfile", "-Command", ps_script])
        if not out:
            return UNKNOWN, None, UNKNOWN
        try:
            data = json.loads(out)
            items = data if isinstance(data, list) else [data]
            best: dict[str, Any] | None = None
            best_ram = -1
            for item in items:
                ram = int(item.get("AdapterRAM") or 0)
                if ram > best_ram:
                    best, best_ram = item, ram
            if not best:
                return UNKNOWN, None, UNKNOWN
            name = (best.get("Name") or UNKNOWN).strip()
            vram_gb = round(best_ram / (1024**3), 2) if best_ram > 0 else None
            if vram_gb is not None and 3.9 <= vram_gb < 4.1:
                vram_gb = None  # AdapterRAM (Int32) satura perto de 4 GB
            return name, vram_gb, self._guess_vendor(name)
        except Exception:
            return UNKNOWN, None, UNKNOWN

    @staticmethod
    def _guess_vendor(gpu_name: str) -> str:
        lowered = gpu_name.lower()
        if "nvidia" in lowered or "geforce" in lowered:
            return "NVIDIA"
        if "radeon" in lowered or "amd" in lowered or "ati " in lowered:
            return "AMD"
        if "intel" in lowered:
            return "Intel"
        return UNKNOWN

    def _detect_gpu_linux(self) -> tuple[str, Optional[float], str]:
        out = _run_command(["lspci"])
        if out:
            for line in out.splitlines():
                low = line.lower()
                if "vga" in low or "3d controller" in low:
                    name = line.split(":", 2)[-1].strip()
                    return name, None, self._guess_vendor(name)
        return UNKNOWN, None, UNKNOWN

    def detect_disk_free(self, path: Optional[str] = None) -> Optional[float]:
        try:
            target = path or os.path.dirname(os.path.abspath(__file__))
            usage = shutil.disk_usage(target)
            return round(usage.free / (1024**3), 2)
        except Exception:
            return None

    def detect(self) -> HardwareProfile:
        """Executa todas as detecções e monta o perfil."""
        ram_total, ram_available = self.detect_ram()
        gpu_name, gpu_vram, gpu_vendor = self.detect_gpu()
        return HardwareProfile(
            os=self.detect_os(),
            architecture=self.detect_architecture(),
            cpu_name=self.detect_cpu_name(),
            cpu_cores=self.detect_cpu_cores(),
            ram_total_gb=ram_total,
            ram_available_gb=ram_available,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram,
            gpu_vendor=gpu_vendor,
            inference_backend=detect_inference_backend(),
            disk_free_gb=self.detect_disk_free(),
        )

