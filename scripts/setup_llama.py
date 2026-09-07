"""Script: baixa e instala o llama.cpp standalone.

Uso:
    .venv\Scripts\python.exe scripts\setup_llama.py [--cpu]

O binário é instalado em bin/llama.cpp/.
Se já instalado, não faz nada.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Adiciona project root ao path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.llama_binary_downloader import LlamaBinaryDownloader


def main():
    parser = argparse.ArgumentParser(description="Instala o llama.cpp standalone")
    parser.add_argument(
        "--cpu", action="store_true",
        help="Forçar download da versão CPU-only (ignora Vulkan).",
    )
    args = parser.parse_args()

    downloader = LlamaBinaryDownloader()

    if downloader.is_installed():
        info = downloader.get_info()
        print(f"llama.cpp já instalado em: {downloader.bin_dir}")
        print(f"  Backend: {info.backend}")
        print(f"  llama-server: {info.server_exe}")
        print(f"  llama-cli: {info.cli_exe}")
        return 0

    prefer = "cpu" if args.cpu else "vulkan"
    print(f"Instalando llama.cpp (prefer: {prefer})...")
    try:
        info = downloader.install(prefer=prefer)
    except Exception as e:
        print(f"\nERRO: {e}")
        print(f"\nBaixe manualmente de:")
        print(f"  https://github.com/ggml-org/llama.cpp/releases")
        print(f"E extraia para: {downloader.bin_dir}")
        return 1

    print(f"\nllama.cpp instalado com sucesso!")
    print(f"  Diretório: {downloader.bin_dir}")
    print(f"  Backend: {info.backend}")
    print(f"  llama-server: {info.server_exe}")
    print(f"  llama-cli: {info.cli_exe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
