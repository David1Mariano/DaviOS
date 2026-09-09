"""Script: baixa o modelo GGUF do DaviOS (por padrao Qwen3-4B-Q4_K_M).

Uso:
    .venv\\Scripts\\python.exe scripts\\download_model.py
    .venv\\Scripts\\python.exe scripts\\download_model.py --url <URL> --dest <caminho>

O download usa o downloader robusto (retomavel, .part, lock, checksum).
Nao inicia downloads duplicados e respeita um arquivo .part existente.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.model_downloader import ModelDownloader

# Modelo oficial Qwen3-4B-GGUF (Q4_K_M), fonte: https://huggingface.co/Qwen/Qwen3-4B-GGUF
DEFAULT_MODEL_FILENAME = "Qwen3-4B-Q4_K_M.gguf"
DEFAULT_MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/"
    "Qwen3-4B-Q4_K_M.gguf"
)
DEFAULT_DEST = "models/balanced"


def main():
    parser = argparse.ArgumentParser(description="Baixa modelo GGUF do DaviOS")
    parser.add_argument(
        "--url", default=DEFAULT_MODEL_URL,
        help="URL do arquivo GGUF (padrao: Qwen/Qwen3-4B-GGUF Q4_K_M).",
    )
    parser.add_argument(
        "--dest", default=str(ROOT / DEFAULT_DEST),
        help="Diretorio onde o modelo deve ficar (padrao: models/balanced).",
    )
    parser.add_argument(
        "--expected-size", type=int, default=None,
        help="Tamanho esperado em bytes (opcional).",
    )
    parser.add_argument(
        "--sha256", default=None,
        help="SHA256 esperado do arquivo (opcional).",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / DEFAULT_MODEL_FILENAME

    print(f"Modelo de referencia: {DEFAULT_MODEL_FILENAME}")
    print(f"Tamanho aproximado : ~2.5 GB (Q4_K_M)")
    print(f"Fonte              : Qwen/Qwen3-4B-GGUF (HuggingFace)")
    print(f"Destino            : {dest_path}")
    print()

    if dest_path.exists() and dest_path.stat().st_size > 0:
        print("Ja existe um arquivo em destino. Verificando...")
        # Se nao fornecermos expected-size/sha256, um arquivo existente
        # com mais de 1GB e considerado plausivel; o provider final valida.
        size = dest_path.stat().st_size
        if size > 1_000_000_000:
            print(f"Arquivo existente com {size} bytes; reutilizando.")
            return 0

    downloader = ModelDownloader(max_retries=12, base_backoff=2.0, max_backoff=40.0)
    result = downloader.download(
        url=args.url,
        dest_path=dest_path,
        expected_size=args.expected_size,
        sha256=args.sha256,
        headers={"User-Agent": "DaviOS/1.0"},
    )

    if result.success:
        print(f"Modelo pronto em: {result.file_path}")
        print(f"Tamanho        : {result.size_bytes} bytes")
        if result.sha256:
            print(f"SHA256        : {result.sha256}")
        print(f"Tentativas     : {result.attempts}")
        return 0

    print()
    print("O download automatico nao foi concluido.")
    print()
    print("Baixe o arquivo manualmente e coloque em:")
    print(f"  models/balanced/{DEFAULT_MODEL_FILENAME}")
    print()
    print("Fonte oficial:")
    print(f"  https://huggingface.co/Qwen/Qwen3-4B-GGUF")
    if dest_path.with_suffix(dest_path.suffix + ".part").exists():
        print()
        print("Um arquivo .part existe; um proximo download continuara dele.")
    return 1


if __name__ == "__main__":
    sys.exit(main())