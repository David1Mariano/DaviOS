"""Debug: verifica o que o llama-server retorna."""

from __future__ import annotations

import json
import sys
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    base_url = "http://127.0.0.1:8080"

    # Teste 1: Sem modelo no payload
    print("=" * 60)
    print("Teste 1: Payload SEM campo 'model'")
    print("=" * 60)
    payload = {
        "messages": [
            {"role": "system", "content": "Voce e um assistente util. Responda em portugues de forma curta."},
            {"role": "user", "content": "Ola! Quem e voce?"},
        ],
        "max_tokens": 64,
        "temperature": 0.7,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            print(f"Resposta completa:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Erro: {e}")

    print()

    # Teste 2: Com modelo Qwen3-4B
    print("=" * 60)
    print("Teste 2: Payload COM campo 'model'")
    print("=" * 60)
    payload2 = {
        "model": "Qwen3-4B-Q4_K_M",
        "messages": [
            {"role": "system", "content": "Voce e um assistente util. Responda em portugues de forma curta."},
            {"role": "user", "content": "Ola! Quem e voce?"},
        ],
        "max_tokens": 64,
        "temperature": 0.7,
    }
    data2 = json.dumps(payload2).encode("utf-8")
    req2 = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data2,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req2, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)
            print(f"Resposta completa:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
    except Exception as e:
        print(f"Erro: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())