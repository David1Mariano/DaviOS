"""Testa a conexao com o llama-server local.

Uso:
    .venv\\Scripts\\python.exe scripts\\test_llama_server.py

Verifica:
    - /health
    - /v1/chat/completions (POST)
"""

from __future__ import annotations

import json
import sys
import urllib.request
from urllib.error import HTTPError, URLError

BASE_URL = "http://127.0.0.1:8080"
TIMEOUT = 30


def check_health() -> bool:
    """Verifica se o servidor esta saudavel."""
    url = f"{BASE_URL}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
            print(f"[OK] /health -> {body}")
            return True
    except HTTPError as e:
        print(f"[FALHA] /health retornou HTTP {e.code}")
        return False
    except (URLError, OSError) as e:
        print(f"[FALHA] Nao foi possivel conectar a {url}")
        print(f"        Erro: {e}")
        return False


def test_chat() -> bool:
    """Testa o endpoint de chat completions."""
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "Qwen3-4B-Q4_K_M",
        "messages": [
            {
                "role": "system",
                "content": "Voce e um assistente util. Responda em portugues de forma curta.",
            },
            {
                "role": "user",
                "content": "Ola. Meu nome e Davi. Responda em uma frase curta.",
            },
        ],
        "max_tokens": 128,
        "temperature": 0.7,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        print(f"[INFO] Enviando requisicao POST para {url}...")
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            result = json.loads(body)

            # Extrai a resposta
            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                print(f"[OK] Resposta do modelo:\n{content}")

                # Mostra metadados
                usage = result.get("usage", {})
                if usage:
                    print(
                        f"[INFO] Tokens: prompt={usage.get('prompt_tokens', '?')}, "
                        f"completion={usage.get('completion_tokens', '?')}"
                    )
                return True
            else:
                print(f"[FALHA] Resposta sem choices: {body[:500]}")
                return False
    except HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"[FALHA] HTTP {e.code}: {error_body[:500]}")
        return False
    except (URLError, OSError) as e:
        print(f"[FALHA] Erro de conexao: {e}")
        return False
    except Exception as e:
        print(f"[FALHA] Erro inesperado: {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("Teste do llama-server local")
    print("=" * 60)
    print()

    # 1. Verificar health
    print("1. Verificando /health...")
    if not check_health():
        print()
        print("llama-server nao esta rodando em 127.0.0.1:8080.")
        print("Inicie o servidor antes de testar.")
        return 1

    print()
    print("2. Testando /v1/chat/completions...")
    if not test_chat():
        return 1

    print()
    print("=" * 60)
    print("Todos os testes passaram!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())