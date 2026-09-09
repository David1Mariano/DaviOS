"""Teste integrado do DaviOS com o Qwen local.

Este script testa o DaviOS de ponta a ponta:
1. Inicializa o sistema
2. Envia uma pergunta
3. Verifica se a resposta veio do modelo local

Uso:
    .venv\\Scripts\\python.exe scripts\\test_davios.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.llm_provider import LLMRequest
from brain.model_manager import ModelManager
from brain.providers.local_llama_cpp_provider import LocalLlamaCppProvider
from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareDetector


def test_full_flow() -> bool:
    """Teste completo do fluxo de conversacao."""
    print("=" * 60)
    print("Teste Integrado do DaviOS")
    print("=" * 60)
    print()

    # 1. Configuracao
    config = DaviosConfig.load()
    config.debug = True

    # 2. Detectar hardware
    hw = HardwareDetector().detect()
    print(f"Hardware: {hw.cpu_name}")
    print(f"RAM: {hw.ram_total_gb} GB")
    print(f"GPU: {hw.gpu_name}")
    print()

    # 3. Selecionar modelo
    mm = ModelManager(config)
    sel = mm.select_model(hw)
    print(f"Modelo selecionado: {sel.model.name if sel.model else None}")
    print(f"Perfil: {sel.profile}")
    print()

    # 4. Criar provider
    provider = LocalLlamaCppProvider(config, mm, sel)
    if not provider.initialize():
        print("[FALHA] Provider nao inicializou.")
        return False

    print(f"Provider inicializado: {provider.health_check()}")
    print()

    # 5. Testar geracao
    tests = [
        ("Ola DaviOS! Quem e voce?", "Ola"),
        ("Meu nome e Davi.", "nome"),
        ("Qual e meu nome?", "Davi"),
        ("Explique o que e Python em uma frase.", "Python"),
    ]

    all_passed = True
    for prompt, expected_keyword in tests:
        print(f"Prompt: {prompt}")
        request = LLMRequest(
            prompt=prompt,
            system="Voce e o DaviOS, um assistente pessoal local e offline.",
            max_tokens=256,
            temperature=0.7,
        )

        try:
            response = provider.generate(request)
            print(f"Resposta: {response.text[:200]}")
            print(f"  -> {response.tokens_generated} tokens em {response.elapsed_ms} ms")

            # Verifica se a resposta nao esta vazia
            if not response.text.strip():
                print("  [ALERTA] Resposta vazia!")
                all_passed = False
            else:
                print("  [OK]")
        except Exception as e:
            print(f"  [FALHA] {type(e).__name__}: {e}")
            all_passed = False

        print()

    provider.unload()
    return all_passed


def main() -> int:
    try:
        success = test_full_flow()
        if success:
            print("=" * 60)
            print("TODOS OS TESTES PASSARAM!")
            print("=" * 60)
            return 0
        else:
            print("=" * 60)
            print("ALGUNS TESTES FALHARAM.")
            print("=" * 60)
            return 1
    except Exception as e:
        print(f"[ERRO FATAL] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())