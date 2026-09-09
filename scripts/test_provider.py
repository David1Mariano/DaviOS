"""Testa o LocalLlamaCppProvider contra o llama-server local."""

from __future__ import annotations

import sys
from pathlib import Path

# Adiciona o diretorio do projeto ao path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.llm_provider import LLMRequest
from brain.model_manager import ModelManager
from brain.providers.local_llama_cpp_provider import LocalLlamaCppProvider
from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareDetector


def main() -> int:
    print("=" * 60)
    print("Teste do LocalLlamaCppProvider")
    print("=" * 60)
    print()

    # Carregar configuracao
    config = DaviosConfig.load()
    config.debug = True

    # Detectar hardware
    hw = HardwareDetector().detect()
    print(f"Hardware: {hw.cpu_name}")
    print(f"RAM: {hw.ram_total_gb} GB")
    print(f"GPU: {hw.gpu_name}")
    print()

    # Selecionar modelo
    mm = ModelManager(config)
    sel = mm.select_model(hw)
    print(f"Modelo selecionado: {sel.model.name if sel.model else None}")
    print(f"Perfil: {sel.profile}")
    print(f"Compativel: {sel.compatible}")
    print()

    # Criar provider
    provider = LocalLlamaCppProvider(config, mm, sel)
    print("Inicializando provider...")
    init_ok = provider.initialize()
    print(f"Init: {init_ok}")
    print(f"Health: {provider.health_check()}")
    print()

    if not init_ok:
        print("[FALHA] Provider nao inicializou.")
        return 1

    # Testar geracao
    print("Testando geracao de resposta...")
    request = LLMRequest(
        prompt="Ola DaviOS! Quem e voce? Responda em uma frase curta.",
        system="Voce e o DaviOS, um assistente pessoal local e offline. Responda de forma curta e direta.",
        max_tokens=512,
        temperature=0.7,
    )

    try:
        response = provider.generate(request)
        print()
        print("=" * 60)
        print("RESPOSTA DO MODELO:")
        print("=" * 60)
        print(response.text)
        print("=" * 60)
        print(f"Provider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Backend: {response.backend}")
        print(f"Tokens gerados: {response.tokens_generated}")
        print(f"Tempo: {response.elapsed_ms} ms")
        print()
        return 0
    except Exception as e:
        print(f"[FALHA] Erro ao gerar resposta: {type(e).__name__}: {e}")
        return 1
    finally:
        provider.unload()


if __name__ == "__main__":
    sys.exit(main())