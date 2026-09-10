"""DaviOS - Assistente pessoal local e offline-first."""

import logging
import os

from brain.conversation_engine import ConversationEngine
from brain.providers.local_llm_provider import (
    LLAMA_CPP_INSTRUCTIONS,
    LocalLLMProvider,
)
from brain.providers.local_llama_cpp_provider import (
    LocalLlamaCppProvider,
    LLAMA_CPP_INSTRUCTIONS as LLAMA_STANDALONE_INSTRUCTIONS,
)
from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareDetector
from core.reasoning import ReasoningEngine
from brain.model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(os.path.dirname(__file__), "logs", "conversation.log"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("davios.main")


def build_engine(config: DaviosConfig):
    """Monta hardware → modelo → provider → cognitive core → engine.

    Tenta primeiro o backend standalone (llama-server.exe, mais robusto e
    sem compilar llama-cpp-python). Se nao disponivel, usa LocalLLMProvider
    (llama-cpp-python) como fallback. Ambos respeitam a interface LLMProvider.
    """
    hardware = HardwareDetector().detect()
    model_manager = ModelManager(config)
    selection = model_manager.select_model(hardware)

    # 1. Backend standalone (llama.cpp binario) — prioridade
    standalone = LocalLlamaCppProvider(
        config=config, model_manager=model_manager, selection=selection
    )
    if standalone.initialize():
        provider = standalone
        print(
            f"[BACKEND] llama.cpp standalone ativo "
            f"({standalone.health_check().get('backend')})."
        )
    else:
        # Diagnostico explicito: nunca cair para regras em silencio quando
        # modelo e binario existem. Reporta a causa real.
        diagnostics = getattr(standalone, "diagnostics", {})
        if diagnostics:
            print("[BACKEND] llama.cpp standalone indisponivel. Causas:")
            for key, value in diagnostics.items():
                print(f"  {key} {value}")
        # 2. Fallback: llama-cpp-python in-process
        provider = LocalLLMProvider(
            config=config, model_manager=model_manager, selection=selection
        )
        provider.initialize()  # tolerante: False se backend/modelo ausentes
        if provider.is_available():
            print("[BACKEND] llama-cpp-python (in-process) ativo.")
        else:
            print("[BACKEND] Nenhum backend LLM disponivel; usando modo regras.")
            print("[PROVIDER] Para instrucoes de instalacao, veja docs/LLM_SETUP.")

    from brain.cognitive_core import CognitiveCore

    # C5 (mudança mínima): registry + router das ferramentas, respeitando as
    # flags já existentes. O loop de execução só fica ativo quando
    # config.tools_visible_to_llm=True; com o padrão (False) o comportamento
    # é exatamente o de antes. O registro em si não habilita execução — a
    # REGRA DE OURO checa actions_enabled/web_enabled por categoria.
    from brain.action_manager import ActionManager
    from brain.tool_adapters import register_default_tools
    from brain.tool_registry import ToolRegistry, ToolRouter
    from brain.web_access import WebAccess

    tools_registry = ToolRegistry()
    register_default_tools(
        tools_registry,
        action_manager=ActionManager(enabled=config.actions_enabled),
        web_access=WebAccess(enabled=config.web_enabled),
    )

    cognitive = CognitiveCore(
        llm_provider=provider, config=config, tool_router=ToolRouter(tools_registry)
    )

    # O MemoryManager do engine e compartilhado com o CognitiveCore no
    # construtor do engine — memória persistente disponível ao LLM.
    engine = ConversationEngine(cognitive_core=cognitive)
    return engine, hardware, model_manager, selection, provider


def print_boot_banner(hardware, model_manager, selection, provider, config):
    profile = selection.profile
    print("DaviOS iniciado.")
    print()
    if config.debug:
        print("Hardware:")
        for line in hardware.summary_lines():
            print(f"  {line}")
        print(f"  Disco livre: {hardware.disk_free_gb} GB")
        print(f"  Backend de inferencia: {hardware.inference_backend}")
        print()
        print(f"Perfil de hardware: {profile}")
        print(f"Modelo: {model_manager.status_summary(selection)}")
        print(f"Backend ativo: {provider.health_check().get('backend', 'none')}")
        print(
            "Status: Modelo local carregado."
            if provider.is_available()
            else "Status: modo regras (sem modelo local)."
        )
        print()
    elif not provider.is_available():
        print("Hardware:")
        for line in hardware.summary_lines():
            print(f"  {line}")
        print()
        print("Status: sem modelo local; usando modo regras.")
        print("Causa (diagnostico de boot):")
        print("  [BACKEND] nenhum backend LLM disponivel nesta maquina.")
        print("  " + LLAMA_STANDALONE_INSTRUCTIONS)
        print()
    else:
        health = provider.health_check()
        print(f"Cerebro local: {health.get('model', 'modelo local')} "
              f"via {health.get('backend', 'llama.cpp')} (localhost, offline).")
        print("Memoria persistente: ativa (SQLite).")
        print()
    print("DaviOS: Ola! Sou o DaviOS. Como posso ajudar?")
    print()


def main():
    config = DaviosConfig.load()
    try:
        engine, hardware, model_manager, selection, provider = build_engine(config)
    except Exception as exc:  # boot nunca quebra silenciosamente
        logger.exception("Falha no boot do DaviOS")
        print("DaviOS iniciado com recursos reduzidos.")
        if config.debug:
            print(f"[DEBUG] {type(exc).__name__}: {exc}")
        engine = ConversationEngine()
        provider = None
        hardware = None
        selection = None
        model_manager = None

    print_boot_banner(hardware, model_manager, selection, provider, config)

    while True:
        try:
            user_input = input("Voce: ")
        except EOFError:
            print()
            print("DaviOS: Ate mais!")
            break
        except KeyboardInterrupt:
            print()
            print("DaviOS: Ate mais!")
            break

        if len(user_input) > 2000:
            print("DaviOS: Essa mensagem e muito longa. Pode resumir?")
            print()
            continue

        try:
            result = engine.process(user_input)
        except Exception as e:
            logger.exception("Erro ao processar mensagem do usuario")
            if config.debug:
                print(f"[DEBUG] {type(e).__name__}: {e}")
            print("DaviOS: Tive um problema ao processar essa mensagem.")
            print()
            continue

        print(f"DaviOS: {result.response}")
        print()

        if result.should_exit:
            break

    if provider is not None:
        provider.unload()
    print("DaviOS encerrado.")


if __name__ == "__main__":
    main()
