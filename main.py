"""DaviOS - Assistente pessoal local e offline-first."""

import logging
import os

from brain.conversation_engine import ConversationEngine
from brain.providers.local_llm_provider import (
    LLAMA_CPP_INSTRUCTIONS,
    LocalLLMProvider,
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
    """Monta hardware → modelo → provider → cognitive core → engine."""
    hardware = HardwareDetector().detect()
    model_manager = ModelManager(config)
    selection = model_manager.select_model(hardware)

    provider = LocalLLMProvider(
        config=config, model_manager=model_manager, selection=selection
    )
    provider.initialize()  # tolerante: False se backend/modelo ausentes

    from brain.cognitive_core import CognitiveCore

    cognitive = CognitiveCore(llm_provider=provider, config=config)

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
