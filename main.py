"""DaviOS - Assistente pessoal local e offline-first."""

import argparse
import logging
import os
import sys
from dataclasses import asdict
from dotenv import load_dotenv

# UTF-8 de ponta a ponta no Windows: sem isso o console usa codepage
# 437/850 e acentos como "ç" e "ã" corrompem no print()/input() (mojibake).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from brain.channels.telegram_channel import TelegramChannel
from brain.channels.telegram_client import TelegramClient
from brain.conversation_engine import ConversationEngine
from brain.core_runtime import CoreRuntime
from brain.providers.local_llm_provider import (
    LLAMA_CPP_INSTRUCTIONS,
    LocalLLMProvider,
)
from brain.providers.local_llama_cpp_provider import (
    LocalLlamaCppProvider,
    LLAMA_CPP_INSTRUCTIONS as LLAMA_STANDALONE_INSTRUCTIONS,
)
from brain.providers.online_router import HybridLLMProvider, OnlineRouter
from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareDetector
from core.reasoning import ReasoningEngine
from brain.model_manager import ModelManager
from voice.voice_manager import VoiceManager
from voice.config import VoiceConfig

log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir, "conversation.log"),
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
    model_manager.attach_provider(standalone)
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
        model_manager.attach_provider(provider)
        provider.initialize()  # tolerante: False se backend/modelo ausentes
        if provider.is_available():
            print("[BACKEND] llama-cpp-python (in-process) ativo.")
        else:
            print("[BACKEND] Nenhum backend LLM disponivel; usando modo regras.")
            print("[PROVIDER] Para instrucoes de instalacao, veja docs/LLM_SETUP.")

    # Router (AUTO): online primeiro com fallback local.
    # Respeita offline_mode e network_policy (consistente com OnlineRouter.is_network_allowed).
    network_allowed = config.router.enabled and not config.offline_mode and config.network_policy != "offline"
    if network_allowed:
        provider = HybridLLMProvider(
            config=config, local_provider=provider, router=OnlineRouter(config=config)
        )
        print("[ROUTER] modo auto: online primeiro, fallback local ativo.")

    from brain.cognitive_core import CognitiveCore

    # C5 (mudança mínima): registry + router das ferramentas, respeitando as
    # flags já existentes. O loop de execução só fica ativo quando
    # config.tools_visible_to_llm=True; com o padrão (False) o comportamento
    # é exatamente o de antes. O registro em si não habilita execução — a
    # REGRA DE OURO checa actions_enabled/web_enabled por categoria.
    from brain.action_manager import ActionManager
    from brain.phone_client import PhoneClient
    from brain.tool_adapters import register_default_tools
    from brain.tool_registry import ToolRegistry, ToolRouter
    from brain.web_access import WebAccess

    tools_registry = ToolRegistry()

    phone_client = PhoneClient(
        base_url=config.phone_base_url,
        timeout=config.phone_timeout,
    )

    register_default_tools(
        tools_registry,
        action_manager=ActionManager(enabled=config.actions_enabled),
        web_access=WebAccess(enabled=config.web_enabled),
        phone_client=phone_client,
    )

    cognitive = CognitiveCore(
        llm_provider=provider, config=config, tool_router=ToolRouter(tools_registry)
    )

    # O MemoryManager do engine e compartilhado com o CognitiveCore no
    # construtor do engine — memória persistente disponível ao LLM.
    # O ModelManager entra aqui para que pedidos explícitos de troca de modelo
    # ("Davi, troca para o Qwen 8B") cheguem a switch_active_model() e não a
    # um caminho paralelo.
    engine = ConversationEngine(cognitive_core=cognitive, model_manager=model_manager, config=config, tool_router=ToolRouter(tools_registry))
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
    
    # Status de voz
    voice_status = "ativada" if config.voice.enabled else "desativada"
    print(f"Voz: {voice_status} (primário={config.voice.primary_provider}, fallback={config.voice.fallback_provider})")
    print()
    print("DaviOS: Ola! Sou o DaviOS. Como posso ajudar?")
    print()


TOKEN_ENV = "DAVIOS_TELEGRAM_TOKEN"


def _resolve_telegram_token():
    """Resolve o token do bot SEM expor valor algum (precedencia do run_telegram).

    Variavel de ambiente DAVIOS_TELEGRAM_TOKEN vence o .env da raiz
    (override=False), exatamente como na entrada legada.
    """
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path, override=False)
    return os.environ.get(TOKEN_ENV)


def _start_telegram_channel(config, token, runtime, voice_config):
    """Inicia o canal Telegram apontado para o MESMO nucleo do console (D1-A/D3-A).

    O canal recebe o ``EngineProxy`` do runtime (nao um engine proprio):
    ``engine_factory`` devolve o proxy e ``engine_cleanup=None`` porque a
    limpeza do nucleo pertence ao runtime (rodará na thread core). A voz e o
    VoiceManager compartilhado protegido pelo lock do runtime (D6-A).
    Ferramentas: o engine unico foi construido com a config REAL (D2-A) — o
    canal NAO aplica effective_channel_config (isso vale apenas para a
    entrada legada run_telegram.py, que continua intacta; D4-A).
    """
    client = TelegramClient(token)
    channel = TelegramChannel(
        client=client,
        engine_factory=lambda: runtime.proxy,
        allowed_user_ids=list(config.telegram_allowed_user_ids or []),
        poll_timeout=30,
        engine_cleanup=None,  # o runtime e o dono da limpeza do nucleo
        voice_manager=runtime.voice,  # LockedVoice sobre o VoiceManager real (D6-A)
        voice_config=voice_config,
    )
    channel.start()
    return channel


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DaviOS - assistente pessoal local e offline-first"
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help=(
            "nao inicia o canal Telegram, mesmo com telegram_enabled=true "
            "e token definido (D1-A: no boot unificado o Telegram sobe automatico)"
        ),
    )
    args = parser.parse_args(argv)

    config = DaviosConfig.load()

    # --- Nucleo unico (Fase 3) --------------------------------------------
    # O engine (e a conexao SQLite do MemoryManager) nasce DENTRO da thread
    # core; console e canal Telegram falam com ele pelo EngineProxy. Um so
    # boot serve as duas interfaces (R2: um so provider/llama-server).
    extras: dict = {}

    def engine_factory():
        engine, hardware, model_manager, selection, provider = build_engine(config)
        extras["hardware"] = hardware
        extras["model_manager"] = model_manager
        extras["selection"] = selection
        extras["provider"] = provider
        return engine

    def final_cleanup():
        # Roda NA thread core no shutdown do runtime. Antes o console
        # descarregava o provider na propria main; agora isso e responsabili-
        # dade do runtime (mesma regra do canal: limpeza na thread dona).
        provider = extras.get("provider")
        if provider is not None:
            try:
                provider.unload()
            except Exception:
                logger.exception("Falha ao descarregar o provider")

    runtime = CoreRuntime(engine_factory, final_cleanup=final_cleanup)
    try:
        runtime.start()
    except Exception as exc:  # boot nunca quebra silenciosamente
        logger.exception("Falha no boot do DaviOS")
        print("DaviOS iniciado com recursos reduzidos.")
        if config.debug:
            print(f"[DEBUG] {type(exc).__name__}: {exc}")

        try:
            runtime.shutdown()  # Encerra o nucleo apos falha no boot.
        except Exception:
            logger.exception("Falha ao parar o nucleo apos boot com erro")

        extras = {}

        # Modo regras: o engine tambem roda na thread do CoreRuntime.
        # O console nao acessa o engine diretamente.
        runtime = CoreRuntime(lambda: ConversationEngine())
        runtime.start()

    # Inicializa a voz e a conecta ao runtime unificado.
    # A conversao preserva os campos da configuracao de voz.
    voice_config = VoiceConfig.from_dict(asdict(config.voice))
    voice_manager = VoiceManager(voice_config)

    voice_manager.initialize()
    runtime.voice_manager = voice_manager

    if extras:
        print_boot_banner(
            extras["hardware"],
            extras["model_manager"],
            extras["selection"],
            extras["provider"],
            config,
        )
    else:
        # Banner minimo do modo reduzido (o print_boot_banner exige
        # selection/provider reais; o caminho antigo passava None e crashava
        # em selection.profile — bug latente evitado aqui).
        print("DaviOS iniciado.")
        print()
        print("Status: modo regras (boot reduzido).")
        voice_status = "ativada" if config.voice.enabled else "desativada"
        print(f"Voz: {voice_status}")
        print()
        print("DaviOS: Ola! Sou o DaviOS. Como posso ajudar?")
        print()

    # --- Canal Telegram automatico (D1-A) ----------------------------------
    # O Ctrl+C do console NAO e substituido por signal handlers: o input()
    # continua dono do KeyboardInterrupt. run_telegram.py (legado, D4-A)
    # continua com os seus proprios handlers para uso como bot dedicado.
    channel = None
    channel_warned = False
    if args.no_telegram:
        print("Canal Telegram desativado (--no-telegram).")
    elif not config.telegram_enabled:
        print("Canal Telegram desativado (telegram_enabled=false).")
    else:
        token = _resolve_telegram_token()
        if not token:
            print(
                f"Canal Telegram desativado: token nao encontrado "
                f"({TOKEN_ENV} via variavel de ambiente ou .env)."
            )
        else:
            try:
                channel = _start_telegram_channel(
                    config, token, runtime, voice_config
                )
            except Exception as exc:  # canal nunca derruba o console
                logger.exception("Falha ao iniciar o canal Telegram")
                print(
                    f"AVISO: canal Telegram nao iniciou "
                    f"({type(exc).__name__}); o console continua ativo."
                )
                channel = None
            else:
                if not list(config.telegram_allowed_user_ids or []):
                    print(
                        "AVISO: whitelist vazia — o Telegram nao processara "
                        "nenhuma mensagem."
                    )
                print(
                    "Canal Telegram ativo (console + Telegram compartilham o "
                    "mesmo cerebro)."
                )

    try:
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
                # O proxy serializa na thread core — o MESMO caminho que o
                # Telegram usa (D3-A: historico/contexto unicos e intercalados).
                result = runtime.proxy.process(user_input)
            except Exception as e:
                logger.exception("Erro ao processar mensagem do usuario")
                if config.debug:
                    print(f"[DEBUG] {type(e).__name__}: {e}")
                print("DaviOS: Tive um problema ao processar essa mensagem.")
                print()
                continue

            print(f"DaviOS: {result.response}")
            print()

            # Sintese e reproducao de voz (nunca derruba a conversa). O
            # runtime.speak passa pelo lock de voz compartilhado (D6-A).
            try:
                if config.voice.enabled:
                    runtime.speak(result.response)
            except Exception as e:
                logger.exception("Erro no sistema de voz")
                if config.debug:
                    print(f"[DEBUG] Voice error: {type(e).__name__}: {e}")

            if channel is not None and channel.is_failed and not channel_warned:
                # R3/R11: falha do Telegram NAO derruba o console; avisa 1x.
                channel_warned = True
                print(
                    "AVISO: canal Telegram parou "
                    f"({channel.fail_reason or 'motivo desconhecido'}); "
                    "o console continua ativo."
                )
                print()

            if result.should_exit:
                break
    finally:
        # Encerramento deterministico (R8), tambem em Ctrl+C durante o
        # process: 1) para o canal (sem mensagens novas do Telegram);
        # 2) drena a ultima reproducao de voz; 3) runtime.shutdown():
        # sentinela, limpeza NA thread core (provider.unload encerra o
        # llama-server filho) e voz sob o lock compartilhado.
        try:
            if channel is not None:
                channel.stop()
                channel.join(5.0)
        except Exception:
            logger.exception("Falha ao parar o canal Telegram")
        try:
            voice_manager.wait_for_playback(timeout=5.0)
        except Exception:
            pass
        runtime.shutdown()
        print("DaviOS encerrado.")


if __name__ == "__main__":
    main()
