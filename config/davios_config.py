"""Configuração central do DaviOS."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


CONFIG_DIR = Path(__file__).resolve().parent


@dataclass
class VoiceConfig:
    """Configuração do sistema de voz/TTS."""
    
    # Controle geral
    enabled: bool = True
    primary_provider: str = "piper"      # "piper" | "sapi5"
    fallback_provider: str = "sapi5"     # "sapi5" | "none"
    
    # Piper
    piper_executable: str = "piper"
    piper_model_path: str = "models/voice/piper/pt_BR-faber-medium.onnx"
    piper_config_path: str = "models/voice/piper/pt_BR-faber-medium.onnx.json"
    piper_data_dir: str = ""
    piper_speaker_id: int = 0
    piper_length_scale: float = 1.0
    piper_noise_scale: float = 0.667
    piper_noise_w: float = 0.8
    
    # SAPI5
    sapi5_voice_name: str = ""  # vazio = auto-detect PT-BR
    sapi5_rate: int = 0         # -10 a 10
    sapi5_volume: int = 100     # 0 a 100
    
    # XTTS-v2 (voice cloning)
    xtts_enabled: bool = False
    xtts_model_path: str = "models/voice/xtts-v2"
    xtts_reference_audio: str = "models/voice/references/reference.wav"
    xtts_language: str = "pt"
    xtts_speaker_wav: str = ""
    xtts_temperature: float = 0.7
    xtts_length_penalty: float = 1.0
    xtts_repetition_penalty: float = 1.0
    xtts_top_k: int = 50
    xtts_top_p: float = 0.85
    xtts_gpt_cond_len: int = 64
    xtts_gpt_cond_chunk_len: int = 4
    xtts_gpt_batch_size: int = 1
    xtts_low_mem: bool = False
    xtts_use_cuda: bool = False
    
    # Timeouts
    synthesis_timeout_seconds: float = 10.0
    playback_timeout_seconds: float = 30.0
    
    # Comportamento
    block_on_playback: bool = False  # se True, aguarda áudio terminar antes de continuar
    skip_empty_text: bool = True
    log_synthesis_errors: bool = True

    # Telegram: envio ocasional de mensagens de voz (sendVoice)
    telegram_voice_enabled: bool = False          # opt-in: desligado por padrão
    telegram_voice_probability: float = 0.35      # chance por resposta elegível
    telegram_voice_min_chars: int = 3             # tamanho mínimo do texto (caracteres)
    telegram_voice_max_chars: int = 300           # tamanho máximo do texto (caracteres)


@dataclass
class RemoteProvider:
    """Provedor remoto compatível com a API OpenAI (/v1/chat/completions)."""

    name: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 8.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class RouterConfig:
    """Camada de roteamento entre provedores remotos e o provedor local."""

    enabled: bool = True
    mode: str = "auto"
    timeout_seconds: float = 8.0
    retry_count: int = 2
    retry_backoff_seconds: float = 1.0


@dataclass
class DaviosConfig:
    """Configuração carregada de config/davios.json com defaults seguros."""

    offline_mode: bool = True
    network_policy: str = "offline"
    backend: str = "llama_cpp"
    # --- Parte B: capacidades opt-in (todas deshabilitadas por defecto) ---
    # actions: permitir ejecutar comandos de una allowlist explicita.
    # web_enabled: permitir solicitudes HTTP salientes (requiere offline_mode=False).
    # agent_server_enabled: levantar socket local para orquestar agentes externos.
    actions_enabled: bool = False
    web_enabled: bool = False
    agent_server_enabled: bool = False
    # Telefone (servidor Termux Android) — opt-in, desabilitado por padrão
    phone_read_enabled: bool = False
    phone_act_enabled: bool = False
    phone_base_url: str = ""
    phone_timeout: float = 10.0
    # VectorCraft (criação de imagens vetoriais) — opt-in, desabilitado por padrão
    vectorcraft_enabled: bool = False
    vectorcraft_cli_path: str = ""
    vectorcraft_mode: str = "headless"
    vectorcraft_timeout: float = 60.0
    vectorcraft_output_dir: str = ""
    # e web_enabled. Nenhum mecanismo de chamada existe nesta etapa.
    tools_visible_to_llm: bool = False
    # --- Canal Telegram (opt-in, tudo desligado por padrão) ---------------
    telegram_enabled: bool = False
    #                           liga/desliga o canal (default False).
    #                           run_telegram.py recusa iniciar o bot quando
    #                           False (apenas --check continua permitido).
    telegram_allowed_user_ids: list[int] = field(default_factory=list)
    #                           VAZIA = ninguém autorizado (nunca "liberar
    #                           tudo" por omissão). O token NÃO vive aqui —
    #                           vem só de os.environ["DAVIOS_TELEGRAM_TOKEN"].
    telegram_tools_enabled: bool = False
    #                           ferramentas (ActionManager/WebAccess) DESLIGADAS.
    owner_user_id: Optional[int] = None
    # Caminho opcional do llama-server.exe (backend standalone). Quando None,
    # o provedor resolve os locais padrão (bin/llama.cpp/ e depois o PATH).
    llama_server_exe: Optional[str] = None
    # Diretório externo de modelos (opcional). Quando informado, o
    # ModelManager.discover_models() também varre esse diretório em busca de
    # arquivos .gguf. Nunca é padrão obrigatório; None desativa a varredura
    # externa.
    external_models_dir: Optional[str] = None
    web_timeout_seconds: float = 5.0
    web_max_content_bytes: int = 200_000
    agent_server_host: str = "127.0.0.1"
    agent_server_port: int = 8765
    # Model management
    active_model_id: Optional[str] = None  # ID do modelo ativo no catálogo
    models_dir: str = "models"
    profiles_file: str = "model_profiles.json"
    profile_override: Optional[str] = None  # LIGHT/BALANCED/PERFORMANCE
    debug: bool = False
    context_window: int = 2048
    temperature: float = 0.7
    repeat_penalty: float = 1.3
    repeat_last_n: int = 256
    max_tokens: int = 256
    threads: int = 0  # 0 = automático (usa cores físicos)
    gpu_layers: int = 0  # 0 = CPU; >0 = offload se o backend suportar
    use_gpu: bool = False
    max_recent_messages: int = 6
    max_history_messages: int = 20
    max_memories_in_prompt: int = 5
    # Provedores remotos (OpenAI-compatible): de davios.json ou
    # DAVIOS_PROVIDER_N_* (credenciais em variáveis de ambiente).
    remote_providers: list[RemoteProvider] = field(default_factory=list)
    # Roteamento: online primeiro + fallback local.
    router: RouterConfig = field(default_factory=RouterConfig)
    # Voice/TTS
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    # system_prompt usado quando tools_visible_to_llm=False (padrao). Contem
    # a frase que nega categoricamente a capacidade de executar comandos,
    # abrir programas ou acessar arquivos — so e contraditoria se tools_visible_to_llm=True.
    system_prompt: str = (
        "Voce e o DaviOS, um assistente pessoal local e offline. "
        "Responda sempre em portugues do Brasil, de forma curta, clara e natural. "
        "Voce nao pode executar comandos no computador, abrir programas ou "
        "acessar arquivos: se pedirem isso, explique educadamente que essa "
        "funcao ainda nao existe. Use as informacoes sobre o usuario quando "
        "forem fornecidas no contexto."
    )
    # system_prompt usado quando tools_visible_to_llm=True. Contem a mesma
    # identidade/idioma/tom, mas SEM a frase de negação de capacidades (que
    # conflitaria com a seção de ferramentas). O PromptBuilder escolhe
    # automaticamente entre system_prompt e system_prompt_tools_enabled.
    system_prompt_tools_enabled: str = (
        "Voce e o DaviOS, um assistente pessoal local e offline. "
        "Responda sempre em portugues do Brasil, de forma curta, clara e natural. "
        "Use as informacoes sobre o usuario quando forem fornecidas no contexto."
    )

    @classmethod
    def load(cls, path: Optional[str] = None) -> "DaviosConfig":
        """Carrega config/davios.json; defaults se o arquivo não existir."""
        config_path = Path(path) if path else CONFIG_DIR / "davios.json"
        data: dict[str, Any] = {}
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        
        # Converte campos de dataclass aninhados (ex: voice)
        fields = cls.__dataclass_fields__
        for field_name, field_def in fields.items():
            if field_name in data and hasattr(field_def.type, '__dataclass_fields__'):
                # É um dataclass aninhado
                nested_cls = field_def.type
                if isinstance(data[field_name], dict):
                    data[field_name] = nested_cls(**data[field_name])
        
        known = {f for f in fields}
        filtered = {k: v for k, v in data.items() if k in known}
        config = cls(**filtered)
        config.apply_env_overrides()
        return config

    def apply_env_overrides(self) -> None:
        """Permite override por variáveis de ambiente (DAVIOS_*)."""
        env_debug = os.environ.get("DAVIOS_DEBUG")
        if env_debug is not None:
            self.debug = env_debug == "1"
        env_profile = os.environ.get("DAVIOS_PROFILE")
        if env_profile:
            self.profile_override = env_profile.upper()
        env_gpu = os.environ.get("DAVIOS_USE_GPU")
        if env_gpu is not None:
            self.use_gpu = env_gpu == "1"


        # Telefone (servidor Termux Android) | opt-in via ambiente
        env_phone_read = os.environ.get("DAVIOS_PHONE_READ_ENABLED")
        if env_phone_read is not None:
            self.phone_read_enabled = env_phone_read == "1"

        env_phone_act = os.environ.get("DAVIOS_PHONE_ACT_ENABLED")
        if env_phone_act is not None:
            self.phone_act_enabled = env_phone_act == "1"

        env_phone_base_url = os.environ.get("DAVIOS_PHONE_BASE_URL")
        if env_phone_base_url:
            self.phone_base_url = env_phone_base_url

        env_phone_timeout = os.environ.get("DAVIOS_PHONE_TIMEOUT")
        if env_phone_timeout:
            self.phone_timeout = float(env_phone_timeout)

        # Configura modo offline e política de rede via ambiente
        # ONLINE exige offline_mode=0 e network_policy=auto.
        env_offline = os.environ.get("DAVIOS_OFFLINE_MODE")
        if env_offline is not None:
            self.offline_mode = env_offline == "1"

        env_policy = os.environ.get("DAVIOS_NETWORK_POLICY")
        if env_policy:
            self.network_policy = env_policy.lower()

        env_router_enabled = os.environ.get("DAVIOS_ROUTER_ENABLED")
        if env_router_enabled is not None:
            self.router.enabled = env_router_enabled == "1"

        env_router_mode = os.environ.get("DAVIOS_ROUTER_MODE")
        if env_router_mode:
            self.router.mode = env_router_mode.lower()

        env_router_timeout = os.environ.get("DAVIOS_ROUTER_TIMEOUT")
        if env_router_timeout:
            self.router.timeout_seconds = float(env_router_timeout)

        env_router_retry = os.environ.get("DAVIOS_ROUTER_RETRY_COUNT")
        if env_router_retry:
            self.router.retry_count = int(env_router_retry)

        env_router_backoff = os.environ.get("DAVIOS_ROUTER_BACKOFF_SECONDS")
        if env_router_backoff:
            self.router.retry_backoff_seconds = float(env_router_backoff)

        # Provedores remotos via DAVIOS_PROVIDER_N_
        # A lista termina quando BASE_URL estiver ausente ou vazia.
        env_provider_index = 0
        while True:
            prefix = "DAVIOS_PROVIDER_" + str(env_provider_index) + "_"
            base_url = os.environ.get(prefix + "BASE_URL")
            if not base_url:
                break

            headers_raw = os.environ.get(prefix + "HEADERS") or ""
            headers: dict[str, str] = {}

            for item in headers_raw.split(","):
                if ":" in item:
                    key, value = item.split(":", 1)
                    headers[key.strip()] = value.strip()

            self.remote_providers.append(
                RemoteProvider(
                    name=os.environ.get(prefix + "NAME")
                    or "provider_" + str(env_provider_index),
                    base_url=base_url,
                    api_key=os.environ.get(prefix + "API_KEY") or "",
                    model=os.environ.get(prefix + "MODEL") or "",
                    timeout_seconds=float(
                        os.environ.get(prefix + "TIMEOUT_SECONDS")
                        or self.router.timeout_seconds
                    ),
                    extra_headers=headers,
                    enabled=os.environ.get(prefix + "ENABLED") != "0",
                )
            )
            env_provider_index += 1

    def profiles_path(self) -> Path:
        return CONFIG_DIR / self.profiles_file

    def models_path(self, project_root: Optional[Path] = None) -> Path:
        root = project_root or Path(__file__).resolve().parent.parent
        p = Path(self.models_dir)
        return p if p.is_absolute() else root / p
