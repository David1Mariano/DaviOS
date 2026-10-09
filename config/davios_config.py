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
    
    # Timeouts
    synthesis_timeout_seconds: float = 10.0
    playback_timeout_seconds: float = 30.0
    
    # Comportamento
    block_on_playback: bool = False  # se True, aguarda áudio terminar antes de continuar
    skip_empty_text: bool = True
    log_synthesis_errors: bool = True


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
    # C3: o LLM SABE que ferramentas existem (listagem informativa no prompt).
    # NÃO significa que ele possa EXECUTÁ-LAS — para isso veja actions_enabled
    # e web_enabled. Nenhum mecanismo de chamada existe nesta etapa.
    tools_visible_to_llm: bool = False
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
    max_memories_in_prompt: int = 5
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

    def profiles_path(self) -> Path:
        return CONFIG_DIR / self.profiles_file

    def models_path(self, project_root: Optional[Path] = None) -> Path:
        root = project_root or Path(__file__).resolve().parent.parent
        p = Path(self.models_dir)
        return p if p.is_absolute() else root / p
