"""Configuração central do DaviOS."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


CONFIG_DIR = Path(__file__).resolve().parent


@dataclass
class DaviosConfig:
    """Configuração carregada de config/davios.json com defaults seguros."""

    offline_mode: bool = True
    network_policy: str = "offline"
    backend: str = "llama_cpp"
    models_dir: str = "models"
    profiles_file: str = "model_profiles.json"
    profile_override: Optional[str] = None  # LIGHT/BALANCED/PERFORMANCE
    debug: bool = False
    context_window: int = 2048
    temperature: float = 0.7
    max_tokens: int = 256
    threads: int = 0  # 0 = automático (usa cores físicos)
    gpu_layers: int = 0  # 0 = CPU; >0 = offload se o backend suportar
    use_gpu: bool = False
    max_recent_messages: int = 6
    max_memories_in_prompt: int = 5
    system_prompt: str = (
        "Voce e o DaviOS, um assistente pessoal local e offline. "
        "Responda sempre em portugues do Brasil, de forma curta, clara e natural. "
        "Voce nao pode executar comandos no computador, abrir programas ou "
        "acessar arquivos: se pedirem isso, explique educadamente que essa "
        "funcao ainda nao existe. Use as informacoes sobre o usuario quando "
        "forem fornecidas no contexto."
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
        known = {f for f in cls.__dataclass_fields__}
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
