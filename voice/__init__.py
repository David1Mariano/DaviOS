"""Módulo de voz/TTS do DaviOS."""

from voice.config import VoiceConfig
from voice.voice_manager import VoiceManager
from voice.providers.base import TTSProvider, TTSProviderInfo
from voice.providers import get_provider_class, list_available_providers

__all__ = [
    "VoiceConfig",
    "VoiceManager",
    "TTSProvider",
    "TTSProviderInfo",
    "get_provider_class",
    "list_available_providers",
]

# Versão do módulo de voz
__version__ = "1.0.0"