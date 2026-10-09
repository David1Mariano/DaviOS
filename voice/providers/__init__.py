"""Providers de TTS disponíveis."""

from voice.providers.base import TTSProvider, TTSProviderInfo
from voice.providers.piper_provider import PiperProvider
from voice.providers.sapi5_provider import SAPI5Provider

__all__ = [
    "TTSProvider",
    "TTSProviderInfo",
    "PiperProvider",
    "SAPI5Provider",
]

# Registry de providers disponíveis
AVAILABLE_PROVIDERS = {
    "piper": PiperProvider,
    "sapi5": SAPI5Provider,
}


def get_provider_class(name: str):
    """Retorna a classe do provider pelo nome."""
    return AVAILABLE_PROVIDERS.get(name.lower())


def list_available_providers() -> list[str]:
    """Lista nomes dos providers disponíveis."""
    return list(AVAILABLE_PROVIDERS.keys())