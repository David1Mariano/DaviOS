"""Interface base para providers de TTS (Text-to-Speech)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger("davios.voice.providers")


@dataclass(frozen=True)
class TTSProviderInfo:
    """Informações sobre o estado e capacidades do provider."""
    name: str
    available: bool
    reason: str = ""
    voice_name: str = ""
    sample_rate: int = 0


class TTSProvider(ABC):
    """Interface base para síntese de fala."""

    def __init__(self, config: dict):
        self._config = config
        self._initialized = False
        self._voice_name = config.get("voice_name", "")
        self._sample_rate = config.get("sample_rate", 0)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Nome identificador do provider (ex: 'piper', 'sapi5')."""
        pass

    @abstractmethod
    def check_availability(self) -> TTSProviderInfo:
        """Verifica se o provider está disponível e pronto para uso.
        
        Returns:
            TTSProviderInfo com available=True se pronto, False caso contrário.
        """
        pass

    @abstractmethod
    def synthesize(self, text: str) -> Optional[bytes]:
        """Sintetiza texto em áudio PCM/WAV.
        
        Args:
            text: Texto a ser sintetizado.
            
        Returns:
            Bytes do áudio (WAV/PCM) ou None se falhar.
        """
        pass

    def initialize(self) -> bool:
        """Inicialização preguiçosa. Called once before first use."""
        if self._initialized:
            return True
        try:
            info = self.check_availability()
            self._initialized = info.available
            if self._initialized:
                self._voice_name = info.voice_name or self._voice_name
                self._sample_rate = info.sample_rate or self._sample_rate
                logger.info("[%s] Provider inicializado: voice=%s, sr=%d", 
                           self.provider_name, self._voice_name, self._sample_rate)
            else:
                logger.warning("[%s] Provider indisponível: %s", 
                              self.provider_name, info.reason)
            return self._initialized
        except Exception as e:
            logger.exception("[%s] Falha na inicialização", self.provider_name)
            self._initialized = False
            return False

    def is_initialized(self) -> bool:
        return self._initialized

    def get_voice_name(self) -> str:
        return self._voice_name

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def shutdown(self) -> None:
        """Limpeza opcional de recursos."""
        self._initialized = False