"""Gerenciador de voz/TTS do DaviOS - orquestra providers e reprodução."""

from __future__ import annotations

import logging
from typing import Optional

from voice.config import VoiceConfig
from voice.providers import get_provider_class, list_available_providers
from voice.providers.base import TTSProvider, TTSProviderInfo
from voice.audio_player import AudioPlayer


logger = logging.getLogger("davios.voice.manager")


class VoiceManager:
    """Orquestra síntese de fala com fallback e reprodução.
    
    Fluxo:
    1. Tenta provider primário
    2. Se falhar, tenta fallback configurado
    3. Se ambos falharem, retorna None (texto preservado)
    4. Reproduz áudio se síntese bem-sucedida
    """
    
    def __init__(self, config: VoiceConfig):
        self._config = config
        self._providers: dict[str, TTSProvider] = {}
        self._player = AudioPlayer({
            "playback_timeout_seconds": config.playback_timeout_seconds,
        })
        self._initialized = False
        self._last_error: str = ""
        self._fallback_attempted = False

    def initialize(self) -> bool:
        """Inicializa providers preguiçosamente (no primeiro uso real)."""
        if self._initialized:
            return True
        
        if not self._config.enabled:
            logger.info("[VoiceManager] Voz desativada na configuração")
            self._initialized = True  # Marca como inicializado para não tentar de novo
            return False
        
        # Pré-verifica disponibilidade dos providers configurados
        primary_info = self._check_provider(self._config.primary_provider)
        if primary_info.available:
            logger.info("[VoiceManager] Provider primário pronto: %s (%s)",
                       self._config.primary_provider, primary_info.voice_name)
        else:
            logger.warning("[VoiceManager] Provider primário indisponível: %s - %s",
                          self._config.primary_provider, primary_info.reason)

        fallback_info = self._check_provider(self._config.fallback_provider)
        if fallback_info.available:
            logger.info("[VoiceManager] Provider de fallback pronto: %s (%s)",
                       self._config.fallback_provider, fallback_info.voice_name)
        elif self._config.fallback_provider != "none":
            logger.warning("[VoiceManager] Provider de fallback indisponível: %s - %s",
                          self._config.fallback_provider, fallback_info.reason)

        self._initialized = True
        return True

    def _check_provider(self, name: str) -> TTSProviderInfo:
        """Verifica disponibilidade de um provider (cria instância temporária)."""
        if name == "none":
            return TTSProviderInfo(name="none", available=False, reason="Desativado")
        
        provider_class = get_provider_class(name)
        if not provider_class:
            return TTSProviderInfo(name=name, available=False, reason="Provider desconhecido")
        
        provider_config = self._config.get_provider_config(name)
        provider = provider_class(provider_config)
        return provider.check_availability()

    def _get_or_create_provider(self, name: str) -> Optional[TTSProvider]:
        """Obtém provider instanciado ou cria novo."""
        if name == "none":
            return None
            
        if name in self._providers:
            return self._providers[name]
        
        provider_class = get_provider_class(name)
        if not provider_class:
            logger.error("[VoiceManager] Provider desconhecido: %s", name)
            return None
        
        provider_config = self._config.get_provider_config(name)
        provider = provider_class(provider_config)
        self._providers[name] = provider
        return provider

    def synthesize(self, text: str) -> Optional[bytes]:
        """Sintetiza texto usando provider primário com fallback.
        
        Args:
            text: Texto a sintetizar.
            
        Returns:
            Bytes de áudio WAV ou None se falhar (texto preservado pelo caller).
        """
        if not self._config.enabled:
            logger.debug("[VoiceManager] Voz desativada, pulando síntese")
            return None

        if self._config.skip_empty_text and (not text or not text.strip()):
            logger.debug("[VoiceManager] Texto vazio, pulando síntese")
            return None

        self.initialize()
        self._last_error = ""
        self._fallback_attempted = False

        # Tenta provider primário
        audio = self._try_synthesize(self._config.primary_provider, text)
        if audio is not None:
            return audio

        # Fallback
        if self._config.fallback_provider != "none":
            self._fallback_attempted = True
            logger.info("[VoiceManager] Tentando fallback: %s", self._config.fallback_provider)
            audio = self._try_synthesize(self._config.fallback_provider, text)
            if audio is not None:
                logger.info("[VoiceManager] Fallback bem-sucedido: %s", self._config.fallback_provider)
                return audio

        logger.error("[VoiceManager] Todos os providers falharam. Último erro: %s", self._last_error)
        return None

    def _try_synthesize(self, provider_name: str, text: str) -> Optional[bytes]:
        """Tenta sintetizar com um provider específico."""
        provider = self._get_or_create_provider(provider_name)
        if not provider:
            self._last_error = f"Provider {provider_name} não disponível"
            return None

        if not provider.is_initialized():
            if not provider.initialize():
                self._last_error = f"Falha ao inicializar {provider_name}"
                return None

        try:
            audio = provider.synthesize(text)
            if audio is not None:
                return audio
            self._last_error = f"Síntese retornou None ({provider_name})"
        except Exception as e:
            self._last_error = f"Exceção em {provider_name}: {type(e).__name__}"
            if self._config.log_synthesis_errors:
                logger.exception("[VoiceManager] Erro em %s", provider_name)
        
        return None

    def speak(self, text: str) -> bool:
        """Sintetiza e reproduz texto.
        
        Args:
            text: Texto a falar.
            
        Returns:
            True se áudio foi reproduzido (ou voz desativada), False se falhou.
        """
        if not self._config.enabled:
            return True  # Não é erro, apenas desativado

        audio = self.synthesize(text)
        if audio is None:
            return False

        try:
            success = self._player.play(audio, blocking=self._config.block_on_playback)
            return success
        except Exception as e:
            logger.exception("[VoiceManager] Erro na reprodução")
            return False

    def speak_async(self, text: str) -> bool:
        """Sintetiza e reproduz sem bloquear (fire-and-forget).
        
        Retorna imediatamente após iniciar reprodução.
        """
        if not self._config.enabled:
            return True

        audio = self.synthesize(text)
        if audio is None:
            return False

        try:
            return self._player.play(audio, blocking=False)
        except Exception as e:
            logger.exception("[VoiceManager] Erro ao iniciar reprodução assíncrona")
            return False

    def wait_for_playback(self, timeout: Optional[float] = None) -> bool:
        """Aguarda toda a reprodução (fila + atual) terminar."""
        return self._player.wait_for_playback(timeout)

    def stop_playback(self) -> None:
        """Para reprodução atual e limpa fila."""
        self._player.stop()

    def get_status(self) -> dict:
        """Retorna status do gerenciador de voz."""
        primary_info = self._check_provider(self._config.primary_provider)
        fallback_info = self._check_provider(self._config.fallback_provider)
        
        return {
            "enabled": self._config.enabled,
            "primary_provider": self._config.primary_provider,
            "primary_available": primary_info.available,
            "primary_voice": primary_info.voice_name,
            "primary_reason": primary_info.reason if not primary_info.available else "",
            "fallback_provider": self._config.fallback_provider,
            "fallback_available": fallback_info.available,
            "fallback_voice": fallback_info.voice_name,
            "fallback_reason": fallback_info.reason if not fallback_info.available else "",
            "last_error": self._last_error,
            "fallback_used_last": self._fallback_attempted,
        }

    def shutdown(self) -> None:
        """Libera recursos."""
        for provider in self._providers.values():
            try:
                provider.shutdown()
            except Exception:
                pass
        self._providers.clear()
        self._player.shutdown()
        self._initialized = False