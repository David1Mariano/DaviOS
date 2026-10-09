"""Configuração do sistema de voz/TTS do DaviOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class VoiceConfig:
    """Configuração de voz integrada ao DaviosConfig."""
    
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "primary_provider": self.primary_provider,
            "fallback_provider": self.fallback_provider,
            "piper_executable": self.piper_executable,
            "piper_model_path": self.piper_model_path,
            "piper_config_path": self.piper_config_path,
            "piper_data_dir": self.piper_data_dir,
            "piper_speaker_id": self.piper_speaker_id,
            "piper_length_scale": self.piper_length_scale,
            "piper_noise_scale": self.piper_noise_scale,
            "piper_noise_w": self.piper_noise_w,
            "sapi5_voice_name": self.sapi5_voice_name,
            "sapi5_rate": self.sapi5_rate,
            "sapi5_volume": self.sapi5_volume,
            "synthesis_timeout_seconds": self.synthesis_timeout_seconds,
            "playback_timeout_seconds": self.playback_timeout_seconds,
            "block_on_playback": self.block_on_playback,
            "skip_empty_text": self.skip_empty_text,
            "log_synthesis_errors": self.log_synthesis_errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def get_provider_config(self, provider_name: str) -> dict[str, Any]:
        """Retorna config específica para um provider."""
        if provider_name == "piper":
            return {
                "piper_executable": self.piper_executable,
                "model_path": self.piper_model_path,
                "config_path": self.piper_config_path,
                "data_dir": self.piper_data_dir,
                "speaker_id": self.piper_speaker_id,
                "length_scale": self.piper_length_scale,
                "noise_scale": self.piper_noise_scale,
                "noise_w": self.piper_noise_w,
                "timeout_seconds": self.synthesis_timeout_seconds,
            }
        elif provider_name == "sapi5":
            return {
                "voice_name": self.sapi5_voice_name,
                "rate": self.sapi5_rate,
                "volume": self.sapi5_volume,
                "timeout_seconds": self.synthesis_timeout_seconds,
            }
        return {}