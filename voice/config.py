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
    
    # XTTS-v2 (voice cloning)
    xtts_enabled: bool = False
    xtts_model_path: str = "models/voice/xtts-v2"
    xtts_reference_audio: str = "models/voice/references/reference.wav"
    xtts_language: str = "pt"
    xtts_speaker_wav: str = ""
    xtts_temperature: float = 0.7
    xtts_length_penalty: float = 1.0
    xtts_repetition_penalty: float = 5.0  # valor default do checkpoint XTTS-v2 (config.json); 1.0 causa super-geracao
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
            "xtts_enabled": self.xtts_enabled,
            "xtts_model_path": self.xtts_model_path,
            "xtts_reference_audio": self.xtts_reference_audio,
            "xtts_language": self.xtts_language,
            "xtts_speaker_wav": self.xtts_speaker_wav,
            "xtts_temperature": self.xtts_temperature,
            "xtts_length_penalty": self.xtts_length_penalty,
            "xtts_repetition_penalty": self.xtts_repetition_penalty,
            "xtts_top_k": self.xtts_top_k,
            "xtts_top_p": self.xtts_top_p,
            "xtts_gpt_cond_len": self.xtts_gpt_cond_len,
            "xtts_gpt_cond_chunk_len": self.xtts_gpt_cond_chunk_len,
            "xtts_gpt_batch_size": self.xtts_gpt_batch_size,
            "xtts_low_mem": self.xtts_low_mem,
            "xtts_use_cuda": self.xtts_use_cuda,
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
        elif provider_name == "xtts":
            return {
                "enabled": self.xtts_enabled,
                "model_path": self.xtts_model_path,
                "reference_audio": self.xtts_reference_audio,
                "language": self.xtts_language,
                "speaker_wav": self.xtts_speaker_wav,
                "temperature": self.xtts_temperature,
                "length_penalty": self.xtts_length_penalty,
                "repetition_penalty": self.xtts_repetition_penalty,
                "top_k": self.xtts_top_k,
                "top_p": self.xtts_top_p,
                "gpt_cond_len": self.xtts_gpt_cond_len,
                "gpt_cond_chunk_len": self.xtts_gpt_cond_chunk_len,
                "gpt_batch_size": self.xtts_gpt_batch_size,
                "low_mem": self.xtts_low_mem,
                "use_cuda": self.xtts_use_cuda,
                "timeout_seconds": self.synthesis_timeout_seconds,
            }
        return {}
