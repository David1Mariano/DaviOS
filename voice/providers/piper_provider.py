"""Provider TTS usando Piper (offline, local)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

from voice.providers.base import TTSProvider, TTSProviderInfo


logger = logging.getLogger("davios.voice.providers.piper")


class PiperProvider(TTSProvider):
    """Provider para Piper TTS via executável CLI."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._executable = config.get("piper_executable", "piper")
        self._model_path = config.get("model_path", "")
        self._config_path = config.get("config_path", "")
        self._data_dir = config.get("data_dir", "")
        self._timeout = config.get("timeout_seconds", 10.0)
        self._speaker_id = config.get("speaker_id", 0)
        self._length_scale = config.get("length_scale", 1.0)
        self._noise_scale = config.get("noise_scale", 0.667)
        self._noise_w = config.get("noise_w", 0.8)

    @property
    def provider_name(self) -> str:
        return "piper"

    def check_availability(self) -> TTSProviderInfo:
        """Verifica se o Piper está instalado e o modelo existe."""
        # Verifica executável
        exe_path = shutil.which(self._executable)
        if not exe_path:
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason=f"Executável '{self._executable}' não encontrado no PATH",
            )

        # Verifica modelo
        if not self._model_path or not Path(self._model_path).exists():
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason=f"Modelo não encontrado: {self._model_path}",
            )

        # Verifica config (opcional mas recomendado)
        config_path = self._config_path or f"{self._model_path}.json"
        if not Path(config_path).exists():
            logger.warning("[%s] Config do modelo não encontrado: %s", 
                          self.provider_name, config_path)

        # Tenta obter info da voz via --help ou metadata
        voice_name = self._extract_voice_name()

        return TTSProviderInfo(
            name=self.provider_name,
            available=True,
            reason="",
            voice_name=voice_name,
            sample_rate=self._get_sample_rate_from_config(config_path),
        )

    def _extract_voice_name(self) -> str:
        """Tenta extrair nome da voz do arquivo de config."""
        try:
            import json
            config_path = self._config_path or f"{self._model_path}.json"
            if Path(config_path).exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                # Estrutura típica do Piper: config.inference.noise_scale etc.
                # Nome pode estar em config.model ou similar
                if "model" in cfg and "name" in cfg["model"]:
                    return cfg["model"]["name"]
                if "speaker_name" in cfg:
                    return cfg["speaker_name"]
        except Exception:
            pass
        return "piper-voice"

    def _get_sample_rate_from_config(self, config_path: str) -> int:
        """Extrai sample rate do config do modelo."""
        try:
            import json
            if Path(config_path).exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                # Piper usa sample_rate no nível superior ou em audio
                return cfg.get("sample_rate", cfg.get("audio", {}).get("sample_rate", 22050))
        except Exception:
            pass
        return 22050

    def synthesize(self, text: str) -> Optional[bytes]:
        """Sintetiza texto usando Piper CLI."""
        if not text or not text.strip():
            logger.debug("[%s] Texto vazio, pulando síntese", self.provider_name)
            return None

        if not self.initialize():
            return None

        # Cria arquivo temporário para saída
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            output_path = tmp_out.name

        try:
            # Constrói comando Piper
            cmd = [
                self._executable,
                "--model", self._model_path,
                "--output_file", output_path,
                "--speaker_id", str(self._speaker_id),
                "--length_scale", str(self._length_scale),
                "--noise_scale", str(self._noise_scale),
                "--noise_w", str(self._noise_w),
            ]

            if self._config_path:
                cmd.extend(["--config", self._config_path])

            if self._data_dir:
                cmd.extend(["--data_dir", self._data_dir])

            logger.debug("[%s] Executando: %s", self.provider_name, " ".join(cmd[:5]) + " ...")

            # Executa com timeout
            result = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=self._timeout,
            )

            if result.returncode != 0:
                logger.error("[%s] Falha na síntese (rc=%d): %s", 
                            self.provider_name, result.returncode, 
                            result.stderr.decode("utf-8", errors="replace")[:200])
                return None

            # Lê o arquivo WAV gerado
            with open(output_path, "rb") as f:
                audio_data = f.read()

            if not audio_data:
                logger.error("[%s] Arquivo de saída vazio", self.provider_name)
                return None

            logger.debug("[%s] Síntese OK: %d bytes", self.provider_name, len(audio_data))
            return audio_data

        except subprocess.TimeoutExpired:
            logger.error("[%s] Timeout na síntese (%.1fs)", self.provider_name, self._timeout)
            return None
        except Exception as e:
            logger.exception("[%s] Erro na síntese", self.provider_name)
            return None
        finally:
            # Limpa arquivo temporário
            try:
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception:
                pass