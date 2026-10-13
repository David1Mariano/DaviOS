"""Provider TTS usando XTTS-v2 (clonagem de voz)."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from voice.providers.base import TTSProvider, TTSProviderInfo


logger = logging.getLogger("davios.voice.providers.xtts")


class XTTSProvider(TTSProvider):
    """Provider para XTTS-v2 via Coqui TTS.
    
    Carregamento preguiçoso: o modelo pesado só é carregado na primeira síntese.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._model_path = config.get("model_path", "models/voice/xtts-v2")
        self._reference_audio = config.get("reference_audio", "models/voice/references/reference.wav")
        self._language = config.get("language", "pt")
        self._speaker_wav = config.get("speaker_wav", "")
        self._temperature = config.get("temperature", 0.7)
        self._length_penalty = config.get("length_penalty", 1.0)
        self._repetition_penalty = config.get("repetition_penalty", 1.0)
        self._top_k = config.get("top_k", 50)
        self._top_p = config.get("top_p", 0.85)
        self._gpt_cond_len = config.get("gpt_cond_len", 64)
        self._gpt_cond_chunk_len = config.get("gpt_cond_chunk_len", 4)
        self._gpt_batch_size = config.get("gpt_batch_size", 1)
        self._low_mem = config.get("low_mem", False)
        self._use_cuda = config.get("use_cuda", False)
        self._timeout = config.get("timeout_seconds", 30.0)
        
        self._tts = None
        self._model_loaded = False

    @property
    def provider_name(self) -> str:
        return "xtts"

    def check_availability(self) -> TTSProviderInfo:
        """Verifica se o XTTS está disponível (dependências + modelo + áudio de referência)."""
        # Verifica dependências sem carregar o modelo
        try:
            import torch
            from TTS.api import TTS
        except ImportError as e:
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason=f"Dependência não instalada: {e}",
            )

        # Verifica se modelo existe (pasta com config.json e model.pth)
        model_dir = Path(self._model_path)
        if not model_dir.exists():
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason=f"Diretório do modelo não encontrado: {self._model_path}",
            )

        config_file = model_dir / "config.json"
        if not config_file.exists():
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason=f"config.json não encontrado em: {model_dir}",
            )

        model_file = model_dir / "model.pth"
        if not model_file.exists():
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason=f"model.pth não encontrado em: {model_dir}",
            )

        # Verifica áudio de referência
        ref_audio = self._speaker_wav or self._reference_audio
        if not ref_audio or not Path(ref_audio).exists():
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason=f"Áudio de referência não encontrado: {ref_audio}",
            )

        # Verifica CUDA se solicitado
        if self._use_cuda and not torch.cuda.is_available():
            logger.warning("[%s] CUDA solicitado mas não disponível, usando CPU", self.provider_name)

        return TTSProviderInfo(
            name=self.provider_name,
            available=True,
            reason="",
            voice_name="xtts-v2-clone",
                                    sample_rate=22050,  # sample rate nativo do checkpoint XTTS-v2 (config.json: audio.sample_rate=22050)
        )

    def _load_model(self) -> bool:
        """Carrega o modelo XTTS-v2 (lazy loading) via API high-level TTS.api.

        Usa TTS(model_path=...) (alto nivel) de modo a expor tts_to_file
        (classe high-level TTS.api.TTS). Carregamento unico e reutilizado
        (cache via _model_loaded); falhas retornam False (nao abortam boot).
        """
        if self._model_loaded and self._tts is not None:
            return True

        try:
            import torch
            from TTS.api import TTS

            logger.info("[%s] Carregando modelo XTTS-v2 de: %s", self.provider_name, self._model_path)

            use_cuda = bool(self._use_cuda and torch.cuda.is_available())
            if self._use_cuda and not use_cuda:
                logger.warning("[%s] CUDA solicitado mas nao disponivel, usando CPU",
                               self.provider_name)

                        # XTTS-v2 e end-to-end (vocoder embutido): carrega via model_path dir.
            # Compat: torch>=2.6 default weights_only=True rejeita checkpoints
            # coqui (contem XttsConfig/XttsAudioConfig pickled). O env var
            # TORCH_FORCE_WEIGHTS_ONLY_LOAD e IGNORADO no torch 2.14.x, e o
            # allowlist de globals (add_safe_globals) e fraco (varias classes
            # pickled). A unica forma confiavel: forcar weights_only=False em
            # TODAS as chamadas torch.load durante o load do modelo (checkpoint
            # local confiavel). Usamos atribuicao FORTE (k['weights_only']=False)
            # e nao setdefault, pois a chamada interna de coqui nao passa o arg e
            # o default do torch e True.
            _orig_torch_load = getattr(torch, "load", None)
            if _orig_torch_load is not None:
                def _wraps(*a, **k):
                    k["weights_only"] = False
                    return _orig_torch_load(*a, **k)
                torch.load = _wraps
                try:
                    self._tts = TTS(
                        model_path=str(self._model_path),
                        config_path=str(Path(self._model_path) / "config.json"),
                        gpu=use_cuda,
                        progress_bar=False,
                    )
                finally:
                    torch.load = _orig_torch_load
            else:
                self._tts = TTS(
                    model_path=str(self._model_path),
                    config_path=str(Path(self._model_path) / "config.json"),
                    gpu=use_cuda,
                    progress_bar=False,
                )

            # low_mem: limita threads em CPU para reduzir pico de RAM/CPU
            if self._low_mem and not use_cuda:
                try:
                    torch.set_num_threads(min(torch.get_num_threads(), 4))
                except Exception:
                    pass

            # Compat: torchaudio>=2.6 load() exige torchcodec+FFmpeg; podem
            # ausentar. Usa soundfile (ja dependencia) como fallback de leitura.
            try:
                import torchaudio as _ta
                if not getattr(_ta, "_davioss_ta_patched", False):
                    _orig_ta_load = _ta.load

                    def _ta_load_compat(audiopath, *a, **k):
                        try:
                            return _orig_ta_load(audiopath, *a, **k)
                        except Exception:
                            import soundfile as _sf
                            import numpy as _np
                            data, sr = _sf.read(str(audiopath), dtype="float32")
                            if data.ndim == 2:
                                data = data.mean(1)
                            return (torch.from_numpy(_np.ascontiguousarray(data)).unsqueeze(0), sr)
                    _ta.load = _ta_load_compat
                    _ta._davioss_ta_patched = True
            except Exception:
                pass

            self._config_obj = None  # high-level TTS gerencia internamente config/vocoder
            self._model_loaded = True
            logger.info("[%s] Modelo carregado (high-level) em %s",
                        self.provider_name, "cuda" if use_cuda else "cpu")
            return True

        except Exception:
            logger.exception("[%s] Falha ao carregar modelo XTTS-v2", self.provider_name)
            self._tts = None
            self._config_obj = None
            self._model_loaded = False
            return False

    def synthesize(self, text: str) -> Optional[bytes]:
        """Sintetiza texto usando XTTS-v2 com clonagem de voz."""
        if not text or not text.strip():
            logger.debug("[%s] Texto vazio, pulando síntese", self.provider_name)
            return None

        if not self.initialize():
            return None

        if not self._load_model():
            return None

        # Determina arquivo de referência
        ref_audio = self._speaker_wav or self._reference_audio
        if not ref_audio or not Path(ref_audio).exists():
            logger.error("[%s] Áudio de referência não encontrado: %s", self.provider_name, ref_audio)
            return None

        try:
            logger.debug("[%s] Sintetizando: %s...", self.provider_name, text[:50])

            # Cria arquivo temporário para saída
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                output_path = tmp_out.name

            # Sintetiza usando o modelo diretamente
            # XTTS usa tts_to_file no modelo
            self._tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=ref_audio,
                language=self._language,
                speed=1.0,
                temperature=self._temperature,
                length_penalty=self._length_penalty,
                repetition_penalty=self._repetition_penalty,
                top_k=self._top_k,
                top_p=self._top_p,
                gpt_cond_len=self._gpt_cond_len,
                gpt_cond_chunk_len=self._gpt_cond_chunk_len,
            )

            # Lê o arquivo WAV gerado
            with open(output_path, "rb") as f:
                audio_data = f.read()

            # Limpa arquivo temporário
            try:
                os.unlink(output_path)
            except Exception:
                pass

            if not audio_data:
                logger.error("[%s] Arquivo de saída vazio", self.provider_name)
                return None

            logger.debug("[%s] Síntese OK: %d bytes", self.provider_name, len(audio_data))
            return audio_data

        except Exception as e:
            logger.exception("[%s] Erro na síntese", self.provider_name)
            return None

    def shutdown(self) -> None:
        """Libera recursos do modelo."""
        if self._tts is not None:
            try:
                # Libera memória do modelo
                import torch
                del self._tts
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception:
                pass
            self._tts = None
        self._config_obj = None
        self._model_loaded = False
        self._initialized = False
        logger.info("[%s] Recursos liberados", self.provider_name)