# voice/tests/test_voice_xtts_fallback.py
"""Fallback do VoiceManager: primario falha -> fallback produz audio.

Teste autocontido (provedores mockados) - sem precisar de torch/TTS/SAPI5 reais.
"""
from __future__ import annotations

from unittest import mock

from voice.config import VoiceConfig
from voice.providers.base import TTSProviderInfo
from voice.voice_manager import VoiceManager


_WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 100


class _FailPrimary:
    provider_name = "xtts_fail"

    def __init__(self, cfg):
        self._cfg = cfg
        self._initialized = False

    def check_availability(self):
        return TTSProviderInfo("xtts_fail", True, "", "ok", 24000)

    def initialize(self):
        self._initialized = True
        return True

    def is_initialized(self):
        return self._initialized

    def synthesize(self, text):
        return None  # primario falha na sintese

    def shutdown(self):
        pass

    def get_voice_name(self):
        return "fail"

    def get_sample_rate(self):
        return 24000


class _OkFallback:
    provider_name = "sapi5_ok"

    def __init__(self, cfg):
        self._cfg = cfg
        self._initialized = False

    def check_availability(self):
        return TTSProviderInfo("sapi5_ok", True, "", "ok", 22050)

    def initialize(self):
        self._initialized = True
        return True

    def is_initialized(self):
        return self._initialized

    def synthesize(self, text):
        return _WAV  # fallback produz audio

    def shutdown(self):
        pass

    def get_voice_name(self):
        return "ok"

    def get_sample_rate(self):
        return 22050


def _cfg(primary="xtts", fallback="sapi5"):
    return VoiceConfig(
        enabled=True,
        primary_provider=primary,
        fallback_provider=fallback,
        xtts_enabled=True,
        xtts_model_path="models/voice/xtts-v2",
        xtts_reference_audio="models/voice/references/reference.wav",
    )


def _factory(name):
    return {"xtts": _FailPrimary, "sapi5": _OkFallback}.get(name)


def test_fallback_used_when_primary_synth_returns_none():
    from voice.voice_manager import get_provider_class  # noqa: F401  (confirma import)
    cfg = _cfg(primary="xtts", fallback="sapi5")
    with mock.patch("voice.voice_manager.get_provider_class", side_effect=_factory):
        vm = VoiceManager(cfg)
        vm.initialize()
        vm._player = mock.MagicMock()  # evita tocar audio
        audio = vm.synthesize("frase de teste do fallback")

    assert audio == _WAV
    assert vm._fallback_attempted is True


def test_no_fallback_when_fallback_disabled():
    """primary falha e fallback='none' -> synthesize retorna None."""
    cfg = _cfg(primary="xtts", fallback="none")
    with mock.patch("voice.voice_manager.get_provider_class", side_effect=_factory):
        vm = VoiceManager(cfg)
        vm.initialize()
        vm._player = mock.MagicMock()
        audio = vm.synthesize("sem fallback")
    assert audio is None
    # _fallback_attempted soa False porque fallback 'none' nao e tentado
    assert vm._fallback_attempted is False
