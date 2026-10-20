"""Testes do XTTSProvider e do fallback do VoiceManager.

Valida a integração XTTS-v2 SEM tocar o .venv principal nem instalar
torch/TTS nele.  Caminhos que exigem torch+coqui (carregamento real do
modelo) são skipados quando a dependência não está presente neste venv.
"""
from __future__ import annotations

import io
import os
import struct
import sys
import tempfile
import types
import wave
from pathlib import Path
from typing import Optional  # noqa: F401  (compat)
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voice.providers.base import TTSProviderInfo
from voice.providers.xtts_provider import XTTSProvider
from voice.config import VoiceConfig

try:
    import importlib.util as _ilu
    _HAS_TORCH = _ilu.find_spec("torch") is not None
    _HAS_TTS = _ilu.find_spec("TTS") is not None
except Exception:
    _HAS_TORCH = _HAS_TTS = False


def _cfg(**ov) -> dict:
    """Config equivalente a VoiceConfig.get_provider_config('xtts')."""
    d = dict(enabled=True, model_path="models/voice/xtts-v2",
             reference_audio="models/voice/references/reference.wav",
             language="pt", speaker_wav="", temperature=0.7,
             length_penalty=1.0, repetition_penalty=1.0, top_k=50, top_p=0.85,
             gpt_cond_len=64, gpt_cond_chunk_len=4, gpt_batch_size=1,
             low_mem=False, use_cuda=False, timeout_seconds=30.0)
    d.update(ov)
    return d


def _write_wav(path, sample_rate=24000, seconds=0.3):
    """WAV mono/16-bit (formato exigido por XTTS e pelo player)."""
    n = max(1, int(sample_rate * seconds))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


def _wav_bytes(sample_rate=22050, seconds=0.3) -> bytes:
    buf = io.BytesIO()
    n = max(1, int(sample_rate * seconds))
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
    return buf.getvalue()


class _FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False


class _FakeTTSModule:
    """Stub mínimo para `from TTS.api import TTS` sem coqui real."""
    class TTS:
        def __init__(self, *a, **k):
            pass

        def tts_to_file(self, **k):
            pass


@pytest.fixture
def fake_deps():
    """Injeta torch+TTS falsos em sys.modules (nada pesado é carregado)."""
    pkg = types.ModuleType("TTS"); api = types.ModuleType("TTS.api")
    api.TTS = _FakeTTSModule.TTS; pkg.api = api
    mods = {"torch": _FakeTorch(), "TTS": pkg, "TTS.api": api}
    with mock.patch.dict("sys.modules", mods, clear=False):
        yield


def _has_pesos() -> bool:
    return (ROOT / "models/voice/xtts-v2/config.json").exists() and \
           (ROOT / "models/voice/xtts-v2/model.pth").exists()


def _has_reference() -> bool:
    return (ROOT / "models/voice/references/reference.wav").exists()


# ---------- identidade / registry ----------

def test_provider_name():
    assert XTTSProvider(_cfg()).provider_name == "xtts"


def test_registry_has_xtts():
    from voice.providers import list_available_providers
    assert "xtts" in list_available_providers()


# ---------- check_availability ----------

def test_availability_no_deps_real_venv():
    """No .venv principal (sem torch/TTS): available=False por ImportError."""
    if _HAS_TTS:
        pytest.skip("torch+TTS presentes neste venv")
    p = XTTSProvider(_cfg())
    info = p.check_availability()
    assert isinstance(info, TTSProviderInfo)
    assert info.available is False
    assert "Dependência" in info.reason


def test_availability_model_missing(fake_deps):
    d = Path(tempfile.mkdtemp()) / "nao_existe"
    p = XTTSProvider(_cfg(model_path=str(d)))
    info = p.check_availability()
    assert info.available is False
    assert "modelo" in info.reason.lower()


def test_availability_reference_missing(fake_deps):
    if not _has_pesos():
        pytest.skip("pesos XTTS não presentes")
    p = XTTSProvider(_cfg(reference_audio="models/voice/references/inexistente.wav"))
    info = p.check_availability()
    assert info.available is False
    assert "refer" in info.reason.lower()


def test_availability_full_ready(fake_deps):
    """Com deps falsos + pesos + reference.wav: available=True."""
    if not _has_pesos():
        pytest.skip("pesos XTTS não presentes")
    if not _has_reference():
        pytest.skip("reference.wav ausente")
    p = XTTSProvider(_cfg())
    info = p.check_availability()
    assert info.available is True
    assert info.name == "xtts"
    assert info.voice_name == "xtts-v2-clone"
    assert info.sample_rate == 22050  # sample rate nativo do checkpoint XTTS-v2 (config.json: audio.sample_rate=22050)


class _FakeHighLevelTTS:
    """Mock do TTS high-level: tts_to_file grava WAV 24kHz válido."""
    def __init__(self):
        self.calls = []

    def tts_to_file(self, **kwargs):
        self.calls.append(kwargs)
        _write_wav(kwargs.get("file_path"), sample_rate=24000, seconds=0.3)
        return kwargs.get("file_path")


def _armed_provider(**cfg_ov):
    """Provider já 'inicializado' e com modelo carregado (para testar synthesize)."""
    p = XTTSProvider(_cfg(**cfg_ov))
    p._initialized = True
    p._model_loaded = True
    p._voice_name = "xtts-v2-clone"
    p._sample_rate = 24000
    return p


def test_synthesize_empty_text():
    p = XTTSProvider(_cfg())
    assert p.synthesize("") is None
    assert p.synthesize("   ") is None


def test_synthesize_unavailable_disabled_venv():
    """.venv principal: XTTS desativado -> synthesize NÃO crasha."""
    if _HAS_TTS:
        pytest.skip("deps presentes")
    p = XTTSProvider(_cfg())
    assert p.synthesize("olá mundo") is None


def test_synthesize_mock_success_returns_wav():
    fake = _FakeHighLevelTTS()
    p = _armed_provider()
    p._tts = fake
    with mock.patch.object(XTTSProvider, "_load_model", return_value=True):
        data = p.synthesize("Frase curta de teste.")
    assert isinstance(data, (bytes, bytearray)) and len(data) > 0
    with wave.open(io.BytesIO(bytes(data))) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2          # 16-bit
        assert w.getframerate() == 24000      # XTTS sample rate
    c = fake.calls[-1]
    assert c.get("text") == "Frase curta de teste."
    assert c.get("language") == "pt"
    assert c.get("speaker_wav")
    assert "speed" in c
    assert "length_penalty" in c        # kwarg valido (transformers GPT)
    assert "gpt_batch_size" not in c    # removido: rejeitado pelo transformers


def test_synthesize_exception_returns_none():
    """Falha interna no tts_to_file -> None, sem propagar exceção."""
    p = _armed_provider()
    bad = mock.MagicMock()
    bad.tts_to_file.side_effect = RuntimeError("boom")
    p._tts = bad
    with mock.patch.object(XTTSProvider, "_load_model", return_value=True):
        assert p.synthesize("x") is None


def test_synthesize_cleans_temp_wav():
    fake = _FakeHighLevelTTS()
    p = _armed_provider()
    p._tts = fake
    before = set(os.listdir(tempfile.gettempdir()))
    with mock.patch.object(XTTSProvider, "_load_model", return_value=True):
        out = p.synthesize("limpa o temporario")
    after = set(os.listdir(tempfile.gettempdir()))
    assert out
    assert not [f for f in (after - before) if f.endswith(".wav")]


def test_load_model_no_deps_returns_false():
    """_load_model sem torch: ImportError contido -> False (não crasha)."""
    if _HAS_TTS:
        pytest.skip("deps presentes")
    p = XTTSProvider(_cfg())
    assert p._load_model() is False
    assert p._tts is None and p._model_loaded is False


def test_load_model_uses_high_level_tts_and_caches(fake_deps):
    """Garante que _load_model usa TTS.api.TTS (high-level) e carrega 1x (cache)."""
    import sys as _sys
    if not _has_pesos():
        pytest.skip("pesos XTTS não presentes")
    p = XTTSProvider(_cfg())
    counter = {"n": 0}
    api = _sys.modules["TTS.api"]
    base = api.TTS

    class CountingTTS(base):
        def __init__(self, *a, **k):
            counter["n"] += 1
            super().__init__(*a, **k)

    api.TTS = CountingTTS
    try:
        ok1 = p._load_model()
        ok2 = p._load_model()
    finally:
        api.TTS = base
    assert ok1 is True and ok2 is True
    assert counter["n"] == 1          # guarda: 1 carregamento, 2 chamadas
    assert p._model_loaded is True
    p.shutdown()


def test_shutdown_clears_state():
    p = XTTSProvider(_cfg())
    p._initialized = True
    p._model_loaded = True
    p._tts = object()
    p._config_obj = object()
    p.shutdown()
    assert p._initialized is False and p._model_loaded is False
    assert p._tts is None and p._config_obj is None


# ---------- integração VoiceManager (fallback) ----------

def test_voice_manager_fallback_primary_fail_to_sapi5():
    """Primary (xtts) falha na síntese -> VoiceManager cai no fallback (sapi5)."""
    from voice.voice_manager import VoiceManager
    fallback_wav = _wav_bytes(sample_rate=22050)

    class PrimFail:
        def __init__(self, cfg):
            self._initialized = False

        @property
        def provider_name(self):
            return "xtts"

        def check_availability(self):
            return TTSProviderInfo("xtts", True, voice_name="xtts-v2-clone", sample_rate=24000)

        def initialize(self):
            self._initialized = True
            return True

        def is_initialized(self):
            return self._initialized

        def synthesize(self, text):
            return None  # primário falha -> dispara fallback

        def shutdown(self):
            pass

    class FallbackOk:
        def __init__(self, cfg):
            self._initialized = False

        @property
        def provider_name(self):
            return "sapi5"

        def check_availability(self):
            return TTSProviderInfo("sapi5", True, voice_name="sapi5", sample_rate=22050)

        def initialize(self):
            self._initialized = True
            return True

        def is_initialized(self):
            return self._initialized

        def synthesize(self, text):
            return fallback_wav

        def shutdown(self):
            pass

    mapping = {"xtts": PrimFail, "sapi5": FallbackOk}
    with mock.patch("voice.voice_manager.get_provider_class",
                    side_effect=lambda name: mapping.get(name)):
        cfg = VoiceConfig(enabled=True, primary_provider="xtts",
                          fallback_provider="sapi5", xtts_enabled=True)
        vm = VoiceManager(cfg)
        vm._player = mock.MagicMock()  # evita reprodução real
        assert vm.initialize() is True
        audio = vm.synthesize("olá, tudo bem?")

    assert audio == fallback_wav
    assert vm._fallback_attempted is True
    assert vm._last_error  # primário registrou falha


def test_voice_manager_all_providers_fail_returns_none():
    """Se primário E fallback retornam None -> None (texto preservado no caller)."""
    from voice.voice_manager import VoiceManager

    class AllFail:
        def __init__(self, cfg):
            self._initialized = False

        @property
        def provider_name(self):
            return "f"

        def check_availability(self):
            return TTSProviderInfo("f", True, voice_name="f", sample_rate=22050)

        def initialize(self):
            self._initialized = True
            return True

        def is_initialized(self):
            return self._initialized

        def synthesize(self, text):
            return None

        def shutdown(self):
            pass

    with mock.patch("voice.voice_manager.get_provider_class",
                    side_effect=lambda name: AllFail):
        cfg = VoiceConfig(enabled=True, primary_provider="xtts",
                          fallback_provider="sapi5", xtts_enabled=True)
        vm = VoiceManager(cfg)
        vm._player = mock.MagicMock()
        vm.initialize()
        assert vm.synthesize("erro") is None
        assert vm._fallback_attempted is True
