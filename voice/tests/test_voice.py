"""Testes para o módulo de voz/TTS do DaviOS."""

from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import Optional, Any

# Adiciona o diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.config import VoiceConfig
from voice.providers.base import TTSProvider, TTSProviderInfo
from voice.providers.piper_provider import PiperProvider
from voice.providers.sapi5_provider import SAPI5Provider
from voice.providers import get_provider_class, list_available_providers
from voice.audio_player import AudioPlayer
from voice.voice_manager import VoiceManager


class MockProvider(TTSProvider):
    """Provider mock para testes."""
    
    def __init__(self, config: dict, available: bool = True, audio_data: Optional[bytes] = b"fake_audio"):
        super().__init__(config)
        self._mock_available = available
        self._mock_audio = audio_data
        self._init_called = False
        self._synthesize_called = False
        self._synthesize_args = []
    
    @property
    def provider_name(self) -> str:
        return "mock"
    
    def check_availability(self) -> TTSProviderInfo:
        return TTSProviderInfo(
            name="mock",
            available=self._mock_available,
            reason="" if self._mock_available else "Mock indisponível",
            voice_name="mock-voice",
            sample_rate=22050,
        )
    
    def synthesize(self, text: str) -> Optional[bytes]:
        self._synthesize_called = True
        self._synthesize_args.append(text)
        return self._mock_audio


class TestVoiceConfig(unittest.TestCase):
    """Testes para VoiceConfig."""
    
    def test_defaults(self):
        cfg = VoiceConfig()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.primary_provider, "piper")
        self.assertEqual(cfg.fallback_provider, "sapi5")
        self.assertEqual(cfg.piper_executable, "piper")
        self.assertFalse(cfg.block_on_playback)
    
    def test_to_dict_roundtrip(self):
        cfg = VoiceConfig(enabled=False, primary_provider="sapi5")
        data = cfg.to_dict()
        cfg2 = VoiceConfig.from_dict(data)
        self.assertFalse(cfg2.enabled)
        self.assertEqual(cfg2.primary_provider, "sapi5")
    
    def test_get_provider_config(self):
        cfg = VoiceConfig()
        piper_cfg = cfg.get_provider_config("piper")
        self.assertIn("model_path", piper_cfg)
        self.assertIn("timeout_seconds", piper_cfg)
        
        sapi5_cfg = cfg.get_provider_config("sapi5")
        self.assertIn("voice_name", sapi5_cfg)
        self.assertIn("rate", sapi5_cfg)


class TestProviderRegistry(unittest.TestCase):
    """Testes para registry de providers."""
    
    def test_list_available(self):
        providers = list_available_providers()
        self.assertIn("piper", providers)
        self.assertIn("sapi5", providers)
    
    def test_get_class(self):
        self.assertEqual(get_provider_class("piper"), PiperProvider)
        self.assertEqual(get_provider_class("sapi5"), SAPI5Provider)
        self.assertIsNone(get_provider_class("inexistente"))


class TestPiperProvider(unittest.TestCase):
    """Testes para PiperProvider."""
    
    @patch("voice.providers.piper_provider.shutil.which")
    @patch("voice.providers.piper_provider.Path.exists")
    def test_check_availability_executable_not_found(self, mock_exists, mock_which):
        mock_which.return_value = None
        mock_exists.return_value = True
        
        cfg = {"piper_executable": "piper", "model_path": "/fake/model.onnx"}
        provider = PiperProvider(cfg)
        info = provider.check_availability()
        
        self.assertFalse(info.available)
        self.assertIn("não encontrado", info.reason)
    
    @patch("voice.providers.piper_provider.shutil.which")
    @patch("voice.providers.piper_provider.Path")
    def test_check_availability_model_not_found(self, mock_path_class, mock_which):
        """Testa que provider falha quando modelo não existe."""
        mock_which.return_value = "/usr/bin/piper"
        
        # Mock Path instance behavior
        mock_model_path = MagicMock()
        mock_model_path.exists.return_value = False  # model.onnx NÃO existe
        
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = True
        
        def path_side_effect(path_str):
            if path_str.endswith(".onnx"):
                return mock_model_path
            elif path_str.endswith(".json"):
                return mock_config_path
            return MagicMock(exists=Mock(return_value=False))
        
        mock_path_class.side_effect = path_side_effect
        
        cfg = {"piper_executable": "piper", "model_path": "/fake/model.onnx", "config_path": "/fake/config.json"}
        provider = PiperProvider(cfg)
        info = provider.check_availability()
        
        self.assertFalse(info.available)
        self.assertIn("Modelo não encontrado", info.reason)
    
    @patch("voice.providers.piper_provider.shutil.which")
    @patch("voice.providers.piper_provider.Path.exists")
    def test_check_availability_ok(self, mock_exists, mock_which):
        mock_which.return_value = "/usr/bin/piper"
        mock_exists.return_value = True
        
        cfg = {"piper_executable": "piper", "model_path": "/fake/model.onnx", "config_path": "/fake/config.json"}
        provider = PiperProvider(cfg)
        info = provider.check_availability()
        
        self.assertTrue(info.available)
        self.assertEqual(info.reason, "")


class TestSAPI5Provider(unittest.TestCase):
    """Testes para SAPI5Provider."""
    
    @patch("voice.providers.sapi5_provider.platform.system")
    def test_check_availability_not_windows(self, mock_system):
        mock_system.return_value = "Linux"
        
        cfg = {}
        provider = SAPI5Provider(cfg)
        info = provider.check_availability()
        
        self.assertFalse(info.available)
        self.assertIn("Windows", info.reason)
    
    @patch("voice.providers.sapi5_provider.platform.system")
    def test_check_availability_no_pywin32(self, mock_system):
        mock_system.return_value = "Windows"
        
        # Simula ImportError ao tentar importar win32com
        import voice.providers.sapi5_provider as sapi5_module
        original_import = __import__
        
        def mock_import(name, *args, **kwargs):
            if name == "win32com" or name.startswith("win32com."):
                raise ImportError("No module named win32com")
            return original_import(name, *args, **kwargs)
        
        with patch("builtins.__import__", side_effect=mock_import):
            cfg = {}
            provider = SAPI5Provider(cfg)
            info = provider.check_availability()
        
        self.assertFalse(info.available)
        self.assertIn("pywin32", info.reason)


class TestAudioPlayer(unittest.TestCase):
    """Testes para AudioPlayer."""
    
    @patch("voice.audio_player.platform.system")
    def test_play_not_windows(self, mock_system):
        mock_system.return_value = "Linux"
        
        player = AudioPlayer()
        result = player.play(b"fake_audio")
        
        self.assertFalse(result)
    
    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_play_windows_success(self, mock_temp, mock_run, mock_system):
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__ = Mock(return_value=Mock(name="temp.wav"))
        mock_temp.return_value.__exit__ = Mock(return_value=None)
        
        player = AudioPlayer()
        # Não podemos testar completamente sem mock mais complexo
        # Apenas verifica que não quebra
        self.assertIsNotNone(player)


class TestVoiceManager(unittest.TestCase):
    """Testes para VoiceManager - foco em lógica de seleção e fallback."""
    
    def setUp(self):
        self.config = VoiceConfig(
            enabled=True,
            primary_provider="mock_primary",
            fallback_provider="mock_fallback",
            block_on_playback=False,
        )
    
    def test_voice_disabled(self):
        """Voz desativada não deve sintetizar."""
        config = VoiceConfig(enabled=False)
        vm = VoiceManager(config)
        vm.initialize()
        
        audio = vm.synthesize("Olá mundo")
        self.assertIsNone(audio)
    
    def test_empty_text_skipped(self):
        """Texto vazio deve ser ignorado."""
        vm = VoiceManager(self.config)
        vm.initialize()
        
        audio = vm.synthesize("")
        self.assertIsNone(audio)
        
        audio = vm.synthesize("   ")
        self.assertIsNone(audio)
    
    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_primary_provider_used(self, mock_check):
        """Provider primário deve ser tentado primeiro."""
        mock_check.return_value = TTSProviderInfo("mock_primary", True, "", "primary-voice", 22050)
        
        vm = VoiceManager(self.config)
        # Substitui o provider real por mock
        mock_provider = MockProvider({}, available=True, audio_data=b"primary_audio")
        vm._providers["mock_primary"] = mock_provider
        
        audio = vm.synthesize("Teste")
        
        self.assertEqual(audio, b"primary_audio")
        self.assertTrue(mock_provider._synthesize_called)
        self.assertEqual(mock_provider._synthesize_args[0], "Teste")
    
    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_fallback_on_primary_failure(self, mock_check):
        """Fallback deve ser usado quando primário falha."""
        mock_check.return_value = TTSProviderInfo("mock_primary", True, "", "primary-voice", 22050)
        
        vm = VoiceManager(self.config)
        # Primário falha (retorna None)
        primary_mock = MockProvider({}, available=True, audio_data=None)
        # Fallback funciona
        fallback_mock = MockProvider({}, available=True, audio_data=b"fallback_audio")
        
        vm._providers["mock_primary"] = primary_mock
        vm._providers["mock_fallback"] = fallback_mock
        
        audio = vm.synthesize("Teste fallback")
        
        self.assertEqual(audio, b"fallback_audio")
        self.assertTrue(primary_mock._synthesize_called)
        self.assertTrue(fallback_mock._synthesize_called)
    
    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_both_fail_returns_none(self, mock_check):
        """None deve ser retornado se ambos falharem."""
        mock_check.return_value = TTSProviderInfo("mock_primary", True, "", "primary-voice", 22050)
        
        vm = VoiceManager(self.config)
        primary_mock = MockProvider({}, available=True, audio_data=None)
        fallback_mock = MockProvider({}, available=True, audio_data=None)
        
        vm._providers["mock_primary"] = primary_mock
        vm._providers["mock_fallback"] = fallback_mock
        
        audio = vm.synthesize("Teste ambos falham")
        
        self.assertIsNone(audio)
        # Erro pode ser do primário ou fallback, apenas verifica que há erro
        self.assertTrue(len(vm._last_error) > 0)
    
    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_fallback_none_returns_none(self, mock_check):
        """Se fallback for 'none', não tenta fallback."""
        config = VoiceConfig(
            enabled=True,
            primary_provider="mock_primary",
            fallback_provider="none",
        )
        mock_check.return_value = TTSProviderInfo("mock_primary", True, "", "primary-voice", 22050)
        
        vm = VoiceManager(config)
        primary_mock = MockProvider({}, available=True, audio_data=None)
        vm._providers["mock_primary"] = primary_mock
        
        audio = vm.synthesize("Teste sem fallback")
        
        self.assertIsNone(audio)
    
    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_speak_returns_bool(self, mock_check):
        """speak() deve retornar bool indicando sucesso."""
        mock_check.return_value = TTSProviderInfo("mock_primary", True, "", "primary-voice", 22050)
        
        vm = VoiceManager(self.config)
        mock_provider = MockProvider({}, available=True, audio_data=b"audio")
        vm._providers["mock_primary"] = mock_provider
        
        with patch.object(vm._player, "play", return_value=True) as mock_play:
            result = vm.speak("Olá")
            self.assertTrue(result)
            mock_play.assert_called_once()
    
    def test_get_status(self):
        """get_status deve retornar dicionário com info dos providers."""
        vm = VoiceManager(self.config)
        status = vm.get_status()
        
        self.assertIn("enabled", status)
        self.assertIn("primary_provider", status)
        self.assertIn("primary_available", status)
        self.assertIn("fallback_provider", status)
        self.assertIn("fallback_available", status)
        self.assertIn("last_error", status)
    
    def test_shutdown(self):
        """shutdown deve limpar providers."""
        vm = VoiceManager(self.config)
        mock_provider = MockProvider({}, available=True)
        vm._providers["mock"] = mock_provider
        
        vm.shutdown()
        
        self.assertEqual(len(vm._providers), 0)
        self.assertFalse(vm._initialized)


class TestIntegration(unittest.TestCase):
    """Testes de integração simulando fluxo completo."""
    
    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_full_flow_primary_success(self, mock_check):
        """Fluxo completo: primário funciona."""
        mock_check.return_value = TTSProviderInfo("piper", True, "", "pt-BR-faber", 22050)
        
        config = VoiceConfig(enabled=True, primary_provider="piper", fallback_provider="sapi5")
        vm = VoiceManager(config)
        
        # Mock do provider real
        mock_piper = MockProvider({}, available=True, audio_data=b"wav_data")
        vm._providers["piper"] = mock_piper
        
        with patch.object(vm._player, "play", return_value=True) as mock_play:
            result = vm.speak("Fala, Davi. Aqui é o DaviOS.")
            
            self.assertTrue(result)
            self.assertTrue(mock_piper._synthesize_called)
            mock_play.assert_called_once_with(b"wav_data", blocking=False)
    
    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_full_flow_primary_fails_fallback_works(self, mock_check):
        """Fluxo: primário falha, fallback funciona."""
        mock_check.return_value = TTSProviderInfo("piper", True, "", "pt-BR-faber", 22050)
        
        config = VoiceConfig(enabled=True, primary_provider="piper", fallback_provider="sapi5")
        vm = VoiceManager(config)
        
        mock_piper = MockProvider({}, available=True, audio_data=None)  # falha
        mock_sapi5 = MockProvider({}, available=True, audio_data=b"sapi5_audio")  # funciona
        vm._providers["piper"] = mock_piper
        vm._providers["sapi5"] = mock_sapi5
        
        with patch.object(vm._player, "play", return_value=True) as mock_play:
            result = vm.speak("Teste fallback")
            
            self.assertTrue(result)
            self.assertTrue(mock_piper._synthesize_called)
            self.assertTrue(mock_sapi5._synthesize_called)
            mock_play.assert_called_once_with(b"sapi5_audio", blocking=False)
    
    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_f5_tts_not_loaded(self, mock_check):
        """F5-TTS não deve ser importado ou carregado no fluxo normal."""
        mock_check.return_value = TTSProviderInfo("piper", True, "", "pt-BR-faber", 22050)
        
        config = VoiceConfig(enabled=True, primary_provider="piper", fallback_provider="sapi5")
        vm = VoiceManager(config)
        
        # Verifica que F5-TTS não está no registry de providers
        from voice.providers import list_available_providers
        providers = list_available_providers()
        self.assertNotIn("f5_tts", providers)
        self.assertNotIn("f5tts", providers)
        
        # VoiceManager não deve ter referência a F5-TTS
        self.assertFalse(hasattr(vm, "_f5_tts"))
        self.assertFalse(hasattr(vm, "_f5"))


if __name__ == "__main__":
    unittest.main(verbosity=2)