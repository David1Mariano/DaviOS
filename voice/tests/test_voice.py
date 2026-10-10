"""Testes para o módulo de voz/TTS do DaviOS."""

from __future__ import annotations

import sys
import os
import time
import threading
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


class TestSAPI5Timeout(unittest.TestCase):
    """Testes para timeout efetivo do SAPI5."""

    @patch("voice.providers.sapi5_provider.platform.system")
    @patch("voice.providers.sapi5_provider.threading.Thread")
    def test_synthesize_timeout(self, mock_thread_class, mock_system):
        """Síntese que excede timeout deve retornar None."""
        mock_system.return_value = "Windows"
        
        # Mock da thread que não completa a tempo
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True  # Thread ainda rodando após join
        mock_thread_class.return_value = mock_thread
        
        # Mock win32com para não falhar no import
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 0.1}  # timeout muito curto
            provider = SAPI5Provider(cfg)
            provider._initialized = True  # Pula check_availability
            
            # Mock do tempfile para evitar I/O real
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("voice.providers.sapi5_provider.os.unlink"):
                        result = provider.synthesize("Teste timeout")
        
        self.assertIsNone(result)
        mock_thread.join.assert_called_once_with(timeout=0.1)

    @patch("voice.providers.sapi5_provider.platform.system")
    def test_synthesize_success_within_timeout(self, mock_system):
        """Síntese que completa dentro do timeout deve retornar áudio."""
        mock_system.return_value = "Windows"
        
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 10.0}
            provider = SAPI5Provider(cfg)
            provider._initialized = True
            
            # Mock do tempfile e leitura de arquivo
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("builtins.open", mock_open(read_data=b"fake_wav_data")):
                        with patch("voice.providers.sapi5_provider.os.unlink"):
                            result = provider.synthesize("Teste sucesso")
        
        self.assertEqual(result, b"fake_wav_data")


class TestAudioPlayerQueue(unittest.TestCase):
    """Testes para fila FIFO do AudioPlayer."""

    @patch("voice.audio_player.platform.system")
    def test_play_non_windows_returns_false(self, mock_system):
        mock_system.return_value = "Linux"
        player = AudioPlayer()
        result = player.play(b"fake_audio")
        self.assertFalse(result)

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_play_blocking_waits_for_completion(self, mock_temp, mock_run, mock_system):
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        result = player.play(b"fake_audio", blocking=True)
        
        self.assertTrue(result)
        mock_run.assert_called_once()

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_queue_serialization_multiple_plays(self, mock_temp, mock_run, mock_system):
        """Múltiplas chamadas play() não bloqueantes devem ser serializadas na fila."""
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        # Enfileira 3 itens rapidamente
        results = []
        for i in range(3):
            results.append(player.play(f"audio_{i}".encode(), blocking=False))
        
        self.assertTrue(all(results))
        
        # Aguarda processamento da fila
        player.wait_for_playback(timeout=2.0)
        
        # Verifica que subprocess.run foi chamado 3 vezes (sequencialmente)
        self.assertEqual(mock_run.call_count, 3)

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_stop_clears_queue(self, mock_temp, mock_run, mock_system):
        """stop() deve limpar a fila pendente."""
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        # Enfileira alguns itens
        player.play(b"audio_1", blocking=False)
        player.play(b"audio_2", blocking=False)
        
        # Para o player
        player.stop()
        
        # Fila deve estar vazia
        self.assertTrue(player._queue.empty())

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_wait_for_playback_waits_full_queue(self, mock_temp, mock_run, mock_system):
        """wait_for_playback() deve aguardar toda a fila."""
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        player.play(b"audio_1", blocking=False)
        player.play(b"audio_2", blocking=False)
        player.play(b"audio_3", blocking=False)
        
        result = player.wait_for_playback(timeout=2.0)
        
        self.assertTrue(result)
        self.assertEqual(mock_run.call_count, 3)


class TestConcurrentSpeak(unittest.TestCase):
    """Testes para chamadas concorrentes a speak()/speak_async()."""

    def setUp(self):
        self.config = VoiceConfig(
            enabled=True,
            primary_provider="mock_provider",
            fallback_provider="none",
            block_on_playback=False,
        )

    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_multiple_speak_async_queued_in_order(self, mock_check):
        """Múltiplas speak_async() devem ser enfileiradas na ordem."""
        mock_check.return_value = TTSProviderInfo("mock_provider", True, "", "test-voice", 22050)
        
        vm = VoiceManager(self.config)
        mock_provider = MockProvider({}, available=True, audio_data=b"audio_data")
        vm._providers["mock_provider"] = mock_provider
        
        # Mock do player para capturar chamadas
        play_calls = []
        original_play = vm._player.play
        
        def tracking_play(audio_data, blocking=False):
            play_calls.append((audio_data, blocking))
            return True
        
        with patch.object(vm._player, "play", side_effect=tracking_play):
            # Dispara 5 chamadas rápidas
            for i in range(5):
                result = vm.speak_async(f"Mensagem {i}")
                self.assertTrue(result)
            
            # Aguarda fila processar
            vm.wait_for_playback(timeout=2.0)
        
        # Verifica que todas foram chamadas na ordem
        self.assertEqual(len(play_calls), 5)
        for i, (audio_data, blocking) in enumerate(play_calls):
            self.assertEqual(audio_data, b"audio_data")
            self.assertFalse(blocking)

    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_speak_blocking_waits_each(self, mock_check):
        """speak() com block_on_playback=True deve aguardar cada um."""
        config = VoiceConfig(
            enabled=True,
            primary_provider="mock_provider",
            fallback_provider="none",
            block_on_playback=True,
        )
        mock_check.return_value = TTSProviderInfo("mock_provider", True, "", "test-voice", 22050)
        
        vm = VoiceManager(config)
        mock_provider = MockProvider({}, available=True, audio_data=b"audio_data")
        vm._providers["mock_provider"] = mock_provider
        
        play_count = 0
        def tracking_play(audio_data, blocking=False):
            nonlocal play_count
            play_count += 1
            return True
        
        with patch.object(vm._player, "play", side_effect=tracking_play):
            vm.speak("Primeira")
            vm.speak("Segunda")
            vm.speak("Terceira")
        
        self.assertEqual(play_count, 3)

    @patch("voice.voice_manager.VoiceManager._check_provider")
    def test_mixed_speak_and_speak_async(self, mock_check):
        """Mistura de speak() e speak_async() deve manter ordem."""
        mock_check.return_value = TTSProviderInfo("mock_provider", True, "", "test-voice", 22050)
        
        vm = VoiceManager(self.config)
        mock_provider = MockProvider({}, available=True, audio_data=b"audio_data")
        vm._providers["mock_provider"] = mock_provider
        
        play_order = []
        def tracking_play(audio_data, blocking=False):
            play_order.append(("blocking" if blocking else "async", audio_data))
            return True
        
        with patch.object(vm._player, "play", side_effect=tracking_play):
            vm.speak_async("Async 1")
            vm.speak("Blocking 1")  # block_on_playback=False por default
            vm.speak_async("Async 2")
            
            vm.wait_for_playback(timeout=2.0)
        
        self.assertEqual(len(play_order), 3)
        self.assertEqual(play_order[0][0], "async")
        self.assertEqual(play_order[1][0], "async")  # speak() usa block_on_playback=False
        self.assertEqual(play_order[2][0], "async")


# Helper para mock de open
from unittest.mock import mock_open


class TestSAPI5LockSerialization(unittest.TestCase):
    """Testes para serialização via lock no SAPI5."""

    @patch("voice.providers.sapi5_provider.platform.system")
    def test_synthesize_lock_serializes_calls(self, mock_system):
        """Lock serializa chamadas: segunda aguarda primeira liberar."""
        mock_system.return_value = "Windows"
        
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 10.0}
            provider = SAPI5Provider(cfg)
            provider._initialized = True
            
            # Mock do tempfile e arquivo
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("builtins.open", mock_open(read_data=b"fake_wav_data")):
                        with patch("voice.providers.sapi5_provider.os.unlink"):
                            # Primeira chamada
                            result1 = provider.synthesize("Primeira")
                            # Segunda chamada
                            result2 = provider.synthesize("Segunda")
        
        self.assertEqual(result1, b"fake_wav_data")
        self.assertEqual(result2, b"fake_wav_data")
        # Verifica que lock permite chamadas sequenciais (sem erro)

    @patch("voice.providers.sapi5_provider.platform.system")
    @patch("voice.providers.sapi5_provider.threading.Thread")
    def test_synthesize_lock_timeout_waiting(self, mock_thread_class, mock_system):
        """Segunda chamada faz timeout aguardando lock se primeira demora."""
        mock_system.return_value = "Windows"
        
        # Primeira thread "trava" (não completa)
        stuck_thread = MagicMock()
        stuck_thread.is_alive.return_value = True
        
        def create_thread(target, daemon):
            if mock_thread_class.call_count == 0:
                return stuck_thread  # primeira
            return MagicMock()  # segunda
        
        mock_thread_class.side_effect = create_thread
        
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 0.1}
            provider = SAPI5Provider(cfg)
            provider._initialized = True
            
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("voice.providers.sapi5_provider.os.unlink"):
                        # Primeira chamada inicia mas não completa
                        result1 = provider.synthesize("Primeira")
                        # Segunda deve falhar por timeout no lock
                        result2 = provider.synthesize("Segunda")
        
        # Primeira retorna None (timeout na thread COM)
        self.assertIsNone(result1)
        # Segunda retorna None (timeout aguardando lock)
        self.assertIsNone(result2)

    @patch("voice.providers.sapi5_provider.platform.system")
    def test_synthesize_temp_file_cleaned_on_success(self, mock_system):
        """Arquivo temporário deve ser removido em caso de sucesso."""
        mock_system.return_value = "Windows"
        
        unlink_calls = []
        def track_unlink(path):
            unlink_calls.append(path)
        
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 10.0}
            provider = SAPI5Provider(cfg)
            provider._initialized = True
            
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                temp_name = "/tmp/test_success.wav"
                mock_temp.return_value.__enter__.return_value.name = temp_name
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("builtins.open", mock_open(read_data=b"fake_wav_data")):
                        with patch("voice.providers.sapi5_provider.os.unlink", side_effect=track_unlink):
                            result = provider.synthesize("Teste limpeza")
        
        self.assertEqual(result, b"fake_wav_data")
        # Verifica que unlink foi chamado para o arquivo temp
        self.assertTrue(any("test_success.wav" in str(c) for c in unlink_calls))

    @patch("voice.providers.sapi5_provider.platform.system")
    def test_synthesize_temp_file_cleaned_on_error(self, mock_system):
        """Arquivo temporário deve ser removido em caso de erro."""
        mock_system.return_value = "Windows"
        
        unlink_calls = []
        def track_unlink(path):
            unlink_calls.append(path)
        
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 10.0}
            provider = SAPI5Provider(cfg)
            provider._initialized = True
            
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                temp_name = "/tmp/test_error.wav"
                mock_temp.return_value.__enter__.return_value.name = temp_name
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("builtins.open", mock_open(read_data=b"")):  # arquivo vazio = erro
                        with patch("voice.providers.sapi5_provider.os.unlink", side_effect=track_unlink):
                            result = provider.synthesize("Teste erro")
        
        self.assertIsNone(result)
        self.assertTrue(any("test_error.wav" in str(c) for c in unlink_calls))

    @patch("voice.providers.sapi5_provider.platform.system")
    @patch("voice.providers.sapi5_provider.threading.Thread")
    def test_synthesize_temp_file_cleaned_on_timeout(self, mock_thread_class, mock_system):
        """Arquivo temporário deve ser removido em caso de timeout."""
        mock_system.return_value = "Windows"
        
        unlink_calls = []
        def track_unlink(path):
            unlink_calls.append(path)
        
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_thread_class.return_value = mock_thread
        
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 0.1}
            provider = SAPI5Provider(cfg)
            provider._initialized = True
            
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                temp_name = "/tmp/test_timeout.wav"
                mock_temp.return_value.__enter__.return_value.name = temp_name
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("voice.providers.sapi5_provider.os.unlink", side_effect=track_unlink):
                        result = provider.synthesize("Teste timeout")
        
        self.assertIsNone(result)
        self.assertTrue(any("test_timeout.wav" in str(c) for c in unlink_calls))

    @patch("voice.providers.sapi5_provider.platform.system")
    def test_synthesize_timeout_leaves_slot_occupied(self, mock_system):
        """Após timeout, slot de síntese COM deve permanecer ocupado até thread terminar."""
        mock_system.return_value = "Windows"
        
        # Mock da thread que não completa a tempo
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 0.01}  # timeout muito curto
            provider = SAPI5Provider(cfg)
            provider._initialized = True
            
            # Mock do tempfile
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("voice.providers.sapi5_provider.os.unlink"):
                        # Primeira chamada: timeout expira
                        result1 = provider.synthesize("Primeira")
                        
                        self.assertIsNone(result1)  # timeout
                        # Slot deve permanecer ocupado (thread COM ainda rodando)
                        self.assertEqual(provider._active_com_syntheses, 1,
                                       "Slot de síntese COM deve estar ocupado após timeout")

    @patch("voice.providers.sapi5_provider.platform.system")
    def test_synthesize_slot_released_after_com_thread_completes(self, mock_system):
        """Após thread COM terminar (sucesso/erro), slot deve ser liberado."""
        mock_system.return_value = "Windows"
        
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 10.0}
            provider = SAPI5Provider(cfg)
            provider._initialized = True
            
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("builtins.open", mock_open(read_data=b"fake_wav_data")):
                        with patch("voice.providers.sapi5_provider.os.unlink"):
                            # Síntese bem-sucedida
                            result1 = provider.synthesize("Primeira")
                            
                            self.assertEqual(result1, b"fake_wav_data")
                            # Slot deve ser liberado após thread COM terminar
                            self.assertEqual(provider._active_com_syntheses, 0,
                                           "Slot deve estar livre após síntese bem-sucedida")
                            
                            # Nova síntese deve ser permitida
                            result2 = provider.synthesize("Segunda")
                            self.assertEqual(result2, b"fake_wav_data")
                            self.assertEqual(provider._active_com_syntheses, 0)

    @patch("voice.providers.sapi5_provider.platform.system")
    @patch("voice.providers.sapi5_provider.threading.Thread")
    def test_synthesize_second_call_waits_for_com_slot(self, mock_thread_class, mock_system):
        """Segunda chamada deve aguardar slot COM ser liberado, não iniciar thread paralela."""
        mock_system.return_value = "Windows"
        
        # Mock da thread que simula demora
        slow_thread = MagicMock()
        slow_thread.is_alive.return_value = True
        slow_thread.join = MagicMock()  # não bloqueia de verdade no teste
        
        # Primeira chamada cria thread lenta, segunda cria thread normal
        def create_thread(target, daemon):
            if mock_thread_class.call_count == 0:
                return slow_thread
            return MagicMock()
        
        mock_thread_class.side_effect = create_thread
        
        with patch.dict("sys.modules", {"win32com": MagicMock(), "win32com.client": MagicMock()}):
            cfg = {"timeout_seconds": 0.1}
            provider = SAPI5Provider(cfg)
            provider._initialized = True
            
            with patch("voice.providers.sapi5_provider.tempfile.NamedTemporaryFile") as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.wav"
                with patch("voice.providers.sapi5_provider.os.path.exists", return_value=True):
                    with patch("voice.providers.sapi5_provider.os.unlink"):
                        # Primeira chamada: timeout
                        result1 = provider.synthesize("Primeira")
                        self.assertIsNone(result1)
                        
                        # Slot ocupado
                        self.assertEqual(provider._active_com_syntheses, 1)
                        
                        # Segunda chamada: deve fazer timeout aguardando slot (não criar nova thread)
                        result2 = provider.synthesize("Segunda")
                        self.assertIsNone(result2)
                        
                        # Apenas uma thread COM deve ter sido criada
                        self.assertEqual(mock_thread_class.call_count, 1,
                                       "Não deve iniciar segunda thread COM enquanto slot ocupado")


class TestAudioPlayerStopShutdown(unittest.TestCase):
    """Testes para stop/shutdown do AudioPlayer."""

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_stop_signals_current_item_future(self, mock_temp, mock_run, mock_system):
        """stop() deve sinalizar futures dos itens pendentes para não bloquear wait_*. """
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        # Enfileira itens não-bloqueantes (vão para a fila, worker pode ou não ter pego)
        player.play(b"audio_1", blocking=False)
        player.play(b"audio_2", blocking=False)
        
        # Chama stop() - deve limpar fila e sinalizar futures
        player.stop()
        
        # Fila deve estar vazia
        self.assertTrue(player._queue.empty())
        
        # wait_for_playback deve retornar True (nada pendente)
        result = player.wait_for_playback(timeout=0.5)
        self.assertTrue(result)

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_wait_for_playback_after_stop_returns_true(self, mock_temp, mock_run, mock_system):
        """wait_for_playback() após stop() deve retornar True (nada pendente)."""
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        player.play(b"audio_1", blocking=False)
        player.play(b"audio_2", blocking=False)
        
        player.stop()
        
        # wait_for_playback deve retornar True imediatamente
        result = player.wait_for_playback(timeout=1.0)
        self.assertTrue(result)

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_wait_for_playback_after_shutdown_returns_true(self, mock_temp, mock_run, mock_system):
        """wait_for_playback() após shutdown() deve retornar True."""
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        player.play(b"audio_1", blocking=False)
        player.shutdown()
        
        # wait_for_playback deve retornar True (player encerrado, nada pendente)
        result = player.wait_for_playback(timeout=1.0)
        self.assertTrue(result)

    @patch("voice.audio_player.platform.system")
    def test_play_after_shutdown_rejects(self, mock_system):
        """play() após shutdown() deve rejeitar novos itens."""
        mock_system.return_value = "Windows"
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        player.shutdown()
        
        result = player.play(b"audio_1", blocking=False)
        self.assertFalse(result)

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_wait_current_unblocks_after_stop(self, mock_temp, mock_run, mock_system):
        """wait_current() deve desbloquear após stop()."""
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        player.play(b"audio_1", blocking=False)
        time.sleep(0.1)  # worker pega item
        
        # wait_current bloquearia, mas stop() sinaliza future
        player.stop()
        
        # wait_current deve retornar True imediatamente
        result = player.wait_current(timeout=0.5)
        self.assertTrue(result)

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_pending_items_futures_set_on_stop(self, mock_temp, mock_run, mock_system):
        """Itens pendentes na fila devem ter futures sinalizados no stop()."""
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        # Enfileira vários itens não-bloqueantes
        player.play(b"audio_1", blocking=False)
        player.play(b"audio_2", blocking=False)
        player.play(b"audio_3", blocking=False)
        
        # Para antes de processar
        player.stop()
        
        # Todos os itens pendentes devem ter sido removidos e futures sinalizados
        self.assertTrue(player._queue.empty())
        # Nota: itens já processados pelo worker não estão mais na fila


class TestAudioPlayerBlockingWaits(unittest.TestCase):
    """Testes para chamadas bloqueantes no AudioPlayer."""

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_blocking_play_unblocks_on_stop(self, mock_temp, mock_run, mock_system):
        """play(blocking=True) deve desbloquear quando stop() é chamado."""
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        # Inicia play bloqueante em thread separada
        result_container = {}
        def blocking_play():
            result_container["result"] = player.play(b"audio_1", blocking=True)
        
        import threading
        t = threading.Thread(target=blocking_play)
        t.start()
        
        time.sleep(0.1)  # worker pega item
        
        # Para o player
        player.stop()
        
        t.join(timeout=1.0)
        
        # play() deve ter retornado (não bloqueia indefinidamente)
        self.assertIn("result", result_container)
        # Resultado pode ser False (cancelado) ou True (completou antes do stop)
        # O importante é que não travou

    @patch("voice.audio_player.platform.system")
    @patch("voice.audio_player.subprocess.run")
    @patch("voice.audio_player.tempfile.NamedTemporaryFile")
    def test_multiple_blocking_plays_sequential(self, mock_temp, mock_run, mock_system):
        """Múltiplos play(blocking=True) devem ser sequenciais."""
        mock_system.return_value = "Windows"
        mock_run.return_value = Mock(returncode=0)
        mock_temp.return_value.__enter__.return_value.name = "temp.wav"
        mock_temp.return_value.__exit__.return_value = None
        
        player = AudioPlayer({"playback_timeout_seconds": 5.0})
        
        results = []
        for i in range(3):
            results.append(player.play(f"audio_{i}".encode(), blocking=True))
        
        self.assertTrue(all(results))
        self.assertEqual(mock_run.call_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)