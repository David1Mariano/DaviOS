"""Testes do envio ocasional de voz pelo Telegram (voice/telegram_voice.py).

Sem rede real, sem TTS real: fakes injetados. Verifica:
  - elegibilidade contextual (critérios simples, sem IA);
  - probabilidade de 35% com fonte aleatória injetável (sem testes frágeis);
  - falha segura: texto não depende de áudio.
"""

from __future__ import annotations

import io
import os
import struct
import unittest
import wave
from unittest.mock import patch

from voice.config import VoiceConfig
from voice import telegram_voice as tv
from brain.channels.telegram_client import TelegramAPIError


def _voice_config(**overrides):
    base = dict(
        enabled=True,
        telegram_voice_enabled=True,
        telegram_voice_probability=0.35,
        telegram_voice_min_chars=3,
        telegram_voice_max_chars=300,
    )
    base.update(overrides)
    return VoiceConfig(**base)


class TestIsVoiceEligible(unittest.TestCase):
    """Critérios contextuais puros (sem aleatoriedade)."""

    def test_saudacao_curta_elegivel(self):
        cfg = _voice_config()
        self.assertTrue(tv.is_voice_eligible("Oi, tudo bem?", cfg))

    def test_confirmacao_curta_elegivel(self):
        cfg = _voice_config()
        self.assertTrue(tv.is_voice_eligible("Sim, posso ajudar com isso.", cfg))

    def test_texto_vazio_nao_elegivel(self):
        cfg = _voice_config()
        self.assertFalse(tv.is_voice_eligible("", cfg))
        self.assertFalse(tv.is_voice_eligible("   ", cfg))

    def test_texto_muito_curto_nao_elegivel(self):
        cfg = _voice_config(telegram_voice_min_chars=10)
        self.assertFalse(tv.is_voice_eligible("ok", cfg))

    def test_texto_muito_longo_nao_elegivel(self):
        cfg = _voice_config(telegram_voice_max_chars=20)
        self.assertFalse(tv.is_voice_eligible("Esta resposta excede o limite.", cfg))

    def test_limite_maximo_inclusivo_elegivel(self):
        cfg = _voice_config(telegram_voice_max_chars=3)
        self.assertTrue(tv.is_voice_eligible("Oi!", cfg))  # exatamente 3 <= 3

    def test_bloco_de_codigo_nao_elegivel(self):
        cfg = _voice_config()
        self.assertFalse(tv.is_voice_eligible("Aqui está:\n```\nprint(1)\n```", cfg))

    def test_traceback_nao_elegivel(self):
        cfg = _voice_config()
        self.assertFalse(
            tv.is_voice_eligible("Traceback (most recent call last):\n  File x", cfg)
        )

    def test_palavra_erro_nao_elegivel(self):
        cfg = _voice_config()
        self.assertFalse(tv.is_voice_eligible("Erro: algo deu errado", cfg))

    def test_linhas_indentadas_majoritarias_nao_elegiveis(self):
        cfg = _voice_config()
        texto = "\n".join("    codigo()" for _ in range(10)) + "\nola"
        self.assertFalse(tv.is_voice_eligible(texto, cfg))


class TestShouldSendVoice(unittest.TestCase):
    """Elegibilidade + probabilidade configurada (35% default)."""

    def test_probability_default_is_35_percent(self):
        cfg = VoiceConfig()  # defaults da classe
        self.assertEqual(cfg.telegram_voice_probability, 0.35)
        self.assertFalse(cfg.telegram_voice_enabled)  # opt-in por padrão

    def test_voz_desativada_nunca_envia(self):
        cfg = _voice_config(telegram_voice_enabled=False)
        # rng baixo (sempre dispararia se habilitado) — deve continuar False.
        self.assertFalse(tv.should_send_voice("Oi", cfg, rng=lambda: 0.01))

    def test_texto_inelegivel_nunca_selecionado(self):
        cfg = _voice_config(telegram_voice_probability=1.0)
        # rng = 0.01 (sempre abaixo do limite): ainda assim NAO envia texto inelegível.
        self.assertFalse(
            tv.should_send_voice("```\ncodigo aqui\n```", cfg, rng=lambda: 0.01)
        )
        self.assertFalse(tv.should_send_voice("", cfg, rng=lambda: 0.01))
        self.assertFalse(
            tv.should_send_voice("Erro: falha ao conectar", cfg, rng=lambda: 0.01)
        )

    def test_abaixo_do_limite_envia(self):
        cfg = _voice_config(telegram_voice_probability=0.35)
        self.assertTrue(tv.should_send_voice("Oi, tudo bem?", cfg, rng=lambda: 0.10))

    def test_acima_do_limite_nao_envia(self):
        cfg = _voice_config(telegram_voice_probability=0.35)
        self.assertFalse(tv.should_send_voice("Oi, tudo bem?", cfg, rng=lambda: 0.90))

    def test_limite_exato_nao_envia(self):
        cfg = _voice_config(telegram_voice_probability=0.35)
        # 0.0 <= roll < probability: roll == 0.35 nao envia.
        self.assertFalse(tv.should_send_voice("Oi!", cfg, rng=lambda: 0.35))
        self.assertTrue(tv.should_send_voice("Oi!", cfg, rng=lambda: 0.349999))

    def test_rng_default_aleatorio(self):
        cfg = _voice_config(telegram_voice_probability=1.0)
        # sem rng injetado: deve usar random.random e aceitar (prob 1.0).
        self.assertTrue(tv.should_send_voice("Oi, tudo bem?", cfg))

    def test_probabilidade_personalizada(self):
        cfg = _voice_config(telegram_voice_probability=0.0)
        self.assertFalse(tv.should_send_voice("Oi!", cfg, rng=lambda: 0.0))
        cfg = _voice_config(telegram_voice_probability=1.0)
        self.assertTrue(tv.should_send_voice("Oi!", cfg, rng=lambda: 0.999))


class TestSendVoiceMessageSafe(unittest.TestCase):
    """Falha segura: síntese/conversão/envio nunca vazam para o texto."""

    def _make_client(self, sent_voice_paths=None, raise_on_send=None):
        sent_voice_paths = sent_voice_paths or []

        class FakeClient:
            def __init__(self):
                self.sent_voice = sent_voice_paths

            def send_voice(self, chat_id, voice_path):
                self.sent_voice.append((chat_id, voice_path))
                if raise_on_send is not None:
                    raise raise_on_send
                return {"ok": True}

        return FakeClient()

    def _make_vm(self, audio_data=b"wav-audio-bytes"):
        class FakeVM:
            def synthesize(self, text):
                return audio_data

        return FakeVM()

    def _make_wav(self, sample_rate=22050, duration_s=0.1):
        """Gera um arquivo WAV/PCM valido (silencio) em bytes."""
        buf = io.BytesIO()
        n_frames = int(sample_rate * duration_s)
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            for _ in range(n_frames):
                w.writeframes(struct.pack("<h", 0))
        return buf.getvalue()

    def test_envio_sucesso_grava_ogg_e_remove_temporarios(self):
        client = self._make_client()
        cfg = _voice_config()
        wav_bytes = self._make_wav()
        result = tv.synthesize_and_send_voice(
            "Oi, tudo bem?", client, 555, self._make_vm(audio_data=wav_bytes), cfg
        )
        self.assertTrue(result)
        self.assertEqual(len(client.sent_voice), 1)
        chat_id, path = client.sent_voice[0]
        self.assertEqual(chat_id, 555)
        self.assertTrue(path.endswith(".ogg"))
        # o arquivo OGG gerado pelo ffmpeg foi removido apos o envio
        self.assertFalse(os.path.exists(path))

    def test_sintese_vazia_nao_envia(self):
        client = self._make_client()
        cfg = _voice_config()
        result = tv.synthesize_and_send_voice(
            "Oi", client, 555, self._make_vm(audio_data=None), cfg
        )
        self.assertFalse(result)
        self.assertEqual(client.sent_voice, [])

    def test_envio_quebra_nao_propaga(self):
        client = self._make_client(raise_on_send=Exception("rede caiu"))
        cfg = _voice_config()
        result = tv.synthesize_and_send_voice("Oi", client, 555, self._make_vm(), cfg)
        self.assertFalse(result)

    def test_log_erro_inclui_error_code_sem_vazar_token(self):
        client = self._make_client(
            raise_on_send=TelegramAPIError(
                "Bad Request: token 123456:ABCDEF vazou na mensagem",
                error_code=400,
            )
        )
        cfg = _voice_config()
        wav_bytes = self._make_wav()
        with patch.object(tv.logger, "warning") as mock_warning:
            result = tv.synthesize_and_send_voice(
                "Oi", client, 555, self._make_vm(audio_data=wav_bytes), cfg
            )
        self.assertFalse(result)
        args, _ = mock_warning.call_args
        mensagem = args[0] % args[1:] if len(args) > 1 else args[0]
        self.assertIn("error_type=TelegramAPIError", mensagem)
        self.assertIn("error_code=400", mensagem)
        self.assertNotIn("123456:ABCDEF", mensagem)

    def test_sem_ffmpeg_retorna_falso_sem_quebrar(self):
        client = self._make_client()
        cfg = _voice_config()
        with patch.object(tv, "_ffmpeg_exe", return_value=None):
            # "ffmpeg" nao encontrado -> FileNotFoundError capturado -> False
            result = tv.synthesize_and_send_voice(
                "Oi", client, 555, self._make_vm(), cfg
            )
        self.assertFalse(result)
        self.assertEqual(client.sent_voice, [])

    def test_conversao_falha_retorna_falso_sem_quebrar(self):
        client = self._make_client()
        cfg = _voice_config()
        with patch("voice.telegram_voice.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = b"erro de codificacao"
            result = tv.synthesize_and_send_voice(
                "Oi", client, 555, self._make_vm(), cfg
            )
        self.assertFalse(result)
        self.assertEqual(client.sent_voice, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
