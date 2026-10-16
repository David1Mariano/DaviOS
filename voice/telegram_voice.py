"""Envio ocasional de mensagens de voz pelo Telegram (adaptador de voz).

Este módulo concentra a decisão de quando enviar áudio, a síntese via
``VoiceManager``, a conversão de WAV (saída dos providers) para OGG Opus
(formato esperado pelo Telegram ``sendVoice``) e o envio em si.

Princípios de design:
  - O áudio NUNCA atrasa nem substitui a resposta textual (fire-and-forget).
  - Qualquer falha (síntese, conversão, envio) é registrada sem expor segredos
    e o texto continua sendo entregue normalmente.
  - Nenhuma nova dependência pesada: reusa ``VoiceManager`` existente e o
    ``ffmpeg`` já presente no ambiente para conversão de formato.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

from voice.config import VoiceConfig
from voice.voice_manager import VoiceManager

logger = logging.getLogger("davios.voice.telegram")

#: Palavras/frases (case-insensitive) que tornam uma resposta inelegível para áudio.
_EXCLUSION_KEYWORDS = (
    "traceback",
    "traceback (most recent call last)",
    "error:",
    "exception",
    "falha:",
    "erro:",
)

#: Limites padrão de tamanho (sobrescritos pela config quando presentes).
DEFAULT_MIN_CHARS = 3
DEFAULT_MAX_CHARS = 300
DEFAULT_PROBABILITY = 0.35


def _get(config: VoiceConfig, name: str, default) -> object:
    """Lê um atributo da config com fallback para o valor default."""
    return getattr(config, name, default)


def is_voice_eligible(text: str, config: VoiceConfig) -> bool:
    """Decide (sem aleatoriedade) se um texto é elegível para mensagem de voz.

    Critérios contextuais simples e configuráveis:
      - texto não vazio;
      - tamanho entre ``telegram_voice_min_chars`` e ``telegram_voice_max_chars``;
      - não contém blocos de código (`` ``` ``);
      - não contém indícios de traceback / erro;
      - não é majoritariamente linhas indentadas (código colado).

    Não depende de IA nem de nova dependência pesada.
    """
    if not text or not text.strip():
        return False

    t = text.strip()

    min_chars = int(_get(config, "telegram_voice_min_chars", DEFAULT_MIN_CHARS))
    max_chars = int(_get(config, "telegram_voice_max_chars", DEFAULT_MAX_CHARS))
    if len(t) < min_chars or len(t) > max_chars:
        return False

    if "```" in t:
        return False

    low = t.lower()
    if any(kw in low for kw in _EXCLUSION_KEYWORDS):
        return False

    lines = t.splitlines()
    if lines:
        indented = sum(1 for ln in lines if ln.startswith("    "))
        if indented / len(lines) > 0.5:
            return False

    return True


def should_send_voice(
    text: str,
    config: VoiceConfig,
    rng: Optional[Callable[[], float]] = None,
) -> bool:
    """Decide se o áudio deve ser enviado para uma resposta.

    Combina:
      1. voz do Telegram habilitada na config;
      2. elegibilidade contextual do texto (``is_voice_eligible``);
      3. probabilidade configurada (default 35%).

    ``rng`` permite injetar a fonte aleatória nos testes (sem testes
    probabilísticos frágeis). Por padrão usa ``random.random``.
    """
    if not _get(config, "telegram_voice_enabled", False):
        return False

    if not is_voice_eligible(text, config):
        return False

    probability = float(_get(config, "telegram_voice_probability", DEFAULT_PROBABILITY))
    roll = (rng or random.random)()
    return 0.0 <= roll < probability


def _ffmpeg_exe(override: Optional[str] = None) -> Optional[str]:
    """Localiza o executável do ffmpeg (override > PATH)."""
    if override:
        return override
    return shutil.which("ffmpeg")


def convert_wav_to_ogg(
    wav_path: str,
    ogg_path: str,
    ffmpeg_exe: Optional[str] = None,
    timeout_seconds: float = 30.0,
) -> bool:
    """Converte um arquivo WAV/PCM para OGG Opus com ffmpeg.

    Retorna True apenas se o arquivo de saída existir e não estiver vazio.
    Qualquer problema (ffmpeg ausente, timeout, erro de codificação) resulta
    em False — chamador trata como falha silenciosa.
    """
    exe = _ffmpeg_exe(ffmpeg_exe) or "ffmpeg"
    try:
        result = subprocess.run(
            [
                exe,
                "-y",  # sobrescreve saída se existir
                "-i",
                wav_path,
                "-c:a",
                "libopus",
                "-b:a",
                "64k",
                "-vbr",
                "on",
                ogg_path,
            ],
            capture_output=True,
            timeout=timeout_seconds,
        )
        if result.returncode != 0:
            logger.debug(
                "[voice.telegram] ffmpeg rc=%d stderr=%s",
                result.returncode,
                result.stderr.decode("utf-8", errors="replace")[:200],
            )
            return False
        return os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 0
    except (OSError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.debug("[voice.telegram] ffmpeg indisponível ou falhou: %s", type(exc).__name__)
        return False


def _cleanup(paths: list[str]) -> None:
    """Remove arquivos temporários (best effort; nunca levanta)."""
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass


def synthesize_and_send_voice(
    text: str,
    client,
    chat_id: int,
    voice_manager: VoiceManager,
    config: VoiceConfig,
    ffmpeg_exe: Optional[str] = None,
) -> bool:
    """Fluxo completo: síntese WAV → conversão OGG → envio ``sendVoice``.

    Retorna True se enviado com sucesso. Falhas são logadas sem segredos e o
    texto continua sendo entregue pelo caller.

    Arquivos temporários (WAV e OGG) são removidos em ``finally``, mesmo em
    caso de falha — nada fica armazenado indefinidamente.
    """
    audio: Optional[bytes] = None
    try:
        audio = voice_manager.synthesize(text)
        if not audio:
            logger.debug("[voice.telegram] síntese retornou vazio/None")
            return False

        # WAV temporário
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(audio)
            wav_path = tmp_wav.name
        ogg_path = wav_path + ".ogg"

        try:
            converted = convert_wav_to_ogg(wav_path, ogg_path, ffmpeg_exe=ffmpeg_exe)
            if not converted:
                logger.warning(
                    "[voice.telegram] conversão WAV→OGG falhou (ffmpeg ausente ou erro)"
                )
                return False

            result = client.send_voice(chat_id, ogg_path)
            return result is not None
        finally:
            _cleanup([wav_path, ogg_path])
    except Exception as exc:  # noqa: BLE001 - áudio nunca derruba o fluxo de texto
        logger.warning(
            "[voice.telegram] falha no envio de voz error_type=%s error_code=%s",
            type(exc).__name__,
            getattr(exc, "error_code", None),
        )
        return False
