"""Reprodução de áudio simples para Windows."""

from __future__ import annotations

import logging
import platform
import subprocess
import tempfile
import os
import threading
import time
from typing import Optional


logger = logging.getLogger("davios.voice.audio")


class AudioPlayer:
    """Player de áudio simples usando player nativo do Windows."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._timeout = self._config.get("playback_timeout_seconds", 30.0)
        self._current_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def play(self, audio_data: bytes, blocking: bool = False) -> bool:
        """Reproduz áudio WAV/PCM.
        
        Args:
            audio_data: Bytes do arquivo WAV.
            blocking: Se True, aguarda término da reprodução.
            
        Returns:
            True se iniciado com sucesso, False caso contrário.
        """
        if not audio_data:
            logger.warning("[AudioPlayer] Dados de áudio vazios")
            return False

        if platform.system() != "Windows":
            logger.warning("[AudioPlayer] Reprodução só suportada no Windows")
            return False

        # Salva em arquivo temporário
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                temp_path = tmp.name
        except Exception as e:
            logger.exception("[AudioPlayer] Falha ao criar arquivo temporário")
            return False

        def _play_and_cleanup():
            try:
                # Usa PowerShell com Media.SoundPlayer (nativo, sem dependências)
                ps_cmd = f'''
                $player = New-Object System.Media.SoundPlayer
                $player.SoundLocation = "{temp_path}"
                $player.PlaySync()
                '''
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True,
                    timeout=self._timeout,
                )
                if result.returncode != 0:
                    logger.warning("[AudioPlayer] PowerShell falhou (rc=%d): %s",
                                  result.returncode, result.stderr.decode("utf-8", errors="replace")[:200])
            except subprocess.TimeoutExpired:
                logger.warning("[AudioPlayer] Timeout na reprodução (%.1fs)", self._timeout)
            except Exception as e:
                logger.exception("[AudioPlayer] Erro na reprodução")
            finally:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception:
                    pass

        if blocking:
            _play_and_cleanup()
            return True
        else:
            # Fire-and-forget com thread
            with self._lock:
                # Se há reprodução anterior ainda rodando, aguarda (evita sobreposição)
                if self._current_thread and self._current_thread.is_alive():
                    logger.debug("[AudioPlayer] Aguardando reprodução anterior terminar")
                    self._current_thread.join(timeout=1.0)
                
                self._current_thread = threading.Thread(target=_play_and_cleanup, daemon=True)
                self._current_thread.start()
                return True

    def stop(self) -> None:
        """Para reprodução atual (best effort)."""
        with self._lock:
            if self._current_thread and self._current_thread.is_alive():
                # SoundPlayer não tem stop fácil; apenas aguarda
                self._current_thread.join(timeout=0.5)
                self._current_thread = None

    def wait_current(self, timeout: Optional[float] = None) -> bool:
        """Aguarda reprodução atual terminar."""
        with self._lock:
            if self._current_thread and self._current_thread.is_alive():
                self._current_thread.join(timeout=timeout or self._timeout)
                return not self._current_thread.is_alive()
            return True