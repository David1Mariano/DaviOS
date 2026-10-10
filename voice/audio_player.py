"""Reprodução de áudio simples para Windows."""

from __future__ import annotations

import logging
import platform
import subprocess
import tempfile
import os
import threading
import time
import queue
from typing import Optional
from dataclasses import dataclass


logger = logging.getLogger("davios.voice.audio")


@dataclass
class PlaybackItem:
    """Item na fila de reprodução."""
    audio_data: bytes
    blocking: bool
    future: "threading.Event"
    result_container: dict


class AudioPlayer:
    """Player de áudio simples usando player nativo do Windows com fila FIFO."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {}
        self._timeout = self._config.get("playback_timeout_seconds", 30.0)
        self._queue: "queue.Queue[PlaybackItem]" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_running = False
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._current_item: Optional[PlaybackItem] = None
        self._accepting_items = True

    def play(self, audio_data: bytes, blocking: bool = False) -> bool:
        """Reproduz áudio WAV/PCM (adiciona à fila FIFO).
        
        Args:
            audio_data: Bytes do arquivo WAV.
            blocking: Se True, aguarda término da reprodução.
            
        Returns:
            True se enfileirado com sucesso, False caso contrário.
        """
        if not audio_data:
            logger.warning("[AudioPlayer] Dados de áudio vazios")
            return False

        if platform.system() != "Windows":
            logger.warning("[AudioPlayer] Reprodução só suportada no Windows")
            return False

        if not self._accepting_items:
            logger.warning("[AudioPlayer] Player encerrado, não aceita novos itens")
            return False

        # Salva em arquivo temporário
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                temp_path = tmp.name
        except Exception as e:
            logger.exception("[AudioPlayer] Falha ao criar arquivo temporário")
            return False

        # Cria evento para sincronização se blocking
        done_event = threading.Event()
        result_container = {"success": False, "error": None}

        item = PlaybackItem(
            audio_data=audio_data,
            blocking=blocking,
            future=done_event,
            result_container=result_container,
        )
        # Armazena o path no item para uso posterior
        item.temp_path = temp_path  # type: ignore[attr-defined]

        # Inicia worker se não estiver rodando
        self._ensure_worker_started()

        # Enfileira
        self._queue.put(item)

        if blocking:
            # Aguarda conclusão
            done_event.wait(timeout=self._timeout * 2)  # timeout generoso para fila
            return result_container.get("success", False)
        else:
            # Fire-and-forget
            return True

    def _ensure_worker_started(self) -> None:
        """Inicia a thread worker se não estiver rodando."""
        with self._lock:
            if not self._worker_running:
                self._worker_running = True
                self._shutdown_event.clear()
                self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
                self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Loop principal do worker - processa fila FIFO sequencialmente."""
        while self._worker_running and not self._shutdown_event.is_set():
            try:
                # Aguarda item com timeout para verificar shutdown
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._current_item = item
            try:
                self._play_item(item)
            finally:
                self._queue.task_done()
                self._current_item = None

    def _play_item(self, item: PlaybackItem) -> None:
        """Reproduz um item da fila."""
        temp_path = getattr(item, "temp_path", None)
        if not temp_path:
            logger.error("[AudioPlayer] Item sem caminho temporário")
            item.result_container["success"] = False
            item.result_container["error"] = "Sem caminho temporário"
            item.future.set()
            return

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
                item.result_container["success"] = False
                item.result_container["error"] = f"PowerShell rc={result.returncode}"
            else:
                item.result_container["success"] = True
        except subprocess.TimeoutExpired:
            logger.warning("[AudioPlayer] Timeout na reprodução (%.1fs)", self._timeout)
            item.result_container["success"] = False
            item.result_container["error"] = "Timeout"
        except Exception as e:
            logger.exception("[AudioPlayer] Erro na reprodução")
            item.result_container["success"] = False
            item.result_container["error"] = str(e)
        finally:
            # Limpa arquivo temporário
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
            # Sinaliza conclusão se blocking
            if item.blocking:
                item.future.set()

    def stop(self) -> None:
        """Para reprodução atual e limpa fila (best effort).
        
        Sinaliza todos os eventos de itens pendentes e do item atual
        para não deixar chamadas bloqueantes pendentes.
        """
        with self._lock:
            # Limpa fila pendente
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    # Limpa temp file do item se existir
                    temp_path = getattr(item, "temp_path", None)
                    if temp_path and os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except Exception:
                            pass
                    item.result_container["success"] = False
                    item.result_container["error"] = "Cancelado pelo stop()"
                    item.future.set()
                    self._queue.task_done()
                except queue.Empty:
                    break

            # Sinaliza item atual para acordar wait_current/wait_for_playback
            # NÃO interrompe a reprodução em andamento (SoundPlayer não permite),
            # mas garante que esperas não fiquem indefinidas.
            if self._current_item:
                self._current_item.future.set()

    def wait_current(self, timeout: Optional[float] = None) -> bool:
        """Aguarda reprodução atual terminar."""
        # Aguarda item atual se houver
        if self._current_item:
            return self._current_item.future.wait(timeout=timeout or self._timeout)
        return True

    def wait_for_playback(self, timeout: Optional[float] = None) -> bool:
        """Aguarda toda a fila ser processada.
        
        Retorna False se shutdown foi chamado ou timeout expirou.
        """
        if not self._accepting_items and self._queue.empty() and not self._current_item:
            return True  # Já encerrado e nada pendente
            
        deadline = time.time() + (timeout or self._timeout * 10)
        while not self._queue.empty() or self._current_item:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            if self._current_item:
                if not self._current_item.future.wait(timeout=min(remaining, 0.5)):
                    continue
            else:
                time.sleep(0.1)
        return True

    def shutdown(self) -> None:
        """Encerra o player completamente.
        
        Não aceita mais itens, limpa fila, sinaliza esperas e aguarda worker.
        """
        self._accepting_items = False
        self._shutdown_event.set()
        self._worker_running = False
        self.stop()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)