"""Provider TTS usando SAPI5 (Windows nativo)."""

from __future__ import annotations

import logging
import platform
import tempfile
import os
import time
import threading
from typing import Optional

from voice.providers.base import TTSProvider, TTSProviderInfo


logger = logging.getLogger("davios.voice.providers.sapi5")


class SAPI5Provider(TTSProvider):
    """Provider para Microsoft SAPI5 (Windows built-in TTS)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._voice_name_filter = config.get("voice_name", "")  # vazio = auto-detect PT-BR
        self._rate = config.get("rate", 0)  # -10 a 10
        self._volume = config.get("volume", 100)  # 0 a 100
        self._timeout = config.get("timeout_seconds", 10.0)
        self._engine = None
        # Lock para serializar chamadas de síntese
        self._synthesis_lock = threading.Lock()
        # Contador de threads COM ativas + Condition para esperar término
        self._active_com_syntheses = 0
        self._com_condition = threading.Condition(self._synthesis_lock)

    @property
    def provider_name(self) -> str:
        return "sapi5"

    def check_availability(self) -> TTSProviderInfo:
        """Verifica se SAPI5 está disponível e encontra voz PT-BR."""
        if platform.system() != "Windows":
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason="SAPI5 só disponível no Windows",
            )

        try:
            import win32com.client
        except ImportError:
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason="pywin32 não instalado (pip install pywin32)",
            )

        try:
            engine = win32com.client.Dispatch("SAPI.SpVoice")
            voices = engine.GetVoices()
            if voices.Count == 0:
                return TTSProviderInfo(
                    name=self.provider_name,
                    available=False,
                    reason="Nenhuma voz SAPI5 instalada no sistema",
                )

            # Procura voz PT-BR
            selected_voice = None
            voice_name = ""
            for i in range(voices.Count):
                voice = voices.Item(i)
                desc = voice.GetDescription()
                # Procura por português/PT-BR/Brazil
                if any(kw in desc.lower() for kw in ["portuguese", "portugu", "brazil", "pt-br", "pt_br"]):
                    selected_voice = voice
                    voice_name = desc
                    break

            # Se não achou PT-BR, usa a primeira disponível
            if selected_voice is None:
                selected_voice = voices.Item(0)
                voice_name = selected_voice.GetDescription()
                logger.warning("[%s] Voz PT-BR não encontrada, usando: %s",
                              self.provider_name, voice_name)

            return TTSProviderInfo(
                name=self.provider_name,
                available=True,
                reason="",
                voice_name=voice_name,
                sample_rate=22050,  # SAPI5 padrão
            )

        except Exception as e:
            logger.exception("[%s] Erro ao verificar vozes SAPI5", self.provider_name)
            return TTSProviderInfo(
                name=self.provider_name,
                available=False,
                reason=f"Erro ao acessar SAPI5: {e}",
            )

    def synthesize(self, text: str) -> Optional[bytes]:
        """Sintetiza texto usando SAPI5 para arquivo WAV com timeout.
        
        Serializa chamadas via lock + contador de threads COM ativas.
        Timeout retorna None mas NÃO libera o slot de síntese enquanto a
        thread COM anterior estiver ativa. Próximas chamadas aguardam
        o término real da thread COM anterior.
        """
        if not text or not text.strip():
            logger.debug("[%s] Texto vazio, pulando síntese", self.provider_name)
            return None

        if platform.system() != "Windows":
            logger.error("[%s] SAPI5 só funciona no Windows", self.provider_name)
            return None

        try:
            import win32com.client
        except ImportError:
            logger.error("[%s] pywin32 não disponível", self.provider_name)
            return None

        if not self.initialize():
            return None

        # Aguarda síntese COM anterior terminar (se houver) antes de prosseguir.
        # Usa Condition associada ao lock para serialização completa.
        with self._com_condition:
            # Aguarda até que não haja threads COM ativas
            # Timeout aqui é generoso: aguarda síntese anterior terminar de verdade
            wait_timeout = self._timeout * 3  # tempo extra para síntese anterior completar
            end_time = time.time() + wait_timeout
            while self._active_com_syntheses > 0:
                remaining = end_time - time.time()
                if remaining <= 0:
                    logger.error("[%s] Timeout aguardando síntese COM anterior terminar (%.1fs)",
                                 self.provider_name, wait_timeout)
                    return None
                self._com_condition.wait(timeout=remaining)
            
            # Marca que uma nova síntese COM vai iniciar
            self._active_com_syntheses += 1

        # Arquivo temporário para saída WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            output_path = tmp_out.name

        result_container = {"audio_data": None, "error": None, "completed": False}

        def _synthesize_in_thread():
            try:
                engine = win32com.client.Dispatch("SAPI.SpVoice")
                
                # Seleciona voz se especificada
                if self._voice_name_filter:
                    voices = engine.GetVoices()
                    for i in range(voices.Count):
                        voice = voices.Item(i)
                        if self._voice_name_filter.lower() in voice.GetDescription().lower():
                            engine.Voice = voice
                            break

                # Configura rate e volume
                engine.Rate = self._rate
                engine.Volume = self._volume

                # Cria stream de arquivo
                stream = win32com.client.Dispatch("SAPI.SpFileStream")
                stream.Open(output_path, 3, False)  # SSFMCreateForWrite = 3
                engine.AudioOutputStream = stream

                # Sintetiza
                engine.Speak(text)
                
                # Aguarda conclusão (SAPI5 é síncrono mas pode ter buffer)
                stream.Close()
                
                # Pequena pausa para garantir flush
                time.sleep(0.1)

                # Lê arquivo gerado
                if os.path.exists(output_path):
                    with open(output_path, "rb") as f:
                        audio_data = f.read()
                    
                    if audio_data:
                        logger.debug("[%s] Síntese OK: %d bytes", self.provider_name, len(audio_data))
                        result_container["audio_data"] = audio_data
                    else:
                        logger.error("[%s] Arquivo de saída vazio", self.provider_name)
                        result_container["error"] = "Arquivo de saída vazio"
                else:
                    logger.error("[%s] Arquivo não foi criado", self.provider_name)
                    result_container["error"] = "Arquivo não foi criado"

            except Exception as e:
                logger.exception("[%s] Erro na síntese", self.provider_name)
                result_container["error"] = str(e)
            finally:
                result_container["completed"] = True
                # Libera o slot de síntese COM quando a thread termina (sucesso, erro ou exceção)
                with self._com_condition:
                    self._active_com_syntheses -= 1
                    self._com_condition.notify_all()

        # Executa síntese em thread separada com timeout
        synthesis_thread = threading.Thread(target=_synthesize_in_thread, daemon=True)
        synthesis_thread.start()
        synthesis_thread.join(timeout=self._timeout)

        if not result_container["completed"]:
            logger.error("[%s] Timeout na síntese (%.1fs) - thread COM continua em background, slot ocupado",
                         self.provider_name, self._timeout)
            try:
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception:
                pass
            # NÃO decrementa _active_com_syntheses aqui - a thread COM fará isso ao terminar
            return None

        if result_container["error"]:
            logger.error("[%s] Falha na síntese: %s", self.provider_name, result_container["error"])
            try:
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception:
                pass
            return None

        # Sucesso: limpa arquivo temporário após ler os dados
        try:
            if os.path.exists(output_path):
                os.unlink(output_path)
        except Exception:
            pass

        return result_container["audio_data"]