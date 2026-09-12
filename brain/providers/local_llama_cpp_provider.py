# Provider local que envolve o binario standalone llama.cpp (llama-server.exe)
# Arquitetura: LocalLlamaCppProvider -> llama-server.exe -> HTTP 127.0.0.1:PORT
# Comunicacao via HTTP localhost (NAO e Internet).
# GPU (Vulkan) opcional: fallback automatico para CPU.
# Nao requer llama-cpp-python.
# Nao requer Internet apos instalacao.

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Optional
from urllib import request as url_request
from urllib.error import HTTPError, URLError

from brain.llm_provider import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUnavailableError,
)
from brain.model_manager import ModelManager, ModelSelection
from config.davios_config import DaviosConfig

logger = logging.getLogger("davios.llm.llama_cpp")

LLAMA_CPP_INSTRUCTIONS = (
    "Backend local nao encontrado. "
    "Para habilitar o cerebro local offline: "
    "1) Instale o llama.cpp: .venv\\Scripts\\python.exe scripts\\setup_llama.py "
    "2) Baixe Qwen3-4B-Q4_K_M.gguf: .venv\\Scripts\\python.exe scripts\\download_model.py "
    "3) Reinicie o DaviOS. "
    "O DaviOS continua funcionando em modo regras sem o modelo."
)

SERVER_STARTUP_TIMEOUT = 60
HEALTH_CHECK_INTERVAL = 0.5


class LocalLlamaCppProvider(LLMProvider):
    """Provider que usa llama-server.exe como subprocesso local."""

    name = "local_llama_cpp"

    def __init__(
        self,
        config: Optional[DaviosConfig] = None,
        model_manager: Optional[ModelManager] = None,
        selection: Optional[ModelSelection] = None,
        server_port: int = 8080,
    ):
        self.config = config or DaviosConfig.load()
        self.model_manager = model_manager or ModelManager(self.config)
        self.selection = selection
        self._server_port = server_port
        self._process: Optional[subprocess.Popen] = None
        self._initialized = False
        self._available = False
        self._gpu_failed = False
        self._backend_used: str = "none"
        self._base_url = f"http://127.0.0.1:{server_port}"
        # Diagnostico de boot: preenchido a cada etapa do initialize().
        self.diagnostics: dict[str, Any] = {}
        # Nome REAL do modelo cargado no servidor (queried ao reutilizar un
        # servidor externo; setado ao iniciar uno nuevo). None = desconhecido.
        self._loaded_model: Optional[str] = None

    def initialize(self) -> bool:
        """Inicia llama-server.exe se ainda nao rodando. Nunca lanca excecao.

        Estrategia:
        1. Verifica se ja existe um servidor rodando (health check)
        2. Se existir, usa o servidor existente
        3. Se nao existir, inicia um novo servidor com o modelo selecionado
        """
        if self._initialized and self._available:
            return True

        self.diagnostics = {}
        server_exe = self._find_server_exe()
        if server_exe is None:
            self.diagnostics["[LLAMA_SERVER]"] = (
                "llama-server.exe nao encontrado em bin/llama.cpp nem no PATH."
            )
            self.diagnostics["[BACKEND]"] = "indisponivel: binario ausente."
            logger.info("[LLM] llama-server.exe nao encontrado.")
            return False
        self.diagnostics["[LLAMA_SERVER]"] = f"encontrado: {server_exe}"

        # 1. Reutiliza um servidor ja rodando externamente. O nome do modelo
        #    EXHIBIDO vem da CONSULTA real ao servidor (_query_loaded_model),
        #    nunca da seleccion teorica por RAM — assim o banner mostra o
        #    modelo de verdade cargado, nao um nome possivelmente errado.
        if self._check_existing_server():
            logger.info("[LLM] Usando llama-server ja rodando em %s", self._base_url)
            self._initialized = True
            self._available = True
            self._backend_used = "external"
            self.diagnostics["[HEALTH]"] = (
                f"servidor externo ja ativo em {self._base_url}; reutilizado."
            )
            self.diagnostics["[PROVIDER]"] = "ativo (servidor externo)."
            self.diagnostics["[BACKEND]"] = "llama.cpp standalone (externo)."
            real_model_path = self._query_loaded_model()
            if real_model_path:
                real_name = Path(real_model_path).name
                self._loaded_model = real_name
                self.diagnostics["[MODEL]"] = (
                    f"{real_name} (carregado no servidor externo)."
                )
                selected = (
                    self.selection.model.name
                    if self.selection and self.selection.model
                    else None
                )
                if selected and selected != real_name:
                    logger.warning(
                        "[LLM] Servidor externo ativo com modelo '%s'; "
                        "selecao teorica era '%s'. Usando o modelo real do "
                        "servidor para exibicion.",
                        real_name, selected,
                    )
            else:
                self._loaded_model = None
                self.diagnostics["[MODEL]"] = (
                    "servidor ativo com modelo desconocido, "
                    "nao verificado contra a seleccao atual."
                )
                logger.warning(
                    "[LLM] servidor ja ativo com modelo desconocido, "
                    "nao verificado contra a seleccion actual."
                )
            return True

        # 2. Caso contrario, inicia um novo servidor com o modelo selecionado
        if self.selection is None or self.selection.model is None:
            self.diagnostics["[MODEL]"] = (
                f"nenhum modelo .gguf compativel "
                f"({self.selection.reason if self.selection else 'sem selecao'})."
            )
            self.diagnostics["[BACKEND]"] = "indisponivel: modelo ausente."
            logger.info("[LLM] Nenhum modelo selecionado.")
            return False

        model_path = self.selection.model.path
        if not model_path.exists():
            self.diagnostics["[MODEL]"] = f"arquivo nao encontrado: {model_path}"
            self.diagnostics["[BACKEND]"] = "indisponivel: modelo ausente."
            logger.warning("[LLM] Modelo nao encontrado: %s", model_path)
            return False
        self.diagnostics["[MODEL]"] = (
            f"{self.selection.model.name} ({self.selection.model.size_gb} GB)"
        )

        # 2. Inicia um novo servidor
        gen = dict(self.selection.generation or {})
        n_threads = gen.get("threads") or self.config.threads or 4
        ctx_size = gen.get("context_window") or self.config.context_window or 2048

        # Primeiro tenta com GPU (Vulkan) se configurado
        if self.config.use_gpu and not self._gpu_failed:
            gpu_layers = gen.get("gpu_layers", self.config.gpu_layers)
            if gpu_layers and gpu_layers > 0:
                success = self._start_server(
                    server_exe, model_path, n_threads, ctx_size,
                    backend="vulkan", gpu_layers=gpu_layers,
                )
                if success:
                    self._gpu_failed = False
                    return True
                self._gpu_failed = True
                logger.warning("[LLM] Vulkan falhou; tentando CPU.")

        # Fallback CPU
        success = self._start_server(
            server_exe, model_path, n_threads, ctx_size,
            backend="cpu", gpu_layers=0,
        )
        if success:
            self._backend_used = "cpu"
            self.diagnostics["[HEALTH]"] = (
                f"llama-server iniciado e respondendo em {self._base_url}."
            )
            self.diagnostics["[PROVIDER]"] = "ativo (processo iniciado pelo DaviOS)."
            self.diagnostics["[BACKEND]"] = "llama.cpp standalone (cpu)."
            return True

        self.diagnostics["[HEALTH]"] = (
            "llama-server nao respondeu ao /health no tempo limite."
        )
        self.diagnostics["[PROVIDER]"] = "inativo: falha ao iniciar servidor."
        self.diagnostics["[BACKEND]"] = "indisponivel: servidor nao subiu."
        return False

    def _check_existing_server(self) -> bool:
        """Verifica se ja existe um llama-server rodando na porta configurada."""
        try:
            url = f"{self._base_url}/health"
            req = url_request.Request(url, method="GET")
            with url_request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8")
                if '"status":"ok"' in body or '"status": "ok"' in body:
                    return True
        except (URLError, HTTPError, ConnectionError, OSError):
            pass
        return False


    def _start_server(
        self,
        server_exe: Path,
        model_path: Path,
        n_threads: int,
        ctx_size: int,
        backend: str,
        gpu_layers: int,
    ) -> bool:
        """Inicia o processo llama-server e aguarda health check."""
        cmd = [
            str(server_exe),
            "--model", str(model_path),
            "--host", "127.0.0.1",
            "--port", str(self._server_port),
            "--threads", str(n_threads),
            "--ctx-size", str(ctx_size),
        ]

        if backend == "vulkan" and gpu_layers > 0:
            cmd.extend(["--backend", "vulkan", "--n-gpu-layers", str(gpu_layers)])

        # --jinja ativa o chat template real do modelo; necessario para
        # desativar o modo thinking do Qwen3 via chat_template_kwargs.
        cmd.extend(["--jinja", "--log-disable"])

        logger.info("[LLM] Iniciando llama-server: %s", " ".join(cmd))

        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
        except Exception as e:
            logger.error("[LLM] Falha ao iniciar llama-server: %s", e)
            return False

        if self._wait_for_server():
            self._initialized = True
            self._available = True
            self._backend_used = backend
            self._loaded_model = model_path.name
            logger.info(
                "[LLM] llama-server pronto (%s) porta %d",
                backend, self._server_port,
            )
            return True
        else:
            logger.error("[LLM] llama-server nao respondeu no timeout.")
            self._cleanup_process()
            return False

    def _wait_for_server(self, timeout: float = SERVER_STARTUP_TIMEOUT) -> bool:
        """Aguarda o servidor ficar pronto via health check HTTP."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                return False
            try:
                req = url_request.Request(f"{self._base_url}/health")
                with url_request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (URLError, HTTPError, ConnectionError, OSError):
                pass
            except Exception:
                pass
            time.sleep(HEALTH_CHECK_INTERVAL)
        return False

    def _find_server_exe(self) -> Optional[Path]:
        """Localiza llama-server.exe no sistema de arquivos."""
        from utils.llama_binary_downloader import get_llama_bin_dir

        bin_dir = get_llama_bin_dir()
        server = bin_dir / "llama-server.exe"
        if server.exists():
            return server

        found = shutil.which("llama-server.exe")
        if found:
            return Path(found)

        return None

    def _query_loaded_model(self) -> Optional[str]:
        """Consulta o servidor ativo para descobrir o modelo REAL cargado.

        Formato confirmado no backend real em uso (llama-server do projeto):
        - GET /v1/models -> {"data": [{"id": "<caminho-al-modelo>", ...}]}
          (esta versión tambien expone "models": [{"name": "<caminho>"}])
        - GET /props    -> {"model_path": "<caminho-al-modelo>", ...}

        Retorna o identificador (caminho) do modelo, o None se nao e
        posible confirmarlo.
        """
        try:
            req = url_request.Request(f"{self._base_url}/v1/models")
            with url_request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            entries = data.get("data") or data.get("models") or []
            if entries:
                ident = entries[0].get("id") or entries[0].get("name")
                if ident:
                    return str(ident)
        except Exception:
            pass
        try:
            req = url_request.Request(f"{self._base_url}/props")
            with url_request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            path = data.get("model_path")
            if path:
                return str(path)
        except Exception:
            pass
        return None

    def is_available(self) -> bool:
        """True se o servidor esta rodando e responde ao health check."""
        if not self._initialized:
            return False

        # Servidor externo: verifica health check diretamente
        if self._backend_used == "external":
            available = self._check_existing_server()
            self._available = available
            return available

        # Servido iniciado pelo provider: verifica processo
        if self._process is None:
            return False
        if self._process.poll() is not None:
            self._available = False
            return False
        try:
            req = url_request.Request(f"{self._base_url}/health")
            with url_request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Envia request HTTP para /v1/chat/completions."""
        if not self.is_available() and not self.initialize():
            raise LLMUnavailableError(LLAMA_CPP_INSTRUCTIONS)

        started = time.perf_counter()

        payload = {
            "model": "davios-local",
            "messages": self._build_messages(request),
            "max_tokens": request.max_tokens or self.config.max_tokens,
            "temperature": (
                request.temperature
                if request.temperature is not None
                else self.config.temperature
            ),
            # Penalidade de repeticao no nivel de sampling (logits): reprime
            # loops de repeticion; o llama-server aceita este campo no
            # endpoint OpenAI-compatible /v1/chat/completions (confirmado nos
            # strings do proprio llama-server-impl.dll em uso).
            "repeat_penalty": (
                request.repeat_penalty
                if request.repeat_penalty is not None
                else self.config.repeat_penalty
            ),
            # repeat_last_n: cuantos tokens recientes considera la penalidad
            # de repeticion (default llama.cpp 64 es corto para cubrir el
            # historico de conversacion en el prompt). Campo aceptado por el
            # backend real: confirmado con POST 200 a /v1/chat/completions.
            "repeat_last_n": (
                request.repeat_last_n
                if request.repeat_last_n is not None
                else self.config.repeat_last_n
            ),
            # Qwen3: desativa o modo thinking para o texto util sair direto
            # em 'content' (evita vazamento de raciocinio na resposta).
            "chat_template_kwargs": {"enable_thinking": False},
        }

        body = json.dumps(payload).encode("utf-8")

        try:
            req = url_request.Request(
                f"{self._base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with url_request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, ConnectionError, OSError) as e:
            logger.warning("[LLM] Erro na geracao: %s", e)
            raise LLMUnavailableError(f"Falha na geracao do modelo local: {e}")
        except Exception as e:
            logger.warning("[LLM] Erro inesperado: %s", e)
            raise LLMUnavailableError(f"Falha na geracao do modelo local: {e}")

        elapsed_ms = (time.perf_counter() - started) * 1000
        text = self._extract_text(result)
        tokens = self._extract_tokens(result)

        return LLMResponse(
            text=text,
            provider=self.name,
            model=(
                self.selection.model.name
                if self.selection and self.selection.model
                else "local"
            ),
            backend=f"llama_cpp:{self._backend_used}",
            tokens_generated=tokens,
            elapsed_ms=round(elapsed_ms, 1),
        )

    @staticmethod
    def _build_messages(request: LLMRequest) -> list[dict[str, str]]:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    @staticmethod
    def _extract_text(output: dict) -> str:
        try:
            choice = output["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content", "")
            # Qwen3 reasoning mode: se content vazio, usar reasoning_content
            if not content:
                content = message.get("reasoning_content", "")
            if not content:
                content = choice.get("text", "")
            # Redes de seguranca: remove raciocinio vazado do Qwen3 mesmo
            # com enable_thinking=false (alguns templates ainda o emitem).
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            content = re.sub(r"^\s*<think>.*", "", content, flags=re.DOTALL)
            return content.strip()
        except Exception:
            return str(output)

    @staticmethod
    def _extract_tokens(output: dict) -> Optional[int]:
        try:
            return output["usage"]["completion_tokens"]
        except (KeyError, TypeError):
            return None

    def unload(self) -> None:
        """Para o processo llama-server e libera recursos."""
        self._cleanup_process()
        self._initialized = False
        self._available = False
        logger.info("[LLM] llama-server finalizado e liberado.")

    def _cleanup_process(self) -> None:
        """Encerra o processo llama-server gracefulmente."""
        if self._process is None:
            return
        try:
            if self._process.poll() is None:
                if os.name == "nt":
                    self._process.terminate()
                else:
                    self._process.send_signal(signal.SIGTERM)
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
        except Exception as e:
            logger.warning("[LLM] Erro ao encerrar llama-server: %s", e)
        finally:
            self._process = None

    def health_check(self) -> dict[str, Any]:
        info = super().health_check()
        info.update(
            {
                "backend": f"llama_cpp:{self._backend_used}" if self._available else "none",
                "model": (
                    self._loaded_model
                    or (
                        self.selection.model.name
                        if self.selection and self.selection.model
                        else None
                    )
                ),
                "gpu_failed": self._gpu_failed,
                "server_port": self._server_port,
                "base_url": self._base_url,
                "process_alive": self._process is not None and self._process.poll() is None,
            }
        )
        return info
