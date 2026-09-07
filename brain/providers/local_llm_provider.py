"""Provider local de LLM (GGUF via llama-cpp-python).

- Roda 100% in-process e offline: nenhum dado sai da máquina.
- ``llama-cpp-python`` é OPCIONAL: sem ele, o provider reporta
  indisponível e o DaviOS continua funcionando em modo regras.
- GPU é opcional: qualquer falha de offload faz fallback para CPU.
- O modelo permanece carregado durante a sessão.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from brain.llm_provider import (
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMUnavailableError,
)
from config.davios_config import DaviosConfig
from brain.model_manager import ModelManager, ModelSelection

logger = logging.getLogger("davios.llm.local")

LLAMA_CPP_INSTRUCTIONS = (
    "Modelo local nao encontrado.\n"
    "Para habilitar o cerebro local:\n"
    "  1. Instale o backend:  pip install llama-cpp-python\n"
    "  2. Baixe um modelo GGUF (ex.: Qwen2.5-1.5B-Instruct-Q4_K_M.gguf)\n"
    "     e coloque em models/balanced/\n"
    "  3. Reinicie o DaviOS.\n"
    "O DaviOS continua funcionando em modo regras sem o modelo."
)


class LocalLLMProvider(LLMProvider):
    """Provider de inferência local via llama-cpp-python (GGUF)."""

    name = "local"

    def __init__(
        self,
        config: Optional[DaviosConfig] = None,
        model_manager: Optional[ModelManager] = None,
        selection: Optional[ModelSelection] = None,
    ):
        self.config = config or DaviosConfig.load()
        self.model_manager = model_manager or ModelManager(self.config)
        self.selection = selection
        self._llama: Any = None
        self._llm: Any = None
        self._initialized = False
        self._gpu_failed = False

    def initialize(self) -> bool:
        """Tenta carregar o modelo selecionado. Nunca lança exceção."""
        if self._initialized and self._llm is not None:
            return True
        try:
            import llama_cpp  # type: ignore
        except Exception:
            logger.info("[LLM] llama-cpp-python nao instalado; modo regras.")
            return False

        if self.selection is None or self.selection.model is None:
            return False

        try:
            self._llama = llama_cpp
            gen = dict(self.selection.generation or {})
            n_threads = gen.get("threads") or self.config.threads or None
            kwargs: dict[str, Any] = {
                "model_path": str(self.selection.model.path),
                "n_ctx": gen.get("context_window", self.config.context_window),
                "n_threads": n_threads,
                "verbose": self.config.debug,
            }
            n_gpu_layers = self._resolve_gpu_layers(gen)
            if n_gpu_layers:
                kwargs["n_gpu_layers"] = n_gpu_layers
            self._llm = self._llama.Llama(**kwargs)
            self._initialized = True
            logger.info(
                "[MODEL] carregado: %s (gpu_layers=%s)",
                self.selection.model.name,
                n_gpu_layers,
            )
            return True
        except Exception as exc:
            if self._gpu_failed:
                logger.warning("[MODEL] falha com GPU; tentando CPU puro: %s", exc)
                return self._retry_cpu_only()
            logger.warning("[MODEL] falha ao carregar modelo: %s", exc)
            self._llm = None
            self._initialized = False
            return False

    def _resolve_gpu_layers(self, gen: dict[str, Any]) -> int:
        """GPU é opcional: só offload se config pedir e ainda não falhou."""
        if self._gpu_failed or not self.config.use_gpu:
            return 0
        layers = gen.get("gpu_layers", self.config.gpu_layers)
        if layers and layers > 0:
            self._gpu_failed = True  # próxima tentativa será CPU
        return layers or 0

    def _retry_cpu_only(self) -> bool:
        try:
            self._llm = self._llama.Llama(
                model_path=str(self.selection.model.path),
                n_ctx=self.selection.generation.get(
                    "context_window", self.config.context_window
                ),
                verbose=self.config.debug,
            )
            self._initialized = True
            self._gpu_failed = True
            logger.info("[MODEL] modelo carregado em CPU (fallback).")
            return True
        except Exception as exc:
            logger.warning("[MODEL] falha tambem em CPU: %s", exc)
            self._llm = None
            self._initialized = False
            return False

    def is_available(self) -> bool:
        return self._llm is not None

    def unload(self) -> None:
        """Libera o modelo da memória."""
        self._llm = None
        self._initialized = False
        logger.info("[MODEL] modelo liberado da memoria.")

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.is_available() and not self.initialize():
            raise LLMUnavailableError(LLAMA_CPP_INSTRUCTIONS)

        started = time.perf_counter()
        try:
            output = self._llm.create_chat_completion(
                messages=self._build_messages(request),
                max_tokens=request.max_tokens or self.config.max_tokens,
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else self.config.temperature
                ),
                stop=request.stop or None,
            )
        except Exception as exc:
            logger.warning("[LLM] erro na geracao: %s", exc)
            raise LLMUnavailableError(f"Falha na geracao do modelo local: {exc}")

        elapsed_ms = (time.perf_counter() - started) * 1000
        text = self._extract_text(output)
        tokens = None
        try:
            tokens = output["usage"]["completion_tokens"]
        except Exception:
            pass

        model_name = (
            self.selection.model.name
            if self.selection and self.selection.model
            else "local"
        )
        return LLMResponse(
            text=text,
            provider=self.name,
            model=model_name,
            backend="llama_cpp",
            tokens_generated=tokens,
            elapsed_ms=round(elapsed_ms, 1),
        )

    @staticmethod
    def _extract_text(output: Any) -> str:
        try:
            choice = output["choices"][0]
            message = choice.get("message") or {}
            return (message.get("content") or choice.get("text") or "").strip()
        except Exception:
            return str(output)

    @staticmethod
    def _build_messages(request: LLMRequest) -> list[dict[str, str]]:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def health_check(self) -> dict[str, Any]:
        info = super().health_check()
        info.update(
            {
                "backend": "llama_cpp" if self._llm is not None else "none",
                "model": (
                    self.selection.model.name
                    if self.selection and self.selection.model
                    else None
                ),
                "gpu_failed": self._gpu_failed,
            }
        )
        return info

