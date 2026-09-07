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
