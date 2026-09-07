"""Interface abstrata de provedores de LLM do DaviOS.

O ConversationEngine enxerga apenas esta interface — nunca o backend
concreto (llama_cpp, etc.). Isso permite trocar backend/provider sem
tocar no fluxo conversacional.

A política de rede (offline-first) é aplicada aqui: providers remotos
futuros devem recusar operação quando offline_mode estiver ativo.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("davios.llm")


class LLMUnavailableError(RuntimeError):
    """Nenhum modelo/provedor disponível para gerar resposta."""


@dataclass
class LLMRequest:
    """Requisição de geração já estruturada (prompt pronto)."""

    prompt: str
    system: str = ""
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "system": self.system,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stop": list(self.stop),
        }


@dataclass
class LLMResponse:
    """Resposta do provider com metadados de execução."""

    text: str
    provider: str = "none"
    model: str = "none"
    backend: str = "none"
    tokens_generated: Optional[int] = None
    elapsed_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "backend": self.backend,
            "tokens_generated": self.tokens_generated,
            "elapsed_ms": self.elapsed_ms,
        }


class LLMProvider(ABC):
    """Contrato comum para todos os provedores de modelo."""

    name: str = "base"

    @abstractmethod
    def initialize(self) -> bool:
        """Carrega o modelo/estabelece recursos. True se ficou pronto."""

    @abstractmethod
    def is_available(self) -> bool:
        """True se há modelo carregado/instalado pronto para gerar."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Gera texto a partir de uma LLMRequest."""

    @abstractmethod
    def unload(self) -> None:
        """Libera o modelo e recursos associados."""

    def health_check(self) -> dict[str, Any]:
        """Diagnóstico rápido de disponibilidade."""
        return {
            "provider": self.name,
            "available": self.is_available(),
        }
