"""Roteador genérico de provedores remotos (endpoint OpenAI-compatible).

Arquitetura:
    ConversationEngine -> CognitiveCore -> LLMProvider
                             |-> HybridLLMProvider (implementa LLMProvider,
                             |     resolve "online primeiro + fallback local"
                             |     pela network_policy)
                             `-> OnlineRouter (decide qual provedor remoto
                                   responder e converte a resposta em LLMResponse)

Regras de negócio
------------------
1. Nenhum provedor é hard-coded: os endpoints e as credenciais vêm de
   ``config/davios.json`` (campos ``remote_providers``) OU de variáveis de
   ambiente em forma genérica ``DAVIOS_PROVIDER_N_*`` (N = índice).
2. ``offline_mode``/``network_policy: "offline"`` desabilita a rede de uma vez.
3. ``network_policy: "auto"`` (padrão recomendado): tente o remoto primeiro;
   em quanto falhar, desça para o local. O local é sempre a garantia de
   resposta.
4. ``network_policy: "online"``: o remoto é obrigatório; se nenhum remoto
   responder, levanta sem cair no local (fica a critério do operador).
5. "Sem bloqueio do assistente": o tempo limite de cada requisição
   (``timeout_seconds``, default 8s) é rígido. Qualquer remoto que não
   responder dentro de X segundos é descartado e o fluxo cai no fallback
   local. O loop de texto do ConversationEngine é sincrono, mas o timeout
   garante que a decisão de fallback aconteça em tempo determinado.

Nota de projeto
---------------
- ``OnlineRouter`` NÃO herda ``LLMProvider`` e não executa a inferência local.
  Isso evita duplicar a lógica de carregamento de modelo de ``LocalLLMProvider``.
- ``HybridLLMProvider`` é o único nó que conhece tanto o roteador quanto o
  provedor local, mantendo ``CognitiveCore`` e ``ConversationEngine`` 100%
  neutros em relação à estratégia de rede.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib import request as url_request
from urllib.error import HTTPError, URLError

from brain.llm_provider import LLMProvider, LLMRequest, LLMResponse, LLMUnavailableError
from config.davios_config import DaviosConfig

logger = logging.getLogger("davios.llm.routing")


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------


@dataclass
class RemoteProvider:
    """Provedor remoto compatível com a API OpenAI (/v1/chat/completions)."""

    name: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 8.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


class RouterTransientError(LLMUnavailableError):
    """Falha que merece retry: rede, timeout, HTTP 429/5xx. Pode ser re-tentada."""


class RouterPermanentError(LLMUnavailableError):
    """Falha que nao se re-tenta: solicitacao invalida, modelo nao encontrado."""


class OnlineRouter:
    """Orientacao de requisicoes para provedores remotos.

    Nao possui estado de modelo. Soma Decide + Converte.
    """

    def __init__(self, config: Optional[DaviosConfig] = None, providers: Optional[list[RemoteProvider]] = None):
        self.config = config or DaviosConfig.load()
        self._providers = providers if providers is not None else self._enriched_providers()

    # -- visibilidade ------------------------------------------------------


    def is_network_allowed(self) -> bool:
        """True so se a rede estiver explicitamente habilitada."""
        if self.config.offline_mode:
            return False
        if self.config.network_policy == "offline":
            return False
        return True

    def get_enabled_providers(self) -> list[RemoteProvider]:
        """Filtra provedores habilitados (e com endpoint configurado)."""
        return [p for p in self._providers if p.enabled and p.base_url]
    # -- fluxo de geracao --------------------------------------------------

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Tenta o(remoto)s; levanta se nao houver resposta."""
        if not self.is_network_allowed():
            raise LLMUnavailableError(
                "modo offline: nenhum provider remoto sera consultado "
                "(offline_mode=true ou network_policy=offline)"
            )

        enabled = self.get_enabled_providers()
        if not enabled:
            raise LLMUnavailableError(
                "network_policy permitia rede, mas nenhum remote_provider "
                "esta habilitado/configurado no davios.json ou via "
                "DAVIOS_PROVIDER_N_*"
            )

        last_error = None
        for attempt in range(self.config.router.retry_count + 1):
            for provider in enabled:
                try:
                    return self._request_one(provider, request)
                except RouterTransientError:
                    # transitorio: abandona este provedor e tenta o proximo
                    continue
                except RouterPermanentError as exc:
                    last_error = exc
                    logger.warning(
                        "[ROUTER] provider=%s falhou de forma permanente: %s",
                        provider.name or "sem-nome",
                        exc,
                    )
                    # erro nao transitorio nao passa ao proximo provedor;
                    # o HybridLLMProvider decide o fallback local.
                    raise
            if attempt < self.config.router.retry_count:
                time.sleep(self._backoff(attempt))

        raise LLMUnavailableError(
            "todos os remote providers desconectados apos "
            + str(self.config.router.retry_count) + " tentativas de retry"
        ) from last_error

    def _request_one(self, provider: RemoteProvider, request: LLMRequest) -> LLMResponse:
        payload = self._build_payload(request, provider)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(provider.extra_headers)
        if provider.api_key:
            headers["Authorization"] = "Bearer " + provider.api_key
        url = provider.base_url.rstrip("/") + "/v1/chat/completions"
        started = time.perf_counter()
        try:
            req = url_request.Request(url, data=body, headers=headers, method="POST")
            with url_request.urlopen(req, timeout=provider.timeout_seconds) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise self._map_http_error(exc, raw, provider) from exc
        except (URLError, socket.timeout, TimeoutError, OSError) as exc:
            raise RouterTransientError(
                "conexao com " + provider.base_url + " falhou: " + repr(exc)
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        text = self._parse_body(raw, provider)
        if text is None:
            raise LLMUnavailableError(
                "resposta invalida do remote " + provider.base_url + ": " + raw[:200]
            )
        return self._to_llm_response(text, provider, elapsed_ms)
    # -- conversao de resposta ----------------------------------------------

    @staticmethod
    def _build_payload(request: LLMRequest, provider: RemoteProvider) -> dict:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        payload = {
            "model": provider.model,
            "messages": messages,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.stop:
            payload["stop"] = request.stop
        return payload

    @staticmethod
    def _parse_body(raw: str, provider: RemoteProvider) -> Optional[str]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        choices = data.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message") or {}
        return (message.get("content") or first.get("text") or "").strip()

    @staticmethod
    def _to_llm_response(text: str, provider: RemoteProvider, elapsed_ms: float) -> LLMResponse:
        return LLMResponse(
            text=text,
            provider="online",
            model=provider.model,
            backend="openai_compatible",
            tokens_generated=None,
            elapsed_ms=round(elapsed_ms, 3),
        )

    @staticmethod
    def _to_llm_error(request: LLMRequest, provider: RemoteProvider, exc: Exception) -> LLMResponse:
        logger.warning(
            "[ROUTER] provider=%s nao retornou texto: %s",
            provider.name or "sem-nome",
            exc,
        )
        return LLMResponse(
            text="",
            provider="online",
            model=provider.model,
            backend="openai_compatible",
            tokens_generated=None,
            elapsed_ms=0.0,
        )

    @staticmethod
    def _is_transient(exc: Exception, provider: RemoteProvider) -> bool:
        """Erro que merece retry: apenas RouterTransientError (rede, timeout, HTTP 429/5xx)."""
        return isinstance(exc, RouterTransientError)

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(8.0, 0.5 * (2 ** attempt))
    # -- provedores ---------------------------------------------------------

    def _enriched_providers(self) -> list[RemoteProvider]:
        """Config de arquivo + overrides por DAVIOS_PROVIDER_N_*."""
        providers = list(self.config.remote_providers)
        for index, env_prefix in enumerate(self._env_provider_prefixes()):
            base_url = self._env(var=env_prefix + "BASE_URL")
            if not base_url:
                continue
            providers.append(
                RemoteProvider(
                    name=self._env(var=env_prefix + "NAME") or "provider_" + str(index),
                    base_url=base_url,
                    api_key=self._env(var=env_prefix + "API_KEY") or "",
                    model=self._env(var=env_prefix + "MODEL") or "",
                    timeout_seconds=float(
                        self._env(var=env_prefix + "TIMEOUT_SECONDS") or self.config.router.timeout_seconds
                    ),
                    extra_headers=self._parse_headers(self._env(var=env_prefix + "HEADERS")),
                    enabled=self._env(var=env_prefix + "ENABLED") != "0",
                )
            )
        return providers

    def _env_provider_prefixes(self) -> list[str]:
        prefixes: list[str] = []
        index = 0
        while True:
            prefix = "DAVIOS_PROVIDER_" + str(index) + "_"
            if not os.environ.get(prefix):
                break
            prefixes.append(prefix)
            index += 1
        return prefixes

    @staticmethod
    def _env(var: str) -> Optional[str]:
        return os.environ.get(var)

    @staticmethod
    def _parse_headers(raw: Optional[str]) -> dict[str, str]:
        headers: dict[str, str] = {}
        if not raw:
            return headers
        for item in raw.split(","):
            if ":" in item:
                key, value = item.split(":", 1)
                headers[key.strip()] = value.strip()
        return headers

    @staticmethod
    def _map_http_error(exc: HTTPError, raw: str, provider: RemoteProvider) -> LLMUnavailableError:
        try:
            detail = json.loads(raw)[:200]
        except json.JSONDecodeError:
            detail = raw[:200]
        if exc.code in (429, 500, 502, 503, 504):
            raise RouterTransientError(
                "HTTP " + str(exc.code) + " de " + provider.base_url + ": " + str(detail)
            ) from exc
        raise RouterPermanentError(
            "HTTP " + str(exc.code) + " de " + provider.base_url + ": " + str(detail)
        ) from exc

class HybridLLMProvider(LLMProvider):
    """Implementa LLMProvider e resolve online primeiro + fallback local.

    Esta e a unica classe que une roteador e provedor local, mantendo
    CognitiveCore/ConversationEngine 100% neutros em relacao a estrategia
    de rede.
    """

    name = "hybrid"

    def __init__(
        self,
        config: Optional[DaviosConfig] = None,
        local_provider: Optional[LLMProvider] = None,
        router: Optional[OnlineRouter] = None,
    ):
        self.config = config or DaviosConfig.load()
        self.local_provider = local_provider
        self.router = router

    def initialize(self) -> bool:
        if self.local_provider is None:
            return False
        return self.local_provider.initialize()

    def is_available(self) -> bool:
        if self.local_provider is not None and self.local_provider.is_available():
            return True
        if self.router is not None and self.router.is_network_allowed():
            return True
        return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Online primeiro (se habilitado e disponivel), fallback local."""
        if self.local_provider is None:
            raise LLMUnavailableError("sem provedor local configurado")

        if self._network_blocked():
            return self._call_local(request, "rede bloqueada (offline_mode / network_policy=offline)")

        if self.router is not None and self.router.is_network_allowed():
            try:
                return self.router.generate(request)
            except LLMUnavailableError as exc:
                logger.warning(
                    "[ROUTER] remote indisponivel, caindo no local: %s", exc
                )

        return self._call_local(request, "fallback local apos falha do remote")

    def _network_blocked(self) -> bool:
        if self.config.offline_mode:
            return True
        return self.config.network_policy == "offline"

    def _call_local(self, request: LLMRequest, diagnosis: str) -> LLMResponse:
        if not self.local_provider.is_available():
            raise LLMUnavailableError(
                "sem provedor local disponivel (" + diagnosis + ")"
            )
        return self.local_provider.generate(request)

    def unload(self) -> None:
        if self.local_provider is not None:
            self.local_provider.unload()

    def health_check(self) -> dict[str, Any]:
        info = super().health_check()
        if self.local_provider is not None:
            info.update(self.local_provider.health_check())
        if self.router is not None:
            info["router_network_allowed"] = self.router.is_network_allowed()
        return info
