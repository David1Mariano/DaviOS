"""Adapters de ferramentas do celular (Termux) para o DaviOS.

Transforma métodos do PhoneClient em instâncias de Tool e registra no ToolRegistry.
Seguem o mesmo padrão de brain/tool_adapters.py: factories + registro idempotente.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from brain.phone_client import PhoneClient
from brain.tool_registry import DuplicateToolError, Tool, ToolError, ToolRegistry

logger = logging.getLogger("davios.tools")


# ---------------------------------------------------------------------------
# Sensores seguros — category "phone_read"
# ---------------------------------------------------------------------------


def make_battery_tool(client: PhoneClient) -> Tool:
    """Cria Tool para ler nível de bateria do celular."""

    def _execute() -> dict[str, Any]:
        result = client.get_battery()
        if not result["ok"]:
            raise ToolError(result["error"])
        return result

    return Tool(
        name="phone_battery",
        description=(
            "Retorna o nível de bateria (0-100) e se está carregando. "
            "Use apenas quando o usuário pedir explicitamente para ver a bateria."
        ),
        arguments=[],
        execute=_execute,
        category="phone_read",
    )


def make_location_tool(client: PhoneClient) -> Tool:
    """Cria Tool para ler localização GPS do celular."""

    def _execute() -> dict[str, Any]:
        result = client.get_location()
        if not result["ok"]:
            raise ToolError(result["error"])
        return result

    return Tool(
        name="phone_location",
        description=(
            "Retorna a localização GPS atual (latitude, longitude, precisão em metros). "
            "Use apenas quando o usuário pedir explicitamente a localização."
        ),
        arguments=[],
        execute=_execute,
        category="phone_read",
    )


def make_device_info_tool(client: PhoneClient) -> Tool:
    """Cria Tool para ler informações do dispositivo."""

    def _execute() -> dict[str, Any]:
        result = client.get_device_info()
        if not result["ok"]:
            raise ToolError(result["error"])
        return result

    return Tool(
        name="phone_device_info",
        description=(
            "Retorna informações do celular (modelo, versão do Android, etc.). "
            "Use apenas quando o usuário pedir explicitamente informações do dispositivo."
        ),
        arguments=[],
        execute=_execute,
        category="phone_read",
    )


# ---------------------------------------------------------------------------
# Ações sensíveis — category "phone_act"
# ---------------------------------------------------------------------------


def make_speak_tool(client: PhoneClient) -> Tool:
    """Cria Tool para fazer o celular falar via TTS."""

    def _execute(text: str) -> dict[str, Any]:
        result = client.speak(text)
        if not result["ok"]:
            raise ToolError(result["error"])
        return result

    return Tool(
        name="phone_speak",
        description=(
            "Faz o celular falar o texto informado via TTS. "
            "Use apenas quando o usuário pedir explicitamente para o celular falar algo."
        ),
        arguments=["text"],
        execute=_execute,
        category="phone_act",
    )


def make_vibrate_tool(client: PhoneClient) -> Tool:
    """Cria Tool para vibrar o celular."""

    def _execute(duration_ms: int) -> dict[str, Any]:
        result = client.vibrate(duration_ms)
        if not result["ok"]:
            raise ToolError(result["error"])
        return result

    return Tool(
        name="phone_vibrate",
        description=(
            "Faz o celular vibrar pela duração informada em milissegundos (50-2000). "
            "Use apenas quando o usuário pedir explicitamente para vibrar o celular."
        ),
        arguments=["duration_ms"],
        execute=_execute,
        category="phone_act",
    )


# ---------------------------------------------------------------------------
# Registro idempotente em lote
# ---------------------------------------------------------------------------


def register_phone_tools(
    registry: ToolRegistry,
    client: PhoneClient,
) -> list[str]:
    """Registra todas as ferramentas do celular no ToolRegistry.

    Idempotente: nomes já presentes são ignorados (DuplicateToolError).
    Retorna lista de nomes efetivamente registrados nesta chamada.
    """
    registered: list[str] = []

    factories_read = [
        make_battery_tool,
        make_location_tool,
        make_device_info_tool,
    ]
    factories_act = [
        make_speak_tool,
        make_vibrate_tool,
    ]

    for maker in factories_read + factories_act:
        try:
            tool = maker(client)
            registry.register(tool)
            registered.append(tool.name)
            logger.debug("[PHONE_ADAPTERS] registrada: %s", tool.name)
        except DuplicateToolError:
            logger.debug("[PHONE_ADAPTERS] '%s' já registrada — ignorada.", maker.__name__)

    return registered