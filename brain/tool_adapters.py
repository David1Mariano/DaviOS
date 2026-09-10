"""Adapters de ferramentas reais do DaviOS (C2).

Transforma comandos já existentes do ActionManager e do WebAccess em
instâncias de Tool e as registra no ToolRegistry.

REGRAS DESTA ETAPA:
    - ActionManager.py e web_access.py NÃO são modificados — apenas
      lemos a interface pública deles:
        ActionManager: list_commands() / is_allowed() / execute(name, args)
        WebAccess:     fetch(url)
    - O registro é EXPLÍCITO: ``register_default_tools(registry)`` deve ser
      chamado apenas em testes por agora. NÃO é chamado em main.py —
      a conexão com Qwen/ConversationEngine/CognitiveCore/PromptBuilder
      vem depois (C3+).
    - Nome e descrição de cada comando real são preservados para uso
      futuro na construção de prompts (C3).

Conversão de resultados:
    - ActionManager (ActionResult.ok=False) → ToolError(result.message),
      capturado pelo ToolRouter → ToolExecutionResult(success=False).
    - WebAccess (WebResult.ok=False)        → ToolError(result.error),
      mesmo caminho padronizado.
    - Sucesso → ActionManager: stdout ou message (str);
                WebAccess: WebResult.to_dict() (text/title/url/status
                preservados para consumo futuro pelo prompt).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from brain.action_manager import ActionManager
from brain.tool_registry import (
    DuplicateToolError,
    Tool,
    ToolError,
    ToolRegistry,
)
from brain.web_access import WebAccess

logger = logging.getLogger("davios.tools")


# ---------------------------------------------------------------------------
# Adapters do ActionManager
# ---------------------------------------------------------------------------


def make_action_manager_tool(
    action_manager: ActionManager,
    name: str,
    description: Optional[str] = None,
) -> Tool:
    """Cria uma Tool que encapsula um comando real da allowlist.

    ``name`` e ``description`` preservam exatamente os valores do comando
    original do ActionManager (usados futuramente no prompt, C3).
    """
    if not action_manager.is_allowed(name):
        raise ToolError(f"Comando '{name}' não existe no ActionManager.")
    if description is None:
        for cmd in action_manager.list_commands():
            if cmd["name"] == name:
                description = cmd["description"]
                break
    if not description:
        description = f"Executa o comando '{name}' do sistema."

    def _execute(args: str = "") -> str:
        """Executa o comando real e devolve a saída textual.

        ActionResult.ok=False vira ToolError — o ToolRouter captura e
        padroniza (success=False, error=mensagem).
        """
        result = action_manager.execute(name, args=args)
        if not result.ok:
            raise ToolError(result.message or f"Comando '{name}' falhou.")
        return result.stdout or result.message

    return Tool(
        name=name,
        description=description,
        arguments=["args"],
        execute=_execute,
        category="actions",
    )


def register_action_manager_tools(
    registry: ToolRegistry,
    action_manager: ActionManager,
) -> list[str]:
    """Registra TODOS os comandos da allowlist do ActionManager como Tools.

    Preserva nome/descrição reais de cada comando. Ignora nomes já
    registrados (idempotente). Retorna os nomes efetivamente registrados.
    """
    registered: list[str] = []
    for cmd in action_manager.list_commands():
        try:
            tool = make_action_manager_tool(
                action_manager, cmd["name"], cmd["description"]
            )
            registry.register(tool)
            registered.append(tool.name)
            logger.debug("[TOOL_ADAPTERS] registrada: %s", tool.name)
        except DuplicateToolError:
            logger.debug("[TOOL_ADAPTERS] '%s' já registrada — ignorada.", cmd["name"])
    return registered


# ---------------------------------------------------------------------------
# Adapters do WebAccess
# ---------------------------------------------------------------------------


def make_web_fetch_tool(web_access: WebAccess) -> Tool:
    """Cria uma Tool que encapsula WebAccess.fetch(url)."""

    def _execute(url: str) -> dict[str, Any]:
        """Executa a busca real e devolve o conteúdo sanitizado.

        WebResult.ok=False vira ToolError(result.error) — o ToolRouter
        captura e padroniza. Sucesso devolve o dict completo (text,
        title, url, status) preservado para o prompt futuro (C3).
        """
        result = web_access.fetch(url)
        if not result.ok:
            raise ToolError(result.error or f"Falha ao acessar '{url}'.")
        return result.to_dict()

    return Tool(
        name="web_fetch",
        description=(
            "Busca e sanitiza o conteúdo de uma página web (HTTP/HTTPS), "
            "retornando o texto limpo, o título e a URL final."
        ),
        arguments=["url"],
        execute=_execute,
        category="web",
    )


def register_web_access_tools(
    registry: ToolRegistry,
    web_access: WebAccess,
) -> list[str]:
    """Registra as ferramentas do WebAccess. Idempotente."""
    registered: list[str] = []
    try:
        registry.register(make_web_fetch_tool(web_access))
        registered.append("web_fetch")
    except DuplicateToolError:
        logger.debug("[TOOL_ADAPTERS] 'web_fetch' já registrada — ignorada.")
    return registered


# ---------------------------------------------------------------------------
# Registro explícito das ferramentas padrão
# ---------------------------------------------------------------------------


def register_default_tools(
    registry: ToolRegistry,
    *,
    action_manager: Optional[ActionManager] = None,
    web_access: Optional[WebAccess] = None,
) -> list[str]:
    """Registra as ferramentas padrão do DaviOS em um ToolRegistry.

    Chamado EXPLICITAMENTE (apenas em testes, por enquanto). Não é
    conectado ao main.py nem a nenhum fluxo de geração de respostas.

    - ``action_manager``: usa a instância informada ou cria uma nova
      habilitada (a allowlist/validação do ActionManager continua
      protegendo cada execução).
    - ``web_access``: usa a instância informada ou cria uma nova com
      enabled=False (acesso à web permanece desativado por padrão).

    Idempotente: nomes já presentes no registry são ignorados, portanto
    chamar duas vezes NÃO duplica ferramentas.

    Retorna a lista de nomes efetivamente registrados nesta chamada.
    """
    am = action_manager if action_manager is not None else ActionManager(enabled=True)
    wa = web_access if web_access is not None else WebAccess(enabled=False)

    registered: list[str] = []
    registered.extend(register_action_manager_tools(registry, am))
    registered.extend(register_web_access_tools(registry, wa))
    return registered
