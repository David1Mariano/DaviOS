"""ToolRegistry e ToolRouter do DaviOS.

Camada intermediária para registrar, localizar e EXECUTAR ferramentas.

NOTA (C2): o ToolRouter agora executa ferramentas de verdade via
``ToolRouter.execute(name, args=...)``:
    - nunca deixa uma exceção da ferramenta derrubar o chamador;
    - retorna um ToolExecutionResult padronizado (success/output/error/tool_name);
    - ferramenta inexistente reaproveita o mesmo comportamento de ``route()``;
    - argumentos inválidos/faltando produzem erro padronizado.

Os adapters das ferramentas reais (ActionManager, WebAccess) ficam em
``brain/tool_adapters.py`` e são registrados explicitamente via
``register_default_tools(registry)`` — apenas em testes, por enquanto.

Arquitetura alvo:
    Qwen → (futuramente) solicita ferramenta → ToolRouter → ToolRegistry → ferramenta

Padrão semelhante a:
    - ActionManager.register() / is_allowed() / execute()
    - AgentHandler.register() / handle()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("davios.tools")


class ToolError(Exception):
    """Erro base para operações com ferramentas."""


class DuplicateToolError(ToolError):
    """Ferramenta com mesmo nome já registrada."""


class InvalidToolError(ToolError):
    """Ferramenta inválida (nome/descrição ausente ou tipo incorreto)."""


class ToolNotFoundError(ToolError):
    """Ferramenta não encontrada no registro."""


@dataclass
class Tool:
    """Representa uma ferramenta disponível no DaviOS.

    ``execute`` é uma referência opcional para a função que efetivamente
    executa a ferramenta. Nesta etapa (C1) a função NÃO é chamada — serve
    apenas como metadado para futura integração com tool calling.
    """

    name: str
    description: str
    arguments: list[str] = field(default_factory=list)
    execute: Optional[Callable[..., Any]] = None
    # C5: categoria da ferramenta para a REGRA DE OURO de autorização:
    # "actions" → exige config.actions_enabled; "web" → exige config.web_enabled;
    # "" (vazio/ desconhecido) → NUNCA é auto-executável (default seguro).
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": list(self.arguments),
            "has_execute": self.execute is not None,
        }


@dataclass
class ToolRouterResult:
    """Resultado de uma tentativa de roteamento.

    Indica se a ferramenta foi encontrada e está pronta para execução.
    ``ToolRouter.execute()`` reaproveita este resultado antes de executar.
    """

    found: bool
    tool: Optional[Tool] = None
    error: str = ""


@dataclass
class ToolExecutionResult:
    """Resultado padronizado da execução de uma ferramenta.

    Sempre presente, mesmo em falha:
        - success=True  → ferramenta executou e devolveu ``output``;
        - success=False → ``error`` descreve o problema (ferramenta
          inexistente, sem execute, argumentos inválidos ou exceção
          capturada durante a execução).
    """

    success: bool
    output: Any = None
    error: str = ""
    tool_name: str = ""


class ToolRegistry:
    """Registro central de ferramentas conhecidas pelo DaviOS.

    Conhece as ferramentas disponíveis, suas descrições e argumentos.
    Não executa ferramentas — apenas registra e recupera metadados.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Registra uma ferramenta. Rejeita duplicatas e dados inválidos."""
        if not isinstance(tool, Tool):
            raise InvalidToolError("Ferramenta deve ser uma instância de Tool.")
        if not tool.name or not tool.name.strip():
            raise InvalidToolError("Ferramenta precisa ter nome não-vazio.")
        if not tool.description or not tool.description.strip():
            raise InvalidToolError(f"Ferramenta '{tool.name}' precisa ter descrição.")
        name = tool.name.strip()
        if name in self._tools:
            raise DuplicateToolError(f"Ferramenta '{name}' já está registrada.")
        self._tools[name] = tool
        logger.debug("[TOOL_REGISTRY] registrada: %s", name)

    def get(self, name: str) -> Optional[Tool]:
        """Recupera uma ferramenta pelo nome. Retorna None se não existir."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Verifica se uma ferramenta está registrada."""
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        """Lista todas as ferramentas registradas (cópia defensiva)."""
        return list(self._tools.values())

    def count(self) -> int:
        """Quantidade de ferramentas registradas."""
        return len(self._tools)


class ToolRouter:
    """Localiza ferramentas registradas e executa de fato (C2).

    ``route()`` mantém o comportamento do C1 (localizar sem executar).
    ``execute()`` chama ``tool.execute(**kwargs)`` capturando qualquer
    exceção — a ferramenta nunca derruba o chamador — e devolve um
    ``ToolExecutionResult`` padronizado.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None) -> None:
        self.registry = registry or ToolRegistry()

    def find(self, name: str) -> Optional[Tool]:
        """Localiza uma ferramenta pelo nome. Equivale a registry.get()."""
        return self.registry.get(name)

    def route(self, name: str, **kwargs: Any) -> ToolRouterResult:
        """Prepara o encaminhamento para uma ferramenta.

        NÃO executa a ferramenta — apenas verifica se existe e retorna
        o resultado indicando found=True/False.
        """
        tool = self.registry.get(name)
        if tool is None:
            return ToolRouterResult(
                found=False,
                error=f"Ferramenta '{name}' não encontrada no registro.",
            )
        logger.debug("[TOOL_ROUTER] encontrada: %s", name)
        return ToolRouterResult(found=True, tool=tool)

    def execute(
        self, name: str, args: Optional[dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """Executa de fato uma ferramenta registrada (C2).

        ``args`` é um dict com os argumentos da ferramenta (desempacotado
        como ``tool.execute(**args)``). Usar um dict explícito evita colisão
        entre o nome da ferramenta e eventuais parâmetros da própria tool.

        Fluxo:
            1. Localiza a ferramenta reutilizando ``route()`` — ferramenta
               inexistente herda exatamente a mesma mensagem de erro.
            2. Verifica que a ferramenta possui um ``execute`` chamável.
            3. Chama ``tool.execute(**args)`` capturando exceções:
               - ``TypeError`` → argumentos inválidos/faltando;
               - qualquer outra ``Exception`` → erro capturado, sem crash.
            4. Devolve um ``ToolExecutionResult`` padronizado.

        Este método nunca levanta exceção (BaseException como
        KeyboardInterrupt NÃO é capturado, por design).
        """
        routed = self.route(name)
        if not routed.found or routed.tool is None:
            return ToolExecutionResult(
                success=False,
                output=None,
                error=routed.error or f"Ferramenta '{name}' não encontrada no registro.",
                tool_name=name,
            )

        tool = routed.tool
        if tool.execute is None or not callable(tool.execute):
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Ferramenta '{name}' não possui função de execução.",
                tool_name=name,
            )

        try:
            output = tool.execute(**dict(args or {}))
        except TypeError as e:
            logger.warning("[TOOL_ROUTER] argumentos inválidos para '%s': %s", name, e)
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"Argumentos inválidos para '{name}': {e}",
                tool_name=name,
            )
        except Exception as e:  # noqa: BLE001 — captura deliberada e padronizada
            logger.warning("[TOOL_ROUTER] exceção na ferramenta '%s': %s", name, e)
            return ToolExecutionResult(
                success=False,
                output=None,
                error=f"{type(e).__name__}: {e}",
                tool_name=name,
            )

        logger.debug("[TOOL_ROUTER] executada: %s", name)
        return ToolExecutionResult(
            success=True,
            output=output,
            error="",
            tool_name=name,
        )
