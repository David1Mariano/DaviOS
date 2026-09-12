"""Testes unitários para execução real de ferramentas (C2).

Cobrem:
    1. ToolRouter.execute com sucesso (resultado correto);
    2. exceção da ferramenta capturada sem crash;
    3. ferramenta inexistente → erro padronizado (reaproveita route());
    4. argumentos faltando/errados → erro padronizado;
    5. adapters do ActionManager (nome, descrição, execução real);
    6. adapters do WebAccess (mesma cobertura);
    7. register_default_tools popula sem duplicar nomes.

Nenhuma conexão com Qwen/ConversationEngine/CognitiveCore/PromptBuilder —
isso é C3+.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.action_manager import ActionManager
from brain.tool_adapters import (
    make_action_manager_tool,
    make_web_fetch_tool,
    register_action_manager_tools,
    register_default_tools,
    register_web_access_tools,
)
from brain.tool_registry import (
    Tool,
    ToolError,
    ToolExecutionResult,
    ToolRegistry,
    ToolRouter,
)
from brain.web_access import WebAccess, WebResult


def _sample_tool(name: str = "double", description: str = "Dobra um número") -> Tool:
    return Tool(
        name=name,
        description=description,
        arguments=["n"],
        execute=lambda n: n * 2,
    )


# ---------------------------------------------------------------------------
# 1. Execução com sucesso
# ---------------------------------------------------------------------------


def test_execute_registered_tool_success():
    reg = ToolRegistry()
    reg.register(_sample_tool())
    router = ToolRouter(registry=reg)
    result = router.execute("double", {"n": 21})
    assert isinstance(result, ToolExecutionResult)
    assert result.success is True
    assert result.output == 42
    assert result.error == ""
    assert result.tool_name == "double"


def test_execute_string_tool_success():
    reg = ToolRegistry()
    reg.register(
        Tool(name="greet", description="Saudação", arguments=["name"],
             execute=lambda name: f"Olá, {name}!")
    )
    router = ToolRouter(registry=reg)
    result = router.execute("greet", {"name": "Davi"})
    assert result.success is True
    assert result.output == "Olá, Davi!"


# ---------------------------------------------------------------------------
# 2. Exceção da ferramenta capturada (sem crash)
# ---------------------------------------------------------------------------


def test_execute_tool_raising_exception_is_captured():
    reg = ToolRegistry()

    def _boom():
        raise ValueError("falha interna simulada")

    reg.register(Tool(name="boom", description="Explode", execute=_boom))
    router = ToolRouter(registry=reg)
    result = router.execute("boom")
    assert result.success is False
    assert result.output is None
    assert "ValueError" in result.error
    assert "falha interna simulada" in result.error
    assert result.tool_name == "boom"


def test_execute_never_raises_even_with_weird_exception():
    class StrangeError(Exception):
        """Exceção incomum, mas ainda assim uma Exception."""

    reg = ToolRegistry()
    reg.register(
        Tool(name="weird", description="Levanta exceção incomum",
             execute=lambda: (_ for _ in ()).throw(StrangeError("hou")))
    )
    router = ToolRouter(registry=reg)
    result = router.execute("weird")  # não deve levantar
    assert result.success is False
    assert "StrangeError" in result.error


def test_execute_tool_without_execute_callable():
    reg = ToolRegistry()
    reg.register(Tool(name="metadata", description="Só metadado"))
    router = ToolRouter(registry=reg)
    result = router.execute("metadata")
    assert result.success is False
    assert "não possui função de execução" in result.error
    assert result.tool_name == "metadata"


# ---------------------------------------------------------------------------
# 3. Ferramenta inexistente (mesmo comportamento do route(), reaproveitado)
# ---------------------------------------------------------------------------


def test_execute_nonexistent_tool_standardized_error():
    router = ToolRouter()
    result = router.execute("nonexistent", {"x": 1})
    assert isinstance(result, ToolExecutionResult)
    assert result.success is False
    assert result.output is None
    assert "não encontrada no registro" in result.error
    assert result.tool_name == "nonexistent"


def test_execute_nonexistent_reuses_route_message():
    router = ToolRouter()
    routed = router.route("falta")
    executed = router.execute("falta")
    assert routed.found is False
    assert executed.error == routed.error


# ---------------------------------------------------------------------------
# 4. Argumentos inválidos/faltando
# ---------------------------------------------------------------------------


def test_execute_missing_arguments_standardized_error():
    reg = ToolRegistry()
    reg.register(_sample_tool())  # requer 'n'
    router = ToolRouter(registry=reg)
    result = router.execute("double")
    assert result.success is False
    assert result.output is None
    assert "Argumentos inválidos" in result.error
    assert "n" in result.error  # menciona o argumento faltando
    assert result.tool_name == "double"


def test_execute_wrong_argument_name_standardized_error():
    reg = ToolRegistry()
    reg.register(_sample_tool())
    router = ToolRouter(registry=reg)
    result = router.execute("double", {"numero": 10})  # nome errado
    assert result.success is False
    assert "Argumentos inválidos" in result.error


def test_execute_unexpected_extra_argument_standardized_error():
    reg = ToolRegistry()
    reg.register(Tool(name="noop", description="Nada", execute=lambda: "ok"))
    router = ToolRouter(registry=reg)
    result = router.execute("noop", {"extra": "inútil"})
    assert result.success is False
    assert "Argumentos inválidos" in result.error


# ---------------------------------------------------------------------------
# 5. Adapters do ActionManager
# ---------------------------------------------------------------------------


@pytest.fixture()
def am_registry():
    """Registry com todos os comandos reais do ActionManager habilitado."""
    am = ActionManager(enabled=True)
    reg = ToolRegistry()
    register_action_manager_tools(reg, am)
    return reg, am


def test_action_manager_adapters_registered(am_registry):
    reg, am = am_registry
    expected = {cmd["name"] for cmd in am.list_commands()}
    assert expected == {"datetime", "time", "echo", "list_dir", "read_file"}
    for cmd in am.list_commands():
        assert reg.has(cmd["name"])
        tool = reg.get(cmd["name"])
        assert tool is not None
        assert tool.name == cmd["name"]
        assert tool.description == cmd["description"]  # descrição real preservada
        assert tool.execute is not None
        assert tool.arguments == ["args"]


def test_action_manager_adapter_real_execution_echo(am_registry):
    reg, _ = am_registry
    router = ToolRouter(registry=reg)
    result = router.execute("echo", {"args": "DaviOS teste real"})
    assert result.success is True
    assert result.error == ""
    assert "DaviOS teste real" in str(result.output)


def test_action_manager_adapter_real_execution_datetime(am_registry):
    reg, _ = am_registry
    router = ToolRouter(registry=reg)
    result = router.execute("datetime")
    assert result.success is True
    assert str(result.output).strip() != ""


def test_action_manager_adapter_failure_becomes_standard_error():
    am = ActionManager(enabled=False)  # desabilitado → ok=False no execute real
    reg = ToolRegistry()
    register_action_manager_tools(reg, am)
    router = ToolRouter(registry=reg)
    result = router.execute("echo", {"args": "oi"})
    assert result.success is False
    assert result.output is None
    assert result.error != ""  # mensagem padronizada do ActionManager


def test_action_manager_adapter_invalid_args_standard_error(am_registry):
    reg, _ = am_registry
    router = ToolRouter(registry=reg)
    # echo não aceita caracteres fora do padrão (ex.: pipe)
    result = router.execute("echo", {"args": "cat file | rm -rf"})
    assert result.success is False
    assert result.error != ""


def test_make_action_manager_tool_rejects_unknown_command():
    am = ActionManager(enabled=True)
    with pytest.raises(ToolError):
        make_action_manager_tool(am, "comando_inexistente")


# ---------------------------------------------------------------------------
# 6. Adapters do WebAccess
# ---------------------------------------------------------------------------


def test_web_access_adapter_registered():
    wa = WebAccess(enabled=False)
    reg = ToolRegistry()
    register_web_access_tools(reg, wa)
    assert reg.has("web_fetch")
    tool = reg.get("web_fetch")
    assert tool is not None
    assert tool.name == "web_fetch"
    assert tool.description
    assert tool.execute is not None
    assert tool.arguments == ["url"]


def test_web_access_adapter_real_execution_disabled_standard_error():
    wa = WebAccess(enabled=False)  # padrão do DaviOS: web desativada
    reg = ToolRegistry()
    register_web_access_tools(reg, wa)
    router = ToolRouter(registry=reg)
    result = router.execute("web_fetch", {"url": "http://exemplo.com"})
    assert result.success is False
    assert result.output is None
    assert result.error != ""  # mensagem real do WebAccess (web desabilitada)


def test_web_access_adapter_success_path_returns_dict():
    class _StubWeb:
        def fetch(self, url: str) -> WebResult:
            return WebResult(
                ok=True, url=url, text="conteúdo sanitizado", title="Título",
                status=200,
            )

    reg = ToolRegistry()
    register_web_access_tools(reg, _StubWeb())  # duck-typing: só usa fetch()
    router = ToolRouter(registry=reg)
    result = router.execute("web_fetch", {"url": "http://exemplo.com"})
    assert result.success is True
    assert isinstance(result.output, dict)
    assert result.output["ok"] is True
    assert result.output["text"] == "conteúdo sanitizado"
    assert result.output["title"] == "Título"
    assert result.output["status"] == 200


def test_web_access_adapter_fetch_failure_becomes_standard_error():
    class _StubWeb:
        def fetch(self, url: str) -> WebResult:
            return WebResult(ok=False, url=url, error="HTTP 404")

    reg = ToolRegistry()
    register_web_access_tools(reg, _StubWeb())
    router = ToolRouter(registry=reg)
    result = router.execute("web_fetch", {"url": "http://exemplo.com/404"})
    assert result.success is False
    assert "HTTP 404" in result.error


# ---------------------------------------------------------------------------
# 7. register_default_tools — popula sem duplicar nomes
# ---------------------------------------------------------------------------


def test_register_default_tools_populates_registry():
    reg = ToolRegistry()
    registered = register_default_tools(reg)
    assert len(registered) > 0
    assert reg.count() == len(registered)
    names = [t.name for t in reg.list_tools()]
    assert len(names) == len(set(names))  # sem duplicatas internas
    for expected in ("datetime", "time", "echo", "list_dir", "web_fetch"):
        assert expected in names


def test_register_default_tools_idempotent_no_duplicates():
    reg = ToolRegistry()
    first = register_default_tools(reg)
    count_after_first = reg.count()
    second = register_default_tools(reg)  # 2ª chamada NÃO deve duplicar
    assert second == []  # nada novo registrado na 2ª chamada
    assert reg.count() == count_after_first
    names = [t.name for t in reg.list_tools()]
    assert len(names) == len(set(names))
    assert sorted(names) == sorted(first)  # mesmos nomes, nenhuma duplicação


def test_register_default_tools_with_explicit_instances():
    am = ActionManager(enabled=True)
    wa = WebAccess(enabled=False)
    reg = ToolRegistry()
    register_default_tools(reg, action_manager=am, web_access=wa)
    for name in ("datetime", "time", "echo", "list_dir", "web_fetch"):
        assert reg.has(name)
