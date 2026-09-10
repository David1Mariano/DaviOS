"""Testes unitários para ToolRegistry e ToolRouter (C1).

Cobrem apenas registro, localização e listagem.
Nenhuma ferramenta é executada — testes validam metadados e erros.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.tool_registry import (
    DuplicateToolError,
    InvalidToolError,
    Tool,
    ToolNotFoundError,
    ToolRegistry,
    ToolRouter,
    ToolRouterResult,
)


def _sample_tool(name: str = "echo", description: str = "Repete texto") -> Tool:
    return Tool(
        name=name,
        description=description,
        arguments=["text"],
        execute=lambda text: text,
    )


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


def test_register_tool():
    reg = ToolRegistry()
    tool = _sample_tool()
    reg.register(tool)
    assert reg.has("echo")
    assert reg.count() == 1


def test_get_tool():
    reg = ToolRegistry()
    tool = _sample_tool()
    reg.register(tool)
    result = reg.get("echo")
    assert result is not None
    assert result.name == "echo"
    assert result.description == "Repete texto"
    assert result.arguments == ["text"]
    assert result.execute is not None


def test_has_tool_exists_and_not_exists():
    reg = ToolRegistry()
    reg.register(_sample_tool())
    assert reg.has("echo") is True
    assert reg.has("nonexistent") is False


def test_list_tools():
    reg = ToolRegistry()
    t1 = _sample_tool("echo", "Repete texto")
    t2 = _sample_tool("datetime", "Retorna data/hora")
    reg.register(t1)
    reg.register(t2)
    tools = reg.list_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {"echo", "datetime"}


def test_reject_duplicate_name():
    reg = ToolRegistry()
    reg.register(_sample_tool("echo"))
    with pytest.raises(DuplicateToolError):
        reg.register(_sample_tool("echo", "Outra descrição"))


def test_reject_invalid_tool():
    reg = ToolRegistry()
    with pytest.raises(InvalidToolError):
        reg.register("não é uma Tool")  # type: ignore[arg-type]


def test_reject_tool_without_name():
    reg = ToolRegistry()
    with pytest.raises(InvalidToolError):
        reg.register(Tool(name="", description="sem nome"))


def test_reject_tool_without_description():
    reg = ToolRegistry()
    with pytest.raises(InvalidToolError):
        reg.register(Tool(name="blank", description=""))


def test_get_nonexistent_returns_none():
    reg = ToolRegistry()
    assert reg.get("nonexistent") is None


def test_list_tools_defensive_copy():
    reg = ToolRegistry()
    reg.register(_sample_tool())
    tools = reg.list_tools()
    tools.clear()
    assert reg.count() == 1


# ---------------------------------------------------------------------------
# ToolRouter
# ---------------------------------------------------------------------------


def test_router_find_existing_tool():
    reg = ToolRegistry()
    tool = _sample_tool()
    reg.register(tool)
    router = ToolRouter(registry=reg)
    result = router.find("echo")
    assert result is not None
    assert result.name == "echo"


def test_router_find_nonexistent_returns_none():
    router = ToolRouter()
    assert router.find("nonexistent") is None


def test_router_route_existing_tool():
    reg = ToolRegistry()
    reg.register(_sample_tool())
    router = ToolRouter(registry=reg)
    result = router.route("echo", text="oi")
    assert isinstance(result, ToolRouterResult)
    assert result.found is True
    assert result.tool is not None
    assert result.tool.name == "echo"
    assert result.error == ""


def test_router_route_nonexistent_tool():
    router = ToolRouter()
    result = router.route("nonexistent")
    assert isinstance(result, ToolRouterResult)
    assert result.found is False
    assert result.tool is None
    assert "não encontrada" in result.error


def test_router_uses_default_registry_when_none_provided():
    router = ToolRouter()
    assert router.registry is not None
    assert router.registry.count() == 0


def test_tool_to_dict():
    tool = _sample_tool("echo", "Repete texto")
    d = tool.to_dict()
    assert d == {
        "name": "echo",
        "description": "Repete texto",
        "arguments": ["text"],
        "has_execute": True,
    }


def test_tool_to_dict_without_execute():
    tool = Tool(name="dummy", description="só metadado")
    d = tool.to_dict()
    assert d["has_execute"] is False
    assert d["arguments"] == []
