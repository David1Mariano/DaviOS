"""Testes unitários para a seção de ferramentas no PromptBuilder (C3).

Cobrem:
    1. build_tools_section com registry vazio/None → string vazia;
    2. build_tools_section com ferramentas → nome, descrição e argumentos;
    3. REGRESSÃO: flag desligada → prompt byte a byte idêntico ao anterior;
    4. flag ligada → prompt contém a seção de ferramentas;
    5. flag ligada + registry vazio → nenhum cabeçalho de ferramentas;
    6. flag tools_visible_to_llm tem padrão False em DaviosConfig.

A seção é PURAMENTE informativa: nenhum protocolo de chamada existe.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.prompt_builder import PromptBuilder
from brain.tool_registry import Tool, ToolRegistry
from brain.tool_request_parser import (
    END_MARKER,
    START_MARKER,
    extract_all_requests,
    parse_tool_request,
)
from config.davios_config import DaviosConfig
from personality.personality import DEFAULT_PERSONALITY


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        Tool(
            name="echo",
            description="Repete o texto informado.",
            arguments=["args"],
            execute=lambda args: args,
        )
    )
    reg.register(
        Tool(
            name="web_fetch",
            description="Busca e sanitiza uma pagina web.",
            arguments=["url"],
        )
    )
    reg.register(
        Tool(
            name="datetime",
            description="Mostra a data atual do sistema.",
        )
    )
    return reg


def _builder(flag: bool) -> PromptBuilder:
    config = DaviosConfig()
    config.tools_visible_to_llm = flag
    return PromptBuilder(config, DEFAULT_PERSONALITY)


# ---------------------------------------------------------------------------
# 1. Registry vazio / None → string vazia
# ---------------------------------------------------------------------------


def test_build_tools_section_empty_registry_returns_empty():
    builder = _builder(flag=True)
    assert builder.build_tools_section(ToolRegistry()) == ""


def test_build_tools_section_none_registry_returns_empty():
    builder = _builder(flag=True)
    assert builder.build_tools_section(None) == ""


# ---------------------------------------------------------------------------
# 2. Registry com ferramentas → nome, descrição e argumentos
# ---------------------------------------------------------------------------


def test_build_tools_section_contains_name_description_arguments():
    builder = _builder(flag=True)
    section = builder.build_tools_section(_make_registry())
    assert section.startswith("### FERRAMENTAS DISPONIVEIS")
    assert "- echo: Repete o texto informado. (argumentos: args)" in section
    assert "- web_fetch: Busca e sanitiza uma pagina web. (argumentos: url)" in section
    assert "- datetime: Mostra a data atual do sistema. (sem argumentos)" in section


def test_build_tools_section_multiple_arguments_comma_separated():
    builder = _builder(flag=True)
    reg = ToolRegistry()
    reg.register(
        Tool(name="multi", description="Ferramenta com dois argumentos.",
             arguments=["primeiro", "segundo"])
    )
    section = builder.build_tools_section(reg)
    assert "- multi: Ferramenta com dois argumentos. (argumentos: primeiro, segundo)" in section


def test_build_tools_section_is_informative_only_no_call_syntax():
    # C4b: a seção agora TAMBÉM documenta o protocolo de solicitação.
    # A LISTAGEM continua sendo apenas listagem; as INSTRUÇÕES existem
    # em bloco próprio, com os marcadores exatos do parser (fonte única).
    builder = _builder(flag=True)
    section = builder.build_tools_section(_make_registry())

    listing_lines = [l for l in section.splitlines() if l.startswith("- ")]
    assert listing_lines
    for line in listing_lines:
        assert "chame" not in line.lower()
        assert "json" not in line.lower()

    # Instruções do protocolo (C4b) presentes, com marcadores do parser:
    assert START_MARKER in section
    assert END_MARKER in section
    assert "Regras:" in section


# ---------------------------------------------------------------------------
# 3. REGRESSÃO: flag desligada → prompt byte a byte idêntico ao anterior
# ---------------------------------------------------------------------------


def test_prompt_flag_off_is_byte_identical_to_previous_behavior():
    config = DaviosConfig()
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)

    class _Ctx:
        messages = [{"user": "qual meu nome?", "response": "Davi"}]
        current_topic = "preferencias"

    built = builder.build(
        "Oi",
        context=_Ctx(),
        memories=[{"target": "nome", "value": "Davi"}],
        intent="conversation",
        resolved_context="voce perguntou sobre minha idade",
    )

    expected_prompt = (
        "Informacoes que voce sabe sobre o usuario:\n"
        "- Nome do usuario: Davi"
        "\n\n"
        "Conversa recente:\n"
        "Usuario: qual meu nome?\n"
        "DaviOS: Davi"
        "\n\n"
        "Assunto atual da conversa: preferencias"
        "\n\n"
        "Referencia a mensagem anterior:\nvoce perguntou sobre minha idade"
        "\n\n"
        "Mensagem do usuario: Oi"
    )
    assert built.prompt == expected_prompt

    expected_system = (
        config.system_prompt.strip()
        + "\n\n"
        + DEFAULT_PERSONALITY.to_system_text()
    )
    assert built.system == expected_system

    # Nenhum vestígio da seção nova em lugar nenhum:
    assert "FERRAMENTAS" not in built.prompt
    assert "FERRAMENTAS" not in built.system
    assert "tools_in_prompt" not in built.metadata


# ---------------------------------------------------------------------------
# 4. Flag ligada → prompt contém a seção de ferramentas
# ---------------------------------------------------------------------------


def test_prompt_flag_on_includes_tools_section():
    builder = _builder(flag=True)
    built = builder.build("Oi", tools_registry=_make_registry())
    assert "### FERRAMENTAS DISPONIVEIS" in built.prompt
    assert "- echo: Repete o texto informado. (argumentos: args)" in built.prompt
    assert "- web_fetch: Busca e sanitiza uma pagina web. (argumentos: url)" in built.prompt
    # A seção vem ANTES da mensagem do usuario (última seção do prompt):
    assert (
        built.prompt.index("### FERRAMENTAS DISPONIVEIS")
        < built.prompt.index("Mensagem do usuario: Oi")
    )
    assert built.metadata.get("tools_in_prompt") is True
    assert built.metadata.get("tools_count") == 3


# ---------------------------------------------------------------------------
# 5. Flag ligada + registry vazio → nenhum cabeçalho de ferramentas
# ---------------------------------------------------------------------------


def test_prompt_flag_on_empty_or_none_registry_has_no_header():
    builder = _builder(flag=True)
    for built in (
        builder.build("Oi", tools_registry=None),
        builder.build("Oi", tools_registry=ToolRegistry()),
    ):
        assert "FERRAMENTAS" not in built.prompt
        # C4b: sem registry, as instruções do protocolo também não aparecem:
        assert START_MARKER not in built.prompt
        assert END_MARKER not in built.prompt
        assert built.prompt == "Mensagem do usuario: Oi"
        assert "tools_in_prompt" not in built.metadata


def test_broken_registry_never_breaks_build():
    class _Broken:
        def list_tools(self):
            raise RuntimeError("boom")

    builder = _builder(flag=True)
    built = builder.build("Oi", tools_registry=_Broken())
    assert built.prompt == "Mensagem do usuario: Oi"


# ---------------------------------------------------------------------------
# 6. Flag com padrão False em DaviosConfig
# ---------------------------------------------------------------------------


def test_tools_visible_to_llm_defaults_off():
    config = DaviosConfig()
    assert config.tools_visible_to_llm is False
    # "saber que existe" é separado de "pode executar":
    assert config.actions_enabled is False
    assert config.web_enabled is False


# ===========================================================================
# C4b — Instruções do protocolo de solicitação no prompt
# ===========================================================================


def test_prompt_flag_on_contains_protocol_instructions_with_parser_markers():
    builder = _builder(flag=True)
    built = builder.build("Oi", tools_registry=_make_registry())
    # Marcadores importados de tool_request_parser (fonte única de verdade):
    assert START_MARKER in built.prompt
    assert END_MARKER in built.prompt
    # (b) apenas UM pedido por resposta — decisão do C4 refletida:
    assert "UM pedido por resposta" in built.prompt
    assert "apenas o primeiro" in built.prompt
    # (c) bloco contém apenas o JSON:
    assert "apenas o JSON" in built.prompt
    assert "sem comentarios" in built.prompt


def test_instructions_come_after_listing_and_before_user_message():
    builder = _builder(flag=True)
    built = builder.build("Oi", tools_registry=_make_registry())
    listing_pos = built.prompt.index("- echo: Repete o texto informado.")
    instructions_pos = built.prompt.index("Para usar uma ferramenta")
    message_pos = built.prompt.index("Mensagem do usuario: Oi")
    assert listing_pos < instructions_pos < message_pos


def test_example_block_uses_tool_from_registry_not_hardcoded():
    builder = _builder(flag=True)
    reg = _make_registry()
    section = builder.build_tools_section(reg)
    blocks = extract_all_requests(section)
    # 1º bloco = template genérico; 2º bloco = exemplo com ferramenta real:
    assert len(blocks) == 2
    template, example = blocks
    assert template.error == ""
    assert example.error == ""
    assert example.tool_name in {t.name for t in reg.list_tools()}
    # A ferramenta do exemplo é a primeira COM argumentos declarados (echo):
    assert example.tool_name == "echo"
    assert example.arguments == {"args": "<valor>"}


def test_example_block_without_arguments_tools_uses_first_tool():
    reg = ToolRegistry()
    reg.register(Tool(name="datetime", description="Mostra a data atual."))
    section = _builder(flag=True).build_tools_section(reg)
    blocks = extract_all_requests(section)
    assert len(blocks) == 2
    assert blocks[1].error == ""
    assert blocks[1].tool_name == "datetime"
    assert blocks[1].arguments == {}


def test_integration_instructions_and_parser_stay_in_sync():
    """O exemplo formatado EXATAMENTE como o prompt instrui é reconhecido
    pelo parser da C4 — instruções e parser não divergiram."""
    builder = _builder(flag=True)
    reg = _make_registry()
    built = builder.build("Oi", tools_registry=reg)

    # Todo bloco presente no prompt é reconhecido sem erro pelo parser:
    blocks = extract_all_requests(built.prompt)
    assert len(blocks) >= 2
    for block in blocks:
        assert block.found is True
        assert block.error == ""

    # Um pedido real, formatado exatamente como o prompt instrui:
    follow_up = parse_tool_request(
        f"{START_MARKER}\n"
        '{"tool": "echo", "arguments": {"args": "teste real"}}\n'
        f"{END_MARKER}"
    )
    assert follow_up.found is True
    assert follow_up.error == ""
    assert follow_up.tool_name == "echo"
    assert follow_up.arguments == {"args": "teste real"}
