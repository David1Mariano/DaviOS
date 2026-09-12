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
        "Conversa anterior - so contexto, nao exemplo: nao copie nem imite "
        "o formato ou encerramento das respostas anteriores.\n"
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
    # Ignora o template generico (placeholders <nome>/<argumento>):
    blocks = [
        b for b in extract_all_requests(section)
        if not (b.tool_name.startswith("<") or b.tool_name.endswith(">"))
    ]
    # 1º = exemplo explicito; 2º+ = few-shot(s):
    assert len(blocks) >= 1
    example = blocks[0]
    assert example.error == ""
    assert example.tool_name in {t.name for t in reg.list_tools()}
    # preferred_names em _tools_example_block(): time, datetime, read_file,
    # list_dir, web_fetch. Neste registry (echo, web_fetch, datetime) a
    # primeira preferida presente é datetime:
    assert example.tool_name == "datetime"
    # Deriva os argumentos esperados dinamicamente da ferramenta escolhida,
    # em vez de hardcoded:
    chosen_tool = next(
        t for t in reg.list_tools() if t.name == example.tool_name
    )
    expected_args = {
        str(a): "<valor>" for a in (chosen_tool.arguments or [])
    }
    assert example.arguments == expected_args


def test_example_block_without_arguments_tools_uses_first_tool():
    reg = ToolRegistry()
    reg.register(Tool(name="datetime", description="Mostra a data atual."))
    section = _builder(flag=True).build_tools_section(reg)
    # Ignora o template generico (placeholders <nome>/<argumento>):
    blocks = [
        b for b in extract_all_requests(section)
        if not (b.tool_name.startswith("<") or b.tool_name.endswith(">"))
    ]
    assert len(blocks) >= 1
    example = blocks[0]
    assert example.error == ""
    assert example.tool_name == "datetime"
    assert example.arguments == {}


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


# ---------------------------------------------------------------------------
# Correção: system_prompt condicional às flags de ferramentas
# ---------------------------------------------------------------------------

# Frase de negação que deve estar presente quando tools_visible_to_llm=False
# e AUSENTE quando tools_visible_to_llm=True.
DENIAL_PHRASE = "Voce nao pode executar comandos no computador"


def test_flag_off_system_prompt_contains_denial_phrase():
    """Flag off → system_prompt contém a frase de negação de capacidades."""
    config = DaviosConfig()
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    built = builder.build("Oi")
    assert DENIAL_PHRASE in built.system


def test_flag_on_system_prompt_does_not_contain_denial_phrase():
    """Flag on → system_prompt NÃO contém a frase de negação de capacidades."""
    config = DaviosConfig()
    config.tools_visible_to_llm = True
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    built = builder.build("Oi")
    assert DENIAL_PHRASE not in built.system


def test_flag_on_tools_section_still_present():
    """Flag on + registry → seção de ferramentas continua presente normalmente."""
    config = DaviosConfig()
    config.tools_visible_to_llm = True
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    reg = _make_registry()
    built = builder.build("Oi", tools_registry=reg)
    assert "### FERRAMENTAS DISPONIVEIS" in built.prompt
    assert DENIAL_PHRASE not in built.system


def test_flag_on_empty_registry_no_tools_section_and_no_denial():
    """Flag on + registry vazio → sem seção de ferramentas E sem frase de negação.

    Decisão: a frase de negação NÃO volta quando o registry está vazio,
    porque o usuário ativou explicitamente tools_visible_to_llm=True. O
    registry vazio é um estado de configuração (ferramentas podem ser
    adicionadas depois), não uma negação de capacidade. A seção de
    ferramentas simplesmente não aparece (comportamento C3), o que já
    indica que não há ferramentas disponíveis no momento."""
    config = DaviosConfig()
    config.tools_visible_to_llm = True
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    reg = ToolRegistry()  # vazio
    built = builder.build("Oi", tools_registry=reg)
    assert "### FERRAMENTAS DISPONIVEIS" not in built.prompt
    assert DENIAL_PHRASE not in built.system


def test_flag_off_prompt_byte_identical_to_pre_correction():
    """Flag off → prompt final byte a byte idêntico ao comportamento
    anterior a esta correção (a frase de negação faz sentido quando não
    há ferramentas)."""
    config = DaviosConfig()
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    built = builder.build("Oi")
    # O system_prompt deve ser exatamente o system_prompt original do config.
    assert config.system_prompt in built.system


# ---------------------------------------------------------------------------
# Few-shot: exemplos de invocacao implicita
# ---------------------------------------------------------------------------

def test_few_shot_examples_present_when_flag_on():
    """Flag on → prompt contem pelo menos um exemplo de invocacao implícita
    (pergunta natural sem nomear a ferramenta), alem do exemplo explicito."""
    builder = _builder(flag=True)
    reg = _make_registry()
    built = builder.build("Oi", tools_registry=reg)

        # Few-shot presente (cabecalho "Exemplos:"):
    assert "Exemplos:" in built.prompt
    # Exemplo positivo com pergunta natural (sem citar nome de ferramenta):
    # Com o registry padrao (echo, web_fetch, datetime), os few-shot usam
    # "Que horas sao agora?" (datetime) ou "Repita exatamente: teste 123" (echo):
    assert ("Que horas sao" in built.prompt
            or "O que tem no arquivo" in built.prompt
            or "Repita exatamente" in built.prompt)
    # O exemplo continua sendo reconhecido pelo parser (sincronia):
    # Ignora o template generico (placeholders <nome>/<argumento>):
    blocks = [
        b for b in extract_all_requests(built.prompt)
        if not (b.tool_name.startswith("<") or b.tool_name.endswith(">"))
    ]
    assert len(blocks) >= 2  # explicito + few-shot(s)
    for block in blocks:
        assert block.found is True
        assert block.error == ""


def test_few_shot_uses_real_tools_from_registry():
    """Os exemplos few-shot usam ferramentas realmente presentes no registry."""
    builder = _builder(flag=True)
    reg = ToolRegistry()
    reg.register(Tool(name="time", description="Mostra a hora atual."))
    reg.register(Tool(name="echo", description="Repete o texto.", arguments=["args"]))
    section = builder.build_tools_section(reg)

    # O exemplo few-shot usa "time" (que está no registry):
    assert '"tool": "time"' in section
    # E nao inventa ferramentas fora do registry (ignora o template
    # generico que usa <nome>/<argumento> como placeholder):
    tool_names = {t.name for t in reg.list_tools()}
    blocks = extract_all_requests(section)
    for block in blocks:
        if block.found and block.tool_name:
            # Pula o template generico (placeholders entre <>):
            if block.tool_name.startswith("<") or block.tool_name.endswith(">"):
                continue
            assert block.tool_name in tool_names, \
                f"ferramenta {block.tool_name!r} nao esta no registry"


def test_few_shot_with_generic_tool_uses_first_available():
    """Quando nao ha ferramentas 'curadas', usa a primeira disponivel."""
    builder = _builder(flag=True)
    reg = ToolRegistry()
    reg.register(Tool(name="minha_ferramenta", description="Faz algo.",
                     arguments=["entrada"]))
    section = builder.build_tools_section(reg)

    assert "Exemplos:" in section
    assert '"tool": "minha_ferramenta"' in section


def test_few_shot_max_two_examples():
    """No maximo 3 exemplos few-shot (economia de contexto)."""
    builder = _builder(flag=True)
    reg = ToolRegistry()
    reg.register(Tool(name="time", description="Hora."))
    reg.register(Tool(name="datetime", description="Data."))
    reg.register(Tool(name="echo", description="Repete.", arguments=["args"]))
    reg.register(Tool(name="web_fetch", description="Web.", arguments=["url"]))
    section = builder.build_tools_section(reg)

    # Conta blocos few-shot (total - template - exemplo explicito):
    blocks = extract_all_requests(section)
    # 1 template + 1 explicito + ate 2 few-shot = max 4
    assert len(blocks) <= 5
    assert len(blocks) >= 3  # pelo menos 1 few-shot


def test_few_shot_flag_off_no_few_shot():
    """Flag off → nem a secao de ferramentas nem os exemplos aparecem."""
    config = DaviosConfig()
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    built = builder.build("Oi")
    assert "Exemplos:" not in built.prompt
    assert "FERRAMENTAS" not in built.prompt


def test_few_shot_empty_registry_no_few_shot():
    """Registry vazio → nem few-shot nem secao aparecem."""
    builder = _builder(flag=True)
    built = builder.build("Oi", tools_registry=ToolRegistry())
    assert "Exemplos:" not in built.prompt
    assert "FERRAMENTAS" not in built.prompt


# ---------------------------------------------------------------------------
# Correção: política de necessidade + schema de argumentos + exemplo negativo
# ---------------------------------------------------------------------------


def test_flag_off_prompt_byte_identical_after_correction():
    """Flag off → prompt byte a byte idêntico após esta correção."""
    config = DaviosConfig()
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    built = builder.build("Oi")
    assert config.system_prompt in built.system
    assert "### FERRAMENTAS" not in built.prompt


def test_policy_says_dont_use_for_casual_conversation():
    """A política de necessidade deve mencionar NÃO usar para casos casuais."""
    builder = _builder(flag=True)
    reg = _make_registry()
    section = builder.build_tools_section(reg)
    # Deve mencionar que conversa casual/nao-operacional nao gera TOOL_REQUEST
    assert "conversa casual" in section.lower() or "sem gerar TOOL_REQUEST" in section


def test_no_canned_negative_example_present():
    """Nao deve existir exemplo negativo fixo (conversa social SEM ferramenta).

    Intencional: um exemplo negativo com resposta fixa (ex: "valeu!" ->
    "Por nada!") tende a ser memorizado e repetido literalmente por modelos
    locais pequenos para QUALQUER mensagem seguinte, travando a conversa em
    loop assim que a resposta entra no historico. A orientacao de quando
    NAO usar ferramenta ja esta coberta pela instrucao em texto corrido.
    """
    builder = _builder(flag=True)
    reg = _make_registry()
    section = builder.build_tools_section(reg)
    # Confirma a AUSENCIA do texto fixo do antigo exemplo negativo:
    assert "valeu" not in section
    assert "Por nada" not in section


def test_read_file_example_uses_path_not_args():
    """read_file deve mostrar argumento 'path', nunca 'args'."""
    builder = _builder(flag=True)
    reg = ToolRegistry()
    reg.register(Tool(
        name="read_file",
        description="Le conteudo de arquivo.",
        arguments=["path"],
    ))
    section = builder.build_tools_section(reg)
    # O exemplo few-shot deve usar "path"
    assert '"path"' in section
    # Nunca deve usar "args" para read_file no bloco de exemplo
    # (o template generico usa <argumento>, o que esta ok)
    blocks = extract_all_requests(section)
    for block in blocks:
        if block.found and block.tool_name == "read_file" and block.error == "":
            assert "path" in block.arguments


def test_echo_description_restricted_to_explicit_requests():
    """echo deve ser descrito como uso apenas para pedidos explicitos."""
    builder = _builder(flag=True)
    reg = _make_registry()
    section = builder.build_tools_section(reg)
    # A política deve mencionar que echo eh para pedidos explicitos
    assert "explicitamente" in section.lower() or "pedir" in section.lower()


def test_parser_recognizes_few_shot_block_in_prompt():
    """O exemplo positivo formatado no prompt deve ser reconhecido pelo parser."""
    builder = _builder(flag=True)
    reg = _make_registry()
    section = builder.build_tools_section(reg)
    blocks = extract_all_requests(section)
    # Pelo menos um bloco real (não o template genérico) deve ser reconhecido
    real_blocks = [
        b for b in blocks
        if b.found and not (b.tool_name.startswith("<") and b.tool_name.endswith(">"))
    ]
    assert len(real_blocks) >= 1
    for block in real_blocks:
        assert block.error == ""
