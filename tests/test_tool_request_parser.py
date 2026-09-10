"""Testes unitários para o parser de solicitações de ferramenta (C4).

Cobrem os 8 grupos exigidos:
    1. texto sem solicitação → found=False, sem erro;
    2. solicitação válida (com texto antes/depois) → dados estruturados;
    3. JSON malformado → found=True + error, sem exceção;
    4. campo obrigatório faltando → error preenchido;
    5. bloco markdown comum NÃO é confundido com solicitação;
    6. argumentos aninhados/tipos variados;
    7. múltiplas solicitações → apenas a PRIMEIRA (decisão documentada);
    8. nunca lança exceção (None, bytes, vazio, texto gigante, etc.).

O parser é puro: nada de registry, execução, rede ou LLM aqui.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.tool_request_parser import (
    END_MARKER,
    START_MARKER,
    ToolRequestParseResult,
    extract_all_requests,
    parse_tool_request,
)


def _block(tool: str, arguments: str | None = None) -> str:
    inner = f'{{"tool": "{tool}"'
    if arguments is not None:
        inner += f', "arguments": {arguments}'
    inner += "}"
    return f"{START_MARKER}\n{inner}\n{END_MARKER}"


# ---------------------------------------------------------------------------
# 1. Texto sem solicitação → found=False, sem erro
# ---------------------------------------------------------------------------


def test_plain_text_without_request_found_false_no_error():
    result = parse_tool_request("Oi! Hoje o tempo está bom. Quer saber mais?")
    assert isinstance(result, ToolRequestParseResult)
    assert result.found is False
    assert result.error == ""
    assert result.tool_name == ""
    assert result.arguments == {}
    assert result.raw_match == ""


def test_empty_string_found_false():
    result = parse_tool_request("")
    assert result.found is False
    assert result.error == ""


# ---------------------------------------------------------------------------
# 2. Solicitação válida (texto antes/depois)
# ---------------------------------------------------------------------------


def test_valid_request_with_surrounding_text():
    text = (
        "Vou verificar isso para você.\n\n"
        + _block("datetime")
        + "\n\nJá te dou a resposta!"
    )
    result = parse_tool_request(text)
    assert result.found is True
    assert result.error == ""
    assert result.tool_name == "datetime"
    assert result.arguments == {}
    assert '"tool": "datetime"' in result.raw_match


def test_valid_request_with_arguments():
    text = "Claro! " + _block("echo", '{"args": "DaviOS teste"}') + " Pronto."
    result = parse_tool_request(text)
    assert result.found is True
    assert result.error == ""
    assert result.tool_name == "echo"
    assert result.arguments == {"args": "DaviOS teste"}


def test_request_without_arguments_field_defaults_to_empty_dict():
    # Regra 3: 'arguments' é opcional (ferramentas como 'datetime').
    result = parse_tool_request(_block("datetime"))
    assert result.found is True
    assert result.error == ""
    assert result.tool_name == "datetime"
    assert result.arguments == {}


def test_extra_fields_in_json_are_ignored():
    # Regra 4: campos extras além de 'tool'/'arguments' são tolerados.
    text = (
        f"{START_MARKER}\n"
        '{"tool": "echo", "pensamento": "hmm", "arguments": {"args": "oi"}}\n'
        f"{END_MARKER}"
    )
    result = parse_tool_request(text)
    assert result.found is True
    assert result.error == ""
    assert result.tool_name == "echo"
    assert result.arguments == {"args": "oi"}


# ---------------------------------------------------------------------------
# 3. JSON malformado → found=True + error, sem exceção
# ---------------------------------------------------------------------------


def test_malformed_json_recognized_as_attempt_with_error():
    text = (
        "Deixa eu ver...\n"
        f"{START_MARKER}\n"
        '{"tool": "echo", "arguments": {"args": "sem fechar\n'
        f"{END_MARKER}\n"
    )
    result = parse_tool_request(text)
    assert result.found is True  # reconheceu a TENTATIVA
    assert result.error != ""
    assert "JSON" in result.error or "malformado" in result.error
    assert result.tool_name == ""


def test_json_not_an_object_is_error():
    text = f"{START_MARKER}\n[\"uma\", \"lista\"]\n{END_MARKER}"
    result = parse_tool_request(text)
    assert result.found is True
    assert result.error != ""
    assert "objeto" in result.error.lower() or "list" in result.error.lower()


def test_unterminated_block_recognized_with_error():
    text = f"Vou usar uma ferramenta:\n{START_MARKER}\n" + '{"tool": "echo"}'
    result = parse_tool_request(text)
    assert result.found is True
    assert result.error != ""
    assert "fechado" in result.error.lower()


def test_empty_block_recognized_with_error():
    result = parse_tool_request(f"{START_MARKER}\n{END_MARKER}")
    assert result.found is True
    assert result.error != ""


# ---------------------------------------------------------------------------
# 4. Campo obrigatório faltando / tipos errados → error preenchido
# ---------------------------------------------------------------------------


def test_missing_tool_field_is_error():
    text = f"{START_MARKER}\n" + '{"arguments": {"x": 1}}' + f"\n{END_MARKER}"
    result = parse_tool_request(text)
    assert result.found is True
    assert result.error != ""
    assert "tool" in result.error.lower()


def test_empty_tool_name_is_error():
    result = parse_tool_request(f"{START_MARKER}\n" + '{"tool": ""}' + f"\n{END_MARKER}")
    assert result.found is True
    assert result.error != ""


def test_tool_field_wrong_type_is_error():
    result = parse_tool_request(f"{START_MARKER}\n" + '{"tool": 42}' + f"\n{END_MARKER}")
    assert result.found is True
    assert result.error != ""


def test_arguments_wrong_type_is_error():
    text = (
        f"{START_MARKER}\n"
        '{"tool": "echo", "arguments": "texto"}\n'
        f"{END_MARKER}"
    )
    result = parse_tool_request(text)
    assert result.found is True
    assert result.tool_name == "echo"  # nome válido já foi extraído
    assert result.error != ""
    assert "arguments" in result.error.lower()


# ---------------------------------------------------------------------------
# 5. Bloco markdown comum NÃO é confundido com solicitação
# ---------------------------------------------------------------------------


def test_markdown_code_block_is_not_a_tool_request():
    text = (
        "Aqui vai um exemplo de JSON:\n"
        "```json\n"
        '{"tool": "echo", "arguments": {"args": "oi"}}\n'
        "```\n"
        "Espero ter ajudado!"
    )
    result = parse_tool_request(text)
    assert result.found is False
    assert result.error == ""


def test_plain_json_without_markers_is_not_a_request():
    text = 'Resposta: {"tool": "echo", "arguments": {"args": "oi"}}'
    result = parse_tool_request(text)
    assert result.found is False
    assert result.error == ""


# ---------------------------------------------------------------------------
# 6. Argumentos aninhados / tipos variados
# ---------------------------------------------------------------------------


def test_arguments_with_mixed_types_and_nesting():
    arguments = (
        '{"texto": "olá", "numero": 42, "ativo": true, '
        '"lista": [1, "dois", 3], "aninhado": {"chave": "valor", '
        '"profundo": {"n": 7}}}'
    )
    result = parse_tool_request(_block("multi", arguments))
    assert result.found is True
    assert result.error == ""
    assert result.arguments == {
        "texto": "olá",
        "numero": 42,
        "ativo": True,
        "lista": [1, "dois", 3],
        "aninhado": {"chave": "valor", "profundo": {"n": 7}},
    }


def test_arguments_types_are_native_python_after_parse():
    result = parse_tool_request(
        _block("multi", '{"n": 3.5, "flag": false, "nada": null}')
    )
    assert result.found is True
    assert result.arguments == {"n": 3.5, "flag": False, "nada": None}


def test_multiline_json_inside_block_is_parsed():
    text = (
        f"{START_MARKER}\n"
        "{\n"
        '  "tool": "echo",\n'
        '  "arguments": {\n'
        '    "args": "com quebras"\n'
        "  }\n"
        "}\n"
        f"{END_MARKER}"
    )
    result = parse_tool_request(text)
    assert result.found is True
    assert result.error == ""
    assert result.arguments == {"args": "com quebras"}


# ---------------------------------------------------------------------------
# 7. Múltiplas solicitações → DECISÃO: apenas a PRIMEIRA
# ---------------------------------------------------------------------------


def test_multiple_requests_first_one_wins():
    text = (
        "Prefixo.\n"
        + _block("echo", '{"args": "primeira"}')
        + "\nMeio.\n"
        + _block("datetime")
        + "\nSufixo."
    )
    result = parse_tool_request(text)
    assert result.found is True
    assert result.error == ""
    assert result.tool_name == "echo"  # a PRIMEIRA, não a segunda
    assert result.arguments == {"args": "primeira"}


def test_first_request_wins_even_if_it_is_invalid():
    # Regra 2: um primeiro bloco malformado é sinalizado, não pulado.
    text = (
        f"{START_MARKER}\n{{inválido\n{END_MARKER}\n"
        + _block("datetime")
    )
    result = parse_tool_request(text)
    assert result.found is True
    assert result.error != ""
    assert result.tool_name == ""


def test_extract_all_requests_returns_every_block():
    text = _block("echo", '{"args": "a"}') + "\n" + _block("datetime")
    results = extract_all_requests(text)
    assert len(results) == 2
    assert results[0].tool_name == "echo"
    assert results[1].tool_name == "datetime"


# ---------------------------------------------------------------------------
# 8. Nunca lança exceção — entradas absurdas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weird_input",
    [
        None,
        "",
        b"",
        b"\xff\xfe sem utf-8 v\xe1lido",
        b"texto em bytes com " + _block("echo", '{"args": "x"}').encode("utf-8"),
        123,
        4.5,
        True,
        ["lista"],
        {"dicionario": True},
        object(),
    ],
)
def test_never_raises_on_weird_inputs(weird_input):
    result = parse_tool_request(weird_input)
    assert isinstance(result, ToolRequestParseResult)
    if isinstance(weird_input, bytes) and b"TOOL_REQUEST" in weird_input:
        assert result.found is True  # bytes decodificados são parseados
    else:
        assert result.found is False


def test_huge_text_never_raises():
    huge = "palavra " * 200_000  # ~1.6 MB de texto sem solicitação
    result = parse_tool_request(huge)
    assert result.found is False
    assert result.error == ""

    huge_with_block = huge + "\n" + _block("echo", '{"args": "fim"}')
    result2 = parse_tool_request(huge_with_block)
    assert result2.found is True
    assert result2.tool_name == "echo"


def test_marker_case_sensitive_uppercase_only():
    # Marcadores em minúsculo NÃO são reconhecidos (formato é fixo).
    text = "<<<tool_request>>>\n" + '{"tool": "echo"}' + "\n<<<end_tool_request>>>"
    result = parse_tool_request(text)
    assert result.found is False
    assert result.error == ""