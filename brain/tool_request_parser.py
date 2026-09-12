"""Parser de solicitações de ferramenta do DaviOS (C4).

Define e reconhece o formato pelo qual o LLM (Qwen) pode expressar
"quero usar a ferramenta X com estes argumentos" dentro do texto livre
que ele gera:

    <<<TOOL_REQUEST>>>
    {"tool": "nome_da_ferramenta", "arguments": {"arg1": "valor1"}}
    <<<END_TOOL_REQUEST>>>

ESCOLHA DO FORMATO (e por quê):
    - Marcadores ASCII únicos e improváveis (``<<<TOOL_REQUEST>>>`` /
      ``<<<END_TOOL_REQUEST>>>``) — NÃO colidem com blocos de código
      markdown comum (```` ``` ````) que o modelo já produz em respostas
      normais; um ```` ```json ```` genérico seria ambíguo e perigoso.
    - JSON dentro do bloco: formato que o modelo produz de forma
      consistente e que mapeia direto para ``Tool.arguments`` (nomes de
      ferramentas e argumentos são os MESMOS exibidos na seção
      "### FERRAMENTAS DISPONIVEIS" do PromptBuilder, C3).

REGRAS DO PARSER (documentadas, não implícitas):
    1. ``found=True`` ⟺ o marcador de abertura está presente no texto
       (reconheceu a TENTATIVA), mesmo que o conteúdo seja inválido —
       nesse caso ``error`` descreve o problema e nenhuma exceção escapa.
    2. MÚLTIPLAS SOLICITAÇÕES: apenas a PRIMEIRA é considerada (válida
       ou não). Racional: o DaviOS executa UMA ação por vez (segurança e
       previsibilidade) e um primeiro bloco malformado deve ser
       sinalizado, não silenciosamente ignorado em favor de um posterior.
    3. ``arguments`` é opcional → vira ``{}`` (ferramentas sem argumentos,
       como "datetime"). Se presente, DEVE ser um objeto JSON.
    4. Campos extras no JSON são ignorados (tolerância a divagações).
    5. O parser entende SINTAXE apenas — NÃO consulta o ToolRegistry;
       verificar se a ferramenta existe é papel do ToolRouter (C2).
    6. Nunca lança exceção: qualquer entrada (None, bytes, texto gigante,
       tipos absurdos) produz um ToolRequestParseResult.

Módulo puro: sem estado, sem I/O, sem rede, sem dependência do LLM.
Ainda NÃO é conectado ao fluxo de resposta (isso é C5+).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

START_MARKER = "<<<TOOL_REQUEST>>>"
END_MARKER = "<<<END_TOOL_REQUEST>>>"

# Bloco completo: marcadores escapados, conteúdo não-guloso, multi-linha.
_BLOCK_RE = re.compile(
    re.escape(START_MARKER) + r"(.*?)" + re.escape(END_MARKER),
    re.DOTALL,
)

# O Qwen ocasionalmente INVERTE a abertura do bloco (">>>TOOL_REQUEST>>>"
# em vez de "<<<TOOL_REQUEST>>>" — observado em producao). A intencao e
# inequivoca (corpo JSON + "<<<END_TOOL_REQUEST>>>"), entao NORMALIZAMOS
# a variante antes do parse/sanitizacao: ferramentas validas com marcador
# invertido passam a funcionar, e o bloco cru jamais chega ao usuario.
_ALT_START_MARKER = ">>>TOOL_REQUEST>>>"


def _normalize_markers(text: str) -> str:
    """Tolerancia: abertura invertida + '>' sobrando no fechamento."""
    if _ALT_START_MARKER in text:
        text = text.replace(_ALT_START_MARKER, START_MARKER)
    # "<<<END_TOOL_REQUEST>>>>" (com '>' extra) conta como fechado.
    text = re.sub(re.escape(END_MARKER) + r">{1,2}", END_MARKER, text)
    return text


@dataclass
class ToolRequestParseResult:
    """Resultado tipado do parser — nunca uma exceção.

    found=True  → um bloco de solicitação foi reconhecido no texto.
                  Com error="" o pedido é válido (tool_name/arguments
                  preenchidos); com error preenchido a tentativa foi
                  reconhecida mas é malformada.
    found=False → nenhum bloco presente (texto normal do modelo);
                  ``error`` fica vazio, pois não há nada de errado.
    raw_match   → conteúdo bruto ENTRE os marcadores, como escrito.
    """

    found: bool
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_match: str = ""
    error: str = ""


def parse_tool_request(text: Any) -> ToolRequestParseResult:
    """Procura a PRIMEIRA solicitação de ferramenta em ``text``.

    Tolera texto livre antes/depois do bloco. Nunca lança exceção —
    qualquer entrada produz um ToolRequestParseResult (ver regras no
    docstring do módulo).
    """
    # Tolerância a entradas absurdas (regra 6 do módulo):
    if text is None:
        return ToolRequestParseResult(found=False)
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", errors="replace")
        except Exception:
            return ToolRequestParseResult(found=False)
    if not isinstance(text, str):
        return ToolRequestParseResult(found=False)

    text = _normalize_markers(text)
    match = _BLOCK_RE.search(text)
    if match is None:
        # Marcador de abertura sem fechamento = tentativa reconhecida.
        if START_MARKER in text:
            return ToolRequestParseResult(
                found=True,
                error="Bloco de solicitação não foi fechado com "
                f"'{END_MARKER}'.",
            )
        return ToolRequestParseResult(found=False)

    return _parse_block(match.group(1))


def _parse_block(raw_inner: str) -> ToolRequestParseResult:
    """Valida o conteúdo de UM bloco. Nunca lança exceção."""
    raw_stripped = raw_inner.strip()
    if not raw_stripped:
        return ToolRequestParseResult(
            found=True,
            raw_match=raw_inner,
            error="Bloco de solicitação vazio.",
        )

    try:
        payload = json.loads(raw_stripped)
    except json.JSONDecodeError as e:
        return ToolRequestParseResult(
            found=True,
            raw_match=raw_inner,
            error=f"JSON malformado dentro do bloco: {e.msg} (linha "
            f"{e.lineno}, coluna {e.colno}).",
        )
    except Exception as e:  # defesa extra: qualquer falha de parse
        return ToolRequestParseResult(
            found=True,
            raw_match=raw_inner,
            error=f"Falha ao interpretar o bloco: {type(e).__name__}.",
        )

    if not isinstance(payload, dict):
        return ToolRequestParseResult(
            found=True,
            raw_match=raw_inner,
            error="Conteúdo do bloco deve ser um objeto JSON, recebido: "
            f"{type(payload).__name__}.",
        )

    # Campos extras são ignorados (regra 4); só 'tool' e 'arguments'
    # são significativos — vocabulário igual ao da seção de ferramentas
    # do PromptBuilder (C3).
    tool_name = payload.get("tool")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return ToolRequestParseResult(
            found=True,
            raw_match=raw_inner,
            error="Campo obrigatório 'tool' ausente, vazio ou não é string.",
        )
    tool_name = tool_name.strip()

    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return ToolRequestParseResult(
            found=True,
            tool_name=tool_name,
            raw_match=raw_inner,
            error="Campo 'arguments' deve ser um objeto JSON "
            f"(recebido: {type(arguments).__name__}).",
        )

    return ToolRequestParseResult(
        found=True,
        tool_name=tool_name,
        arguments=dict(arguments),
        raw_match=raw_inner,
        error="",
    )


def extract_all_requests(text: Any) -> list[ToolRequestParseResult]:
    """Extrai TODOS os blocos presentes (utilitário opcional).

    NOTA: ``parse_tool_request`` (a API oficial do DaviOS) considera
    apenas o primeiro bloco — ver regra 2 do módulo. Esta função existe
    para diagnóstico/futuras decisões, e segue a mesma regra de nunca
    lançar exceção.
    """
    if text is None:
        return []
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", errors="replace")
        except Exception:
            return []
    if not isinstance(text, str):
        return []
    return [_parse_block(m.group(1)) for m in _BLOCK_RE.finditer(text)]


# Constante de fallback — usada quando a sanitização remove tudo.
_SANITIZE_FALLBACK = "Desculpe, nao consegui formular uma resposta."


def sanitize_response_text(text: Any) -> str:
    """Remove blocos residuais ``<<<TOOL_REQUEST>>>`` de qualquer texto.

    Rede de segurança final (C6): garante que o usuário NUNCA veja o
    bloco do protocolo cru na resposta — mesmo que o Qwen o ecoe/vaze
    por qualquer motivo (primeira ou segunda chamada ao LLM).

    Comportamento:
        - Blocos completos (abertura + fechamento) são removidos, junto
          com os marcadores. Texto ao redor preservado.
        - Bloco malformado (abertura sem fechamento): remove do
          ``START_MARKER`` até o FINAL do texto (não sabemos onde ele
          "deveria" terminar; ser defensivo aqui é seguro porque o
          parser já tratou esse caso antes desta função ser chamada).
        - Se o resultado fica vazio/só espaços, retorna um fallback
          documentado em ``_SANITIZE_FALLBACK``.
        - Entradas não-string (None, bytes, etc.) são toleradas: bytes
          decodificam como UTF-8; outros tipos retornam o fallback.

    Nunca lança exceção.
    """
    if text is None:
        return _SANITIZE_FALLBACK
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8", errors="replace")
        except Exception:
            return _SANITIZE_FALLBACK
    if not isinstance(text, str):
        return _SANITIZE_FALLBACK

    # 0. Normaliza variantes do marcador (ex: abertura invertida do Qwen)
    #    ANTES de remover — sem isso o bloco cru vazaria para o usuario.
    text = _normalize_markers(text)

    # 1. Remove blocos completos (abertura + fechamento).
    cleaned = _BLOCK_RE.sub("", text)

    # 2. Bloco malformado: abertura sem fechamento → corta do marker em diante.
    if START_MARKER in cleaned:
        idx = cleaned.index(START_MARKER)
        cleaned = cleaned[:idx]

    # 3. Normaliza quebras de linha consecutivas (3+ → 2) e limpa bordas.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    # 4. Remove prefixo 'DaviOS:' do início (caso o Qwen o gere).
    #    Só remove no início, preserva menções no meio do texto.
    cleaned = re.sub(r"^DaviOS\s*:\s*", "", cleaned, count=1)

    # 5. Fallback se sobrou nada de útil.
    if not cleaned:
        return _SANITIZE_FALLBACK

    return cleaned
