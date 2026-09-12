"""Fechamento do loop de ferramentas do DaviOS (C5).

Orquestra o ciclo completo UMA vez por resposta do usuário:
    Qwen responde → parser reconhece pedido (C4) → REGRA DE OURO de
    autorização → ToolRouter executa (C2) → resultado/negação volta ao
    Qwen em uma SEGUNDA chamada → resposta final ao usuário.

REGRA DE OURO (todas as condições devem ser verdadeiras para executar):
    1. ``config.tools_visible_to_llm`` ligada (o Qwen pode saber e pedir);
    2. flag da CATEGORIA da ferramenta ligada:
       category="actions" → config.actions_enabled
       category="web"     → config.web_enabled
       (categoria vazia/desconhecida NUNCA executa — default seguro);
    3. ferramenta existe no ToolRegistry (via ToolRouter.find).
Falhando qualquer condição → NÃO executa; o Qwen recebe uma negação
GENÉRICA (sem vazar nomes de flags ou detalhes de configuração).

DECISÕES DOCUMENTADAS:
    - LIMITE DE SEGURANÇA: no máximo 1 execução por resposta do usuário.
      O parser roda APENAS sobre a primeira resposta do Qwen; a segunda
      resposta (com o resultado incorporado) NUNCA é re-parseada — não
      existe loop de idas e vindas. Isso é decisão de projeto, não
      limitação temporária (múltiplas execuções encadeadas seriam C6+).
    - PEDIDO MALFORMADO (JSON quebrado/bloco não fechado): reconhecido,
      logado, NÃO executado, e o usuário vê o texto cru do Qwen
      inalterado (incluindo o bloco malformado) SEM segunda chamada ao
      LLM. Justificativa: (a) economiza uma chamada de LLM; (b) o texto
      normal ao redor do bloco permanece intacto e útil; (c) remover o
      bloco com regex arrisca corromper conteúdo legítimo; (d) caso é
      raro e inofensivo visualmente.
    - SEGUNDA CHAMADA AO LLM FALHA → degradação graciosa: o resultado
      (ou a negação) é devolvido diretamente como texto final, sem
      derrubar a conversa.
    - LOGGING: todo pedido reconhecido é logado em INFO (logger
      "davios.toolflow") com nome da ferramenta, APENAS AS CHAVES dos
      argumentos (nunca valores — podem conter dados do usuário),
      autorização e resultado resumido.

Ainda fora de escopo (C6): sanitização de blocos residuais na resposta
final, múltiplas execuções, confirmação interativa.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from brain.tool_registry import Tool, ToolRouter
from brain.tool_request_parser import parse_tool_request

logger = logging.getLogger("davios.toolflow")

# Proteção do contexto do modelo local: resultado truncado no follow-up.
MAX_RESULT_CHARS = 1500

DENIAL_MESSAGE_TEMPLATE = (
    "O pedido de uso da ferramenta '{name}' nao foi autorizado pelas "
    "configuracoes de seguranca atuais."
)


@dataclass
class ToolFlowOutcome:
    """Resultado do ciclo de ferramentas para UMA resposta do usuário."""

    final_text: str
    request_found: bool = False
    tool_name: str = ""
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    authorized: bool = False
    tool_executed: bool = False
    execution_success: bool = False
    execution_error: str = ""
    second_llm_call: bool = False
    denial_reason: str = ""  # motivo INTERNO (logs/metadata); genérico no prompt

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_found": self.request_found,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "authorized": self.authorized,
            "tool_executed": self.tool_executed,
            "execution_success": self.execution_success,
            "execution_error": self.execution_error,
            "second_llm_call": self.second_llm_call,
            "denial_reason": self.denial_reason,
        }


class ToolExecutionOrchestrator:
    """Coordena UM ciclo de ferramenta por resposta (componente puro).

    Depende de: config (flags), ToolRouter (registro + execução) e de um
    callback ``run_llm(prompt, system) -> str`` para a segunda chamada ao
    modelo — injetado pelo CognitiveCore, mantendo este módulo livre de
    conhecimento sobre LLMProvider.
    """

    def __init__(
        self,
        config: Any,
        tool_router: Optional[ToolRouter] = None,
    ) -> None:
        self.config = config
        self.router = tool_router

    def is_enabled(self) -> bool:
        """Condição 1 da REGRA DE OURO + router injetado."""
        return bool(
            getattr(self.config, "tools_visible_to_llm", False)
        ) and self.router is not None

    def handle_first_response(
        self,
        llm_text: str,
        *,
        original_prompt: str,
        original_system: str,
        run_llm: Callable[[str, str], str],
    ) -> ToolFlowOutcome:
        """Processa a PRIMEIRA resposta do Qwen (no máximo 1 execução).

        ``run_llm(prompt, system)`` executa a segunda chamada ao modelo.
        Nunca propaga exceção de ferramenta/parse; falha da segunda
        chamada ao LLM degrada graciosamente (ver docstring do módulo).
        """
        if not self.is_enabled():
            # Flag off: comportamento idêntico ao pré-C5 (nem parse).
            return ToolFlowOutcome(final_text=llm_text)

        parsed = parse_tool_request(llm_text)

        if not parsed.found:
            if "TOOL_REQUEST" in llm_text:
                # Variante de marcador ainda malformada apos a normalizacao
                # (ex: "<<TOOL_REQUEST>>"): o bloco NUNCA segue como conversa
                # — degrada com denial; o sanitizer final garante que o
                # usuario nao veja nada cru.
                logger.warning(
                    "[TOOLFLOW] bloco TOOL_REQUEST-like malformado; "
                    "bloqueado (sem execucao)"
                )
                return ToolFlowOutcome(
                    final_text="",
                    request_found=True,
                    denial_reason="bloco de ferramenta malformado",
                )
            # Texto normal do Qwen: fluxo normal, sem segunda chamada.
            return ToolFlowOutcome(final_text=llm_text)

        logger.info(
            "[TOOLFLOW] pedido reconhecido: tool=%s argumentos=%s",
            parsed.tool_name or "<malformado>",
            sorted(parsed.arguments.keys()),
        )

        if parsed.error:
            # DECISÃO (2c): texto cru segue inalterado, sem segunda chamada.
            logger.info(
                "[TOOLFLOW] pedido malformado ignorado (sem execucao): %s",
                parsed.error,
            )
            return ToolFlowOutcome(
                final_text=llm_text,
                request_found=True,
                tool_name=parsed.tool_name,
                tool_arguments=parsed.arguments,
                denial_reason="pedido malformado",
            )

        # --- Condição 3 da REGRA DE OURO: existe no registry? ---
        tool = self.router.find(parsed.tool_name)
        if tool is None:
            logger.info(
                "[TOOLFLOW] negado: tool=%s motivo_interno=ferramenta inexistente",
                parsed.tool_name,
            )
            return self._deny(
                llm_text, parsed, "ferramenta inexistente no registro",
                original_prompt, original_system, run_llm,
            )

        # --- Condição 2 da REGRA DE OURO: flag da categoria ---
        allowed, internal_reason = self._category_allowed(tool)
        if not allowed:
            logger.info(
                "[TOOLFLOW] negado: tool=%s motivo_interno=%s",
                parsed.tool_name,
                internal_reason,
            )
            return self._deny(
                llm_text, parsed, internal_reason,
                original_prompt, original_system, run_llm,
            )

        # --- Autorizado: executa UMA vez ---
        logger.info("[TOOLFLOW] autorizado: tool=%s", parsed.tool_name)
        result = self.router.execute(parsed.tool_name, args=parsed.arguments)
        logger.info(
            "[TOOLFLOW] execucao concluida: tool=%s success=%s",
            parsed.tool_name,
            result.success,
        )

        outcome = ToolFlowOutcome(
            final_text=llm_text,
            request_found=True,
            tool_name=parsed.tool_name,
            tool_arguments=parsed.arguments,
            authorized=True,
            tool_executed=True,
            execution_success=result.success,
            execution_error=result.error,
        )

        if result.success:
            result_text = self._format_result(result.output)
        else:
            result_text = result.error

        followup_prompt = self._build_followup_prompt(
            original_prompt, parsed.tool_name, result_text, result.success
        )

        outcome.final_text = self._run_second_llm(
            run_llm, followup_prompt, original_system,
            fallback=f"Resultado da ferramenta '{parsed.tool_name}': {result_text}",
            outcome=outcome,
        )
        return outcome

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _category_allowed(self, tool: Tool) -> tuple[bool, str]:
        """Condição 2 da REGRA DE OURO (flag por categoria)."""
        category = (getattr(tool, "category", "") or "").strip().lower()
        if category == "actions":
            if bool(getattr(self.config, "actions_enabled", False)):
                return True, ""
            return False, "categoria 'actions' desabilitada"
        if category == "web":
            if bool(getattr(self.config, "web_enabled", False)):
                return True, ""
            return False, "categoria 'web' desabilitada"
        # Categoria vazia/desconhecida: default seguro — nunca executa.
        return False, f"categoria nao auto-executavel: '{category or 'vazia'}'"

    def _deny(
        self,
        llm_text: str,
        parsed: Any,
        internal_reason: str,
        original_prompt: str,
        original_system: str,
        run_llm: Callable[[str, str], str],
    ) -> ToolFlowOutcome:
        """Negação: mensagem GENÉRICA ao Qwen (sem vazar configuração).

        O followup NÃO inclui a seção de ferramentas (mesma correção do bug
        'bloco cru aparece + resultado atrasado')."""
        denial = DENIAL_MESSAGE_TEMPLATE.format(name=parsed.tool_name)
        # Extrai apenas a mensagem original do usuário (sem ferramentas)
        user_message = ""
        if "Mensagem do usuario:" in original_prompt:
            user_message = original_prompt.split("Mensagem do usuario:")[-1].strip()
        followup_prompt = (
            f"Mensagem original do usuario: {user_message}\n\n"
            f"{denial} Responda ao usuario de forma clara, sem usar ferramentas."
        )
        outcome = ToolFlowOutcome(
            final_text=llm_text,
            request_found=True,
            tool_name=parsed.tool_name,
            tool_arguments=parsed.arguments,
            authorized=False,
            denial_reason=internal_reason,
        )
        outcome.final_text = self._run_second_llm(
            run_llm, followup_prompt, original_system,
            fallback=denial, outcome=outcome,
        )
        return outcome

    @staticmethod
    def _build_followup_prompt(
        original_prompt: str,
        tool_name: str,
        result_text: str,
        success: bool,
    ) -> str:
        """Monta o followup para a 2ª chamada ao LLM SEM a seção de ferramentas.

        O followup NÃO deve conter a seção "### FERRAMENTAS DISPONIVEIS"
        nem as instruções/exemplos do protocolo <<<TOOL_REQUEST>>> — o
        segundo LLM não deve ter como pedir outra ferramenta. Extrai
        apenas a mensagem original do usuário do prompt e reconstrói
        um followup limpo com o resultado da execução.
        """
        # Extrai a mensagem original do usuário (última seção do prompt)
        user_message = ""
        if "Mensagem do usuario:" in original_prompt:
            user_message = original_prompt.split("Mensagem do usuario:")[-1].strip()

        if success:
            resultado_bloco = (
                f"Resultado da execucao da ferramenta '{tool_name}':\n"
                f"{result_text}"
            )
        else:
            resultado_bloco = (
                f"A execucao da ferramenta '{tool_name}' falhou: "
                f"{result_text}"
            )

        followup = (
            f"Mensagem original do usuario: {user_message}\n\n"
            f"{resultado_bloco}\n\n"
            "Responda ao usuario de forma natural usando o resultado acima. "
            "NAO peca nenhuma ferramenta nesta resposta."
        )
        return followup

    def _run_second_llm(
        self,
        run_llm: Callable[[str, str], str],
        prompt: str,
        system: str,
        *,
        fallback: str,
        outcome: ToolFlowOutcome,
    ) -> str:
        """Segunda chamada ao Qwen, com degradação graciosa em falha."""
        try:
            text = run_llm(prompt, system)
            outcome.second_llm_call = True
            return text
        except Exception as e:  # LLM caiu na 2ª chamada: não derruba a conversa
            logger.warning(
                "[TOOLFLOW] segunda chamada ao LLM falhou (%s); usando "
                "resultado direto como resposta.",
                type(e).__name__,
            )
            return fallback

    @staticmethod
    def _format_result(output: Any) -> str:
        """Serializa o resultado da ferramenta para o follow-up."""
        if isinstance(output, str):
            text = output
        else:
            try:
                text = json.dumps(output, ensure_ascii=False, default=str)
            except Exception:
                text = str(output)
        text = text.strip() or "(sem saida)"
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "..."
        return text