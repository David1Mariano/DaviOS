"""Testes de representacao de contexto: prompt com historico e referencias.

Reparo de imports/fixtures: todos os simbolos usados abaixo existem no
projeto - DaviosConfig (config/davios_config.py), PromptBuilder
(brain/prompt_builder.py), DEFAULT_PERSONALITY (personality/personality.py),
resolve_context_reference (brain/intent_classifier.py) e _Ctx (mesma forma
de ConversationContext, definido em tests/test_intent_classifier.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain.intent_classifier import resolve_context_reference
from brain.prompt_builder import PromptBuilder
from config.davios_config import DaviosConfig
from personality.personality import DEFAULT_PERSONALITY
from test_intent_classifier import _Ctx


def _msg(user: str, response: str) -> dict:
    """Mensagem no formato real de ConversationContext (user/response)."""
    return {"user": user, "response": response}


# ============================================================================
# TESTE 2: Mensagem atual não aparece duplicada no histórico
# ============================================================================

class TestCurrentMessageNotDuplicated:
    """A mensagem que deve ser respondida não pode aparecer também no histórico."""

    def setup_method(self):
        self.config = DaviosConfig()
        self.config.max_recent_messages = 6
        self.builder = PromptBuilder(self.config, DEFAULT_PERSONALITY)

    def test_current_message_only_in_current_section(self):
        """A mensagem atual aparece na seção 'Mensagem do usuario:',
        mas NÃO no histórico (que só tem mensagens anteriores)."""
        ctx = _Ctx(messages=[_msg("mensagem anterior", "resposta anterior")])
        current = "esta e a mensagem atual"
        built = self.builder.build(current, context=ctx)

        # A mensagem atual aparece na seção final
        assert f"Mensagem do usuario: {current}" in built.prompt

        # O histórico não deve conter a mensagem atual
        hist_block = built.prompt.split("Mensagem do usuario:")[0]
        assert current not in hist_block

    def test_history_only_has_previous_messages(self):
        ctx = _Ctx(messages=[_msg("antiga 1", "resp 1"), _msg("antiga 2", "resp 2")])
        built = self.builder.build("nova msg", context=ctx)
        hist_section = built.prompt.split("Mensagem do usuario:")[0]
        assert "antiga 1" in hist_section
        assert "antiga 2" in hist_section
        assert "nova msg" not in hist_section


# ============================================================================
# TESTE 5: Referência "e a população?" mantém contexto da pergunta anterior
# ============================================================================

class TestReferenceMaintainsContext:
    """Referências como 'e a população?' devem manter o contexto
    da pergunta anterior sobre capital do Brasil."""

    def setup_method(self):
        self.config = DaviosConfig()
        self.config.max_recent_messages = 6
        self.builder = PromptBuilder(self.config, DEFAULT_PERSONALITY)

    def test_follow_up_knows_previous_topic(self):
        """O prompt deve conter informação suficiente para vincular
        'e a população?' à pergunta sobre capital."""
        ctx = _Ctx(
            messages=[_msg("Qual é a capital do Brasil?", "Brasília.")],
            current_topic="capital",
        )
        built = self.builder.build("E a população?", context=ctx)

        # Tanto a pergunta anterior quanto a resposta devem estar visíveis
        assert "capital do Brasil" in built.prompt
        assert "Brasília" in built.prompt

        # A mensagem atual também deve estar lá
        assert "E a população?" in built.prompt


# ============================================================================
# TESTE 6: Contexto não é inventado quando não existe antecedente
# ============================================================================

class TestNoInventedContext:
    """Quando não há histórico anterior, o sistema não deve inventar
    contexto falso."""

    def setup_method(self):
        self.config = DaviosConfig()
        self.builder = PromptBuilder(self.config, DEFAULT_PERSONALITY)

    def test_empty_history_means_no_invented_context(self):
        """Sem mensagens anteriores, o prompt não pode conter nenhuma
        mensagem fictícia: nem entrada 'Usuario:' nem resposta 'DaviOS:'
        no histórico. O cabeçalho legado 'Conversa recente:' pode ser
        emitido (comportamento byte a byte de HEAD), desde que venha
        sem conteúdo inventado."""
        ctx = _Ctx(messages=[])
        built = self.builder.build("Oi, tudo bem?", context=ctx)

        # Histórico (tudo antes da seção final) sem mensagens inventadas
        hist_section = built.prompt.split("Mensagem do usuario:")[0]
        assert "Usuario:" not in hist_section
        assert "DaviOS:" not in hist_section

        # Apenas a mensagem atual, na seção correta e sem repetição
        assert "Mensagem do usuario: Oi, tudo bem?" in built.prompt
        assert built.prompt.count("Oi, tudo bem?") == 1

    def test_resolve_context_reference_returns_none_without_history(self):
        """resolve_context_reference deve retornar None quando
        não há mensagens anteriores."""
        ctx = _Ctx(messages=[])
        result = resolve_context_reference("e disso?", context=ctx)
        assert result is None

    def test_resolve_context_reference_returns_none_for_full_question(self):
        """Perguntas completas não devem ter referência resolvida
        mesmo com histórico (não são follow-ups)."""
        ctx = _Ctx(
            messages=[_msg("algo anterior", "resp anterior")]
        )
        result = resolve_context_reference(
            "por que o céu é azul?", context=ctx
        )
        assert result is None
