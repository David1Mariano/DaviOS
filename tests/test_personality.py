"""Testes da personalidade (Personality) e das regras injetadas no sistema.

Como nao e possivel testar automaticamente se o modelo de linguagem vai de
fato variar as respostas na pratica, estes testes confirmam apenas que a
INSTRUCAO esta presente no texto de sistema gerado corretamente (e que as
regras antigas continuam la).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from personality.personality import DEFAULT_PERSONALITY, Personality


def _system_text() -> str:
    return DEFAULT_PERSONALITY.to_system_text()


def test_new_engagement_rule_casual_no_generic_closer():
    """Conversa casual nao deve fechar sempre com 'como posso te ajudar' e
    deve usar memoria/historico quando disponivel."""
    text = _system_text()
    assert "como posso te ajudar" in text
    assert "varie a resposta" in text
    assert "USE essa informacao" in text
    assert "historico" in text


def test_new_rule_asks_about_past_topic():
    """Quando ha projeto/interesse/assunto mencionado antes, perguntar
    ativamente sobre ele."""
    text = _system_text()
    assert "pergunte ativamente" in text


def test_new_rule_genuine_reactions_without_feelings():
    """Reacoes genuinas, mas sem afirmar consciencia/sentimentos reais."""
    text = _system_text()
    assert "entusiasmo" in text
    assert "eu sinto" in text
    assert "sem alegar ter consciencia" in text


def test_new_rule_vary_response_structure():
    """Evitar fechar toda resposta curta com a mesma pergunta."""
    text = _system_text()
    assert "varie a estrutura das respostas" in text


def test_new_rule_no_false_memory_promise():
    """Nao prometer persistencia de memoria que nao foi confirmada."""
    text = _system_text()
    assert "vou lembrar disso" in text
    assert "nao pode cumprir" in text


def test_old_rules_still_present():
    """Regras antigas (anti-eco, idioma, honestidade) continuam presentes."""
    text = _system_text()
    assert "Evite repetir ou ecoar a frase do usuario" in text
    assert "Responda em portugues do Brasil." in text
    assert "Nunca finja executar acoes no computador" in text
    assert "admita com honestidade" in text


def test_all_rules_rendered_as_bullets_in_system_text():
    """Cada regra aparece no formato '- regra' em to_system_text()."""
    text = _system_text()
    for rule in DEFAULT_PERSONALITY.rules:
        assert f"- {rule}" in text


def test_engagement_rules_reach_built_system():
    """Via PromptBuilder, as novas regras chegam ao texto de sistema final."""
    from brain.prompt_builder import PromptBuilder
    from config.davios_config import DaviosConfig

    built = PromptBuilder(DaviosConfig()).build("oi", memories=[])
    assert "pergunte ativamente" in built.system
    assert "eu sinto" in built.system


def test_personality_to_dict_includes_rules():
    """to_dict() reflete a lista completa de regras."""
    data = Personality().to_dict()
    assert "rules" in data
    assert isinstance(data["rules"], list)
    assert any("pergunte ativamente" in r for r in data["rules"])
