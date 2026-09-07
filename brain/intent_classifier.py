"""Classificador de intenção do ConversationEngine.

Baseado em regras leves (sem LLM): rápido, determinístico e offline.
Intenções: GREETING, FAREWELL, MEMORY_STATEMENT, MEMORY_QUERY,
GENERAL_QUESTION, OPINION, COMMAND, CONVERSATION, UNKNOWN.
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    MEMORY_STATEMENT = "memory_statement"
    MEMORY_QUERY = "memory_query"
    GENERAL_QUESTION = "general_question"
    OPINION = "opinion"
    COMMAND = "command"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


QUESTION_STARTERS = {
    "qual", "quais", "quem", "que", "o que", "oque", "como", "quando",
    "onde", "por que", "porque", "porque", "quanto", "quanta", "quantos",
    "quantas", "explique", "explica", "explicar", "me explique",
    "me ajuda", "me ajude", "ajuda", "ajude", "consegue", "pode",
}

MEMORY_QUERY_MARKERS = (
    "meu nome", "minha ", "meu ", "eu gosto", "eu nao gosto",
    "eu odeio", "eu adoro", "eu prefiro", "o que eu", "quem eu",
    "lembra", "lembre", "voce lembra", "ja falei",
)

OPINION_MARKERS = (
    "o que voce acha", "voce acha", "sua opiniao", "qual sua opiniao",
    "o que voce pensa", "voce concorda", "voce gosta de conversar",
    "o que voce sente",
)

COMMAND_MARKERS = (
    "abra ", "abrir ", "execute ", "executar ", "rode ", "rodar ",
    "apague ", "apagar ", "delete ", "instale ", "desligue ",
    "desligar ", "feche ", "fechar ",
)

MEMORY_STATEMENT_MARKERS = (
    "eu gosto", "eu nao gosto", "nao gosto mais", "eu odeio", "eu adoro",
    "eu amo", "eu detesto", "meu nome", "minha comida favorita",
    "meu favorito", "eu prefiro", "eu estudo", "estou estudando",
    "eu trabalho", "eu moro", "eu tenho",
)

# Perguntas gerais típicas de conhecimento
GENERAL_QUESTION_HINTS = (
    "o que e", "oque e", "o que sao", "por que o", "porque o",
    "como funciona", "explique", "qual a diferenca", "defina",
    "me ensine", "me ajude a", "significa",
)


class IntentClassifier:
    """Classificador de intenção por regras, extensível para LLM futuramente."""

    def classify(self, text: str) -> Intent:
        original = (text or "").strip()
        normalized = original.lower().rstrip("!.?,")
        has_question_mark = "?" in original

        if self._is_farewell(normalized):
            return Intent.FAREWELL
        if self._is_greeting(normalized):
            return Intent.GREETING
        if self._matches_any(normalized, COMMAND_MARKERS):
            return Intent.COMMAND
        if self._matches_any(normalized, OPINION_MARKERS):
            return Intent.OPINION
        if self.is_memory_query(normalized, has_question_mark):
            return Intent.MEMORY_QUERY
        if self.is_memory_statement(normalized):
            return Intent.MEMORY_STATEMENT
        if self.is_general_question(normalized, original=original):
            return Intent.GENERAL_QUESTION
        return Intent.CONVERSATION

    # ------------------------------------------------------------------

    @staticmethod
    def _matches_any(text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_greeting(text: str) -> bool:
        words = text.split()
        if not words or len(words) > 4:
            return False
        if "tudo bem" in text or "tudo bom" in text:
            greetings = {"oi", "ola", "eae", "iai", "hello", "hi", "opa",
                         "bom", "boa", "dia", "tarde", "noite", "e", "ai",
                         "tudo", "bem"}
            return any(w in greetings for w in words)
        greetings = {
            "oi", "ola", "eae", "e", "ai", "iai", "hello", "hi", "opa",
            "bom", "boa", "dia", "tarde", "noite", "tudo", "bem",
        }
        return all(w in greetings for w in words)

    @staticmethod
    def _is_farewell(text: str) -> bool:
        farewells = {
            "sair", "tchau", "encerrar", "exit", "quit", "bye",
            "ate mais", "falow", "falou", "flw", "adeus",
        }
        return text in farewells

    def is_memory_query(self, text: str, has_question_mark: bool = False) -> bool:
        if not self._matches_any(text, MEMORY_QUERY_MARKERS):
            return False
        return has_question_mark or self._is_interrogative_start(text)

    def is_memory_statement(self, text: str) -> bool:
        return self._matches_any(text, MEMORY_STATEMENT_MARKERS)

    def is_general_question(self, text: str, original: str = "") -> bool:
        if self._matches_any(text, GENERAL_QUESTION_HINTS):
            return True
        first_word = text.split()[0] if text else ""
        if first_word in QUESTION_STARTERS and not self._matches_any(
            text, MEMORY_QUERY_MARKERS
        ):
            return True
        return (
            "?" in original
            and not self._matches_any(text, MEMORY_STATEMENT_MARKERS)
        )

    @staticmethod
    def _is_interrogative_start(text: str) -> bool:
        first = text.split()[0] if text else ""
        return first in QUESTION_STARTERS or first in {"qual", "quem", "quais"}


def extract_topic_from_previous(context) -> str:
    """Extrai o tópico atual do contexto conversacional (para continuidade)."""
    topic = getattr(context, "current_topic", "") or ""
    return topic.strip()


def resolve_context_reference(text: str, context) -> Optional[str]:
    """Tenta resolver referências como 'isso', 'disso' ao tópico anterior.

    Retorna o texto expandido ou None se não houver referência resolvida.
    """
    lowered = text.lower()
    has_reference = re.search(
        r"\b(isso|disso|disto|dessa|desse|aquele|aquilo|esse|essa)\b", lowered
    )
    if not has_reference:
        return None
    topic = extract_topic_from_previous(context)
    if not topic:
        return None
    last_response = getattr(context, "last_response", "") or ""
    parts = [
        f"Mensagem atual do usuario: {text}",
        f"Assunto recente da conversa: {topic}",
        f"Sua ultima resposta: {last_response}",
    ]
    return "\n".join(parts)
