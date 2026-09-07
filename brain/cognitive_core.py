"""CognitiveCore do DaviOS.

Coordena o raciocínio antes da resposta:
    mensagem atual + contexto + memórias relevantes + intenção
        → PromptBuilder → LLMProvider → texto de resposta

O LLM NÃO controla memória nem sistema: ele apenas produz linguagem.
A memória continua sendo gravada/recuperada pelo MemoryManager; o
CognitiveCore apenas seleciona o que é relevante para o prompt.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from brain.intent_classifier import Intent, IntentClassifier
from brain.llm_provider import LLMProvider, LLMRequest, LLMUnavailableError
from brain.prompt_builder import PromptBuilder
from config.davios_config import DaviosConfig
from personality.personality import DEFAULT_PERSONALITY

logger = logging.getLogger("davios.cognitive")

# Palavras muito comuns não servem como chave de memória
STOPWORDS = {
    "para", "com", "uma", "que", "por", "mais", "como", "mas", "isso",
    "este", "esta", "esse", "essa", "sobre", "quando", "muito", "pelo",
    "pela", "sem", "dos", "das", "nao", "sim", "voce", "meu", "minha",
    "tambem", "entao", "aqui", "onde", "todo", "toda", "ele", "ela",
}


class CognitiveCore:
    """Núcleo cognitivo: memória relevante + contexto → prompt → LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[DaviosConfig] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ):
        self.config = config or DaviosConfig.load()
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder or PromptBuilder(
            self.config, DEFAULT_PERSONALITY
        )
        self.classifier = IntentClassifier()

    # ------------------------------------------------------------------

    def generate_response(self, user_message: str, *, context=None, intent: Optional[Intent] = None) -> dict[str, Any]:
        """Produz resposta via LLM. Lança LLMUnavailableError se indisponível."""
        resolved_intent = intent or self.classifier.classify(user_message)
        resolved_context = self._resolve_context_reference(user_message, context)
        memories = self.retrieve_relevant_memories(user_message, context)

        built = self.prompt_builder.build(
            user_message,
            context=context,
            memories=memories,
            intent=resolved_intent.value,
            resolved_context=resolved_context,
        )
        logger.debug(
            "[CONTEXT] prompt montado com %s memorias", len(memories)
        )

        request = LLMRequest(
            prompt=built.prompt,
            system=built.system,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        response = self.llm_provider.generate(request)
        return {
            "text": response.text,
            "intent": resolved_intent.value,
            "memories_used": memories,
            "llm": response.to_dict(),
        }

    # ------------------------------------------------------------------
    # Recuperação de memórias relevantes (simples, extensível a RAG)
    # ------------------------------------------------------------------

    def retrieve_relevant_memories(self, text: str, context=None, memory_manager=None) -> list[Any]:
        """Seleciona memórias relevantes para a mensagem atual.

        Estratégia atual (sem embeddings): palavras-chave da mensagem +
        tópico atual da conversa. Não envia o banco inteiro ao modelo.
        """
        manager = memory_manager
        if manager is None:
            manager = getattr(self, "memory_manager", None)
        if manager is None:
            return []

        keywords = self._extract_keywords(text)
        if context is not None:
            topic = getattr(context, "current_topic", "")
            if topic:
                keywords.append(topic.lower())

        relevant: list[Any] = []
        seen_targets: set[str] = set()

        for keyword in keywords:
            try:
                facts = manager.database.find_facts_by_target(keyword)
            except Exception:
                facts = []
            for fact in facts:
                if not getattr(fact, "is_active", True):
                    continue
                key = (fact.target or "").casefold()
                if key in seen_targets:
                    continue
                seen_targets.add(key)
                relevant.append(fact)
                if len(relevant) >= self.config.max_memories_in_prompt:
                    return relevant

        # Preferências ativas completam o contexto quando ainda há espaço
        try:
            preferences = manager.database.find_active_preferences()
        except Exception:
            preferences = []
        for fact in preferences:
            if len(relevant) >= self.config.max_memories_in_prompt:
                break
            key = (fact.target or "").casefold()
            if key in seen_targets:
                continue
            seen_targets.add(key)
            relevant.append(fact)
        return relevant

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        words = re.findall(r"[^\W_]+", (text or "").lower(), flags=re.UNICODE)
        keywords = [
            w for w in words if len(w) > 3 and w not in STOPWORDS
        ]
        # Preserva ordem, remove duplicatas
        seen: set[str] = set()
        ordered = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                ordered.append(word)
        return ordered[:6]

    @staticmethod
    def _resolve_context_reference(text: str, context) -> Optional[str]:
        from brain.intent_classifier import resolve_context_reference

        return resolve_context_reference(text, context)
