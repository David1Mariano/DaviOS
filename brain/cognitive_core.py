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
from brain.tool_execution_flow import ToolExecutionOrchestrator
from brain.tool_registry import ToolRouter
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
        memory_manager=None,
        tool_router: Optional[ToolRouter] = None,
    ):
        self.config = config or DaviosConfig.load()
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder or PromptBuilder(
            self.config, DEFAULT_PERSONALITY
        )
        self.memory_manager = memory_manager
        self.classifier = IntentClassifier()
        # C5: loop de ferramentas. Inerte quando tools_visible_to_llm=False
        # (padrão) ou sem router injetado — o fluxo é idêntico ao pré-C5.
        self.tool_flow = ToolExecutionOrchestrator(
            self.config, tool_router=tool_router
        )
        # Reaproveita o MESMO registry usado pelo ToolRouter (sem duplicar).
        # Com tool_router=None (ex: testes antigos), registry fica None e a
        # seção de ferramentas simplesmente não aparece — degrada com segurança.
        self.tools_registry = tool_router.registry if tool_router else None

    # ------------------------------------------------------------------

    def generate_response(
        self,
        user_message: str,
        *,
        context=None,
        intent: Optional[Intent] = None,
        new_facts: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Produz resposta via LLM. Lança LLMUnavailableError se indisponível.

        ``new_facts`` carrega fatos recém-extraídos/salvos nesta mesma
        mensagem (ex.: "Meu nome é Davi"), garantindo que o modelo use a
        informação acabada de fornecer.
        """
        # Aceita Intent (enum) ou string simples (ex.: decision.intent)
        if intent is None:
            resolved_intent = self.classifier.classify(user_message)
            intent_value = resolved_intent.value
        elif isinstance(intent, str):
            resolved_intent = intent
            intent_value = intent
        else:
            resolved_intent = intent
            intent_value = intent.value
        resolved_context = self._resolve_context_reference(user_message, context)
        memories = self.retrieve_relevant_memories(
            user_message, context, intent=resolved_intent
        )
        logger.info(
            "[MEMORY] retrieved=%d relevant=%d",
            len(memories), len(memories),
        )

        built = self.prompt_builder.build(
            user_message,
            context=context,
            memories=memories,
            intent=intent_value,
            resolved_context=resolved_context,
            new_facts=new_facts,
            style_profile=self._current_style_profile(),
            tools_registry=self.tools_registry,
        )
        logger.info(
            "[LLM] prompt_has_tools_section=%s prompt_len=%d",
            "FERRAMENTAS DISPONIVEIS" in built.prompt,
            len(built.prompt),
        )
        logger.info(
            "[CONTEXT] recent_messages=%d memories=%d new_facts=%d",
            len(getattr(context, "messages", []) or []),
            len(memories),
            len(new_facts or []),
        )

        request = LLMRequest(
            prompt=built.prompt,
            system=built.system,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        logger.info("[LLM] generation_started provider=%s", self.llm_provider.name)
        response = self.llm_provider.generate(request)
        logger.info("[LLM] generation_completed backend=%s", response.backend)
        logger.info("[LLM] raw_response text=%r", response.text)

        # C5: fechamento do loop de ferramentas — apenas quando
        # tools_visible_to_llm=True E router injetado. Com a flag off o
        # fluxo é idêntico ao pré-C5 (uma única chamada ao LLM).
        if self.tool_flow.is_enabled():
            outcome = self.tool_flow.handle_first_response(
                response.text,
                original_prompt=built.prompt,
                original_system=built.system,
                run_llm=self._run_followup_llm,
            )
            return {
                "text": outcome.final_text,
                "intent": intent_value,
                "memories_used": memories,
                "llm": response.to_dict(),
                "tool_flow": outcome.to_dict(),
            }

        return {
            "text": response.text,
            "intent": intent_value,
            "memories_used": memories,
            "llm": response.to_dict(),
        }

    def _run_followup_llm(self, prompt: str, system: str) -> str:
        """Segunda chamada ao LLM dentro do ciclo de ferramentas (C5).

        Pode lançar LLMUnavailableError — o orquestrador degrada
        graciosamente nesse caso (não derruba a conversa).
        """
        request = LLMRequest(
            prompt=prompt,
            system=system,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        response = self.llm_provider.generate(request)
        return response.text

    # ------------------------------------------------------------------
    # Recuperação de memórias relevantes (simples, extensível a RAG)
    # ------------------------------------------------------------------

    def _current_style_profile(self):
        """Recupera o StyleProfile do MemoryManager sem acoplar modulos.

        Retorna None quando o manager nao existe ou o perfil nao esta
        disponivel, para manter a geracao funcionando por regras.
        """
        manager = getattr(self, "memory_manager", None)
        if manager is None:
            return None
        try:
            return manager.style_profile
        except Exception:
            return None

    def retrieve_relevant_memories(self, text: str, context=None, memory_manager=None, intent=None) -> list[Any]:
        """Seleciona memórias relevantes para a mensagem atual.

        Estrategia (sem embeddings):
        1. Palavras-chave da mensagem + tópico atual da conversa;
        2. Fatos de identidade (ex.: nome do usuario);
        3. Fatos de projetos/estudos (working_on, studies);
        4. Preferências ativas *apenas* quando a query eh sobre a memoria
           do proprio usuario (MEMORY_QUERY) — evita injetar preferencias em
           conversas nao relacionadas.
        Nao envia o banco inteiro ao modelo.
        """
        from brain.intent_classifier import Intent

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
        is_memory_query = intent == Intent.MEMORY_QUERY

        def _cap() -> bool:
            return len(relevant) >= self.config.max_memories_in_prompt

        def _add(fact: Any) -> None:
            if not getattr(fact, "is_active", True):
                return
            key = (fact.target or "").casefold()
            if key in seen_targets:
                return
            seen_targets.add(key)
            relevant.append(fact)

        # 1. Correspondencia direta por palavra-chave (maior prioridade)
        for keyword in keywords:
            if _cap():
                break
            try:
                facts = manager.database.find_facts_by_target(
                    keyword, include_inactive=False
                )
            except Exception:
                facts = []
            for fact in facts:
                if _cap():
                    break
                _add(fact)

        # 2-3. Identidade e declarativos: sempre incluidos (pequeno conjunto,
        #    sempre relevantes para contextualizar o usuario).
        if not _cap():
            for family in ("identity", "working_on", "studies"):
                try:
                    facts = manager.database.find_active_preferences(relation_family=family)
                except Exception:
                    facts = []
                for fact in facts:
                    if _cap():
                        break
                    _add(fact)

        # 4. Preferencias sob consulta explicita sobre o usuario
        if is_memory_query and not _cap():
            try:
                preferences = manager.database.find_active_preferences()
            except Exception:
                preferences = []
            for fact in preferences:
                if _cap():
                    break
                _add(fact)
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
