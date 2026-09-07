"""Conversation Engine do DaviOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.reasoning import ReasoningEngine, ReasoningResult
from memory.memory import Memory
from memory.memory_interpreter import MemoryInterpreter
from memory.memory_manager import MemoryManager

from brain.intent_classifier import Intent, IntentClassifier
from brain.response_generator import ResponseGenerator


@dataclass
class ConversationContext:
    """Contexto conversacional."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    last_user_message: str = ""
    last_response: str = ""
    current_topic: str = ""
    last_intent: str = ""
    relevant_info: dict[str, Any] = field(default_factory=dict)

    def add_message(self, user_msg: str, response: str, intent: str) -> None:
        self.messages.append(
            {"user": user_msg, "response": response, "intent": intent}
        )
        self.last_user_message = user_msg
        self.last_response = response
        self.last_intent = intent
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_user_message": self.last_user_message,
            "last_response": self.last_response,
            "current_topic": self.current_topic,
            "last_intent": self.last_intent,
            "message_count": len(self.messages),
        }


@dataclass
class ConversationResult:
    """Resultado padronizado de uma interacao."""

    response: str
    intent: str = "conversation"
    memory_action: str = "none"
    memories_used: list[Any] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    should_exit: bool = False


class ConversationEngine:
    """Orquestrador principal da conversacao."""

    EXIT_COMMANDS = {
        "sair", "tchau", "encerrar", "exit", "quit", "bye",
        "ate mais", "falow", "falou", "flw",
    }

    GREETING_COMMANDS = {
        "oi", "ola", "eae", "e ai", "iai", "hello", "hi",
        "opa", "bom dia", "boa tarde", "boa noite",
    }

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        response_generator: Optional[ResponseGenerator] = None,
        cognitive_core=None,
    ):
        self.memory_manager = memory_manager or MemoryManager()
        self.reasoning = ReasoningEngine()
        self.response_generator = response_generator or ResponseGenerator()
        self.cognitive_core = cognitive_core
        self.classifier = IntentClassifier()
        self.context = ConversationContext()

    def process(self, user_input: str) -> ConversationResult:
        """Processa uma mensagem do usuario."""
        text = (user_input or "").strip()
        if not text:
            return ConversationResult(
                response="Nao entendi. Pode repetir?",
                intent="empty_input",
            )

        normalized = text.lower().rstrip("!.?")

        if self._is_exit_command(normalized):
            return ConversationResult(
                response="Ate mais!",
                intent="exit",
                should_exit=True,
            )

        if self._is_greeting(normalized):
            response = self.response_generator.generate_greeting()
            self.context.add_message(text, response, "greeting")
            return ConversationResult(
                response=response,
                intent="greeting",
                context=self.context.to_dict(),
            )

        intent = self.classifier.classify(text)

        if intent == Intent.COMMAND:
            response = (
                "Essa funcao ainda nao esta disponivel. Por enquanto eu "
                "converso, lembro de informacoes e respondo perguntas."
            )
            self.context.add_message(text, response, "command")
            return ConversationResult(
                response=response,
                intent="command",
                context=self.context.to_dict(),
            )

        if intent == Intent.MEMORY_QUERY:
            return self._handle_question(text)

        if intent in (Intent.GENERAL_QUESTION, Intent.OPINION):
            return self._handle_llm_question(text, intent)

        if intent == Intent.MEMORY_STATEMENT:
            return self._handle_statement(text)

        # CONVERSATION/UNKNOWN: statement com possiveis fatos ou papo aberto
        return self._handle_statement(text)

    def reset_context(self) -> None:
        self.context = ConversationContext()

    def _is_exit_command(self, normalized: str) -> bool:
        return normalized in self.EXIT_COMMANDS

    def _is_greeting(self, normalized: str) -> bool:
        return normalized in self.GREETING_COMMANDS

    def _is_question(self, text_lower: str) -> bool:
        if text_lower.endswith("?"):
            return True
        starters = {
            "qual", "quem", "o que", "oque", "como", "onde",
            "quando", "por que", "porque", "quanto", "quanta",
            "quantos", "quantas", "quais",
        }
        first_word = text_lower.split()[0] if text_lower else ""
        return first_word in starters

    def _handle_question(self, text: str) -> ConversationResult:
        response, memories_used = self.response_generator.answer_question(
            text, self.memory_manager, self.context
        )
        self.context.add_message(text, response, "question")
        return ConversationResult(
            response=response,
            intent="question",
            memory_action="recall",
            memories_used=memories_used,
            context=self.context.to_dict(),
        )

    def _handle_llm_question(self, text: str, intent: Intent) -> ConversationResult:
        """Perguntas abertas/opiniao: LLM quando disponivel, regras caso contrario."""
        if self.cognitive_core is not None and self._llm_ready():
            try:
                outcome = self.cognitive_core.generate_response(
                    text, context=self.context, intent=intent
                )
                response = outcome["text"]
                memories_used = outcome.get("memories_used", [])
                self.context.add_message(text, response, intent.value)
                return ConversationResult(
                    response=response,
                    intent=intent.value,
                    memory_action="llm_context",
                    memories_used=memories_used,
                    context=self.context.to_dict(),
                )
            except Exception as exc:
                import logging

                logging.getLogger("davios.conversation").warning(
                    "[LLM] falha ao gerar resposta: %s", exc
                )
                response = (
                    "Tive um problema ao consultar meu modelo local agora. "
                    "Posso tentar de novo em instantes?"
                )
                self.context.add_message(text, response, intent.value)
                return ConversationResult(
                    response=response,
                    intent=intent.value,
                    memory_action="llm_error",
                    context=self.context.to_dict(),
                )

        # Sem LLM: mantem o comportamento anterior de regras.
        # Com provider LOCAL real e sem modelo instalado, explica como instalar.
        from brain.providers.local_llm_provider import LLAMA_CPP_INSTRUCTIONS, LocalLLMProvider

        if isinstance(getattr(self.cognitive_core, "llm_provider", None), LocalLLMProvider):
            self.context.add_message(text, "modelo ausente", intent.value)
            return ConversationResult(
                response=LLAMA_CPP_INSTRUCTIONS,
                intent=intent.value,
                memory_action="llm_unavailable",
                context=self.context.to_dict(),
            )
        return self._handle_question(text)

    def _llm_ready(self) -> bool:
        try:
            return bool(self.cognitive_core.llm_provider.is_available())
        except Exception:
            return False


    def _handle_statement(self, text: str) -> ConversationResult:
        intent = self.classifier.classify(text)
        context = self.memory_manager.context_analyzer.analyze(text)
        emotions = self._build_emotions(context)
        interpretation = MemoryInterpreter().interpret(text, context, emotions)

        if not interpretation["facts"]:
            # Papo aberto / frase sem fato memoravel: LLM se disponivel
            if self.cognitive_core is not None and self._llm_ready():
                try:
                    outcome = self.cognitive_core.generate_response(
                        text, context=self.context, intent=intent
                    )
                    response = outcome["text"]
                    self.context.add_message(text, response, intent.value)
                    return ConversationResult(
                        response=response,
                        intent=intent.value,
                        memory_action="llm_context",
                        memories_used=outcome.get("memories_used", []),
                        context=self.context.to_dict(),
                    )
                except Exception:
                    pass  # cai no fallback de regras abaixo
            response = self.response_generator.generate_fallback(text, self.context)
            self.context.add_message(text, response, "conversation")
            return ConversationResult(
                response=response,
                intent="conversation",
                memory_action="none",
                context=self.context.to_dict(),
            )

        memory = Memory(
            content=text,
            memory_type="preference" if interpretation.get("memory_candidate") else "episodic",
            importance=0.7,
            emotion="neutral",
            emotional_intensity=0.0,
            facts=interpretation["facts"],
        )

        match = self.memory_manager.match_memory_for_facts(memory.facts)
        decision = self.reasoning.evaluate_input(
            user_input=text,
            memory_context=context,
            emotions=emotions,
            relevant_memories=[match["matching_fact"]] if match["matching_fact"] else [],
            current_facts=memory.facts,
            existing_memory=match["existing_memory"],
            matching_fact=match["matching_fact"],
        )

        result = self.memory_manager.apply_memory_operation(
            reasoning_result=decision,
            new_memory=memory,
            existing_memory=match["existing_memory"],
        )

        response = self.response_generator.generate_from_memory_result(
            result, decision, text, self.context
        )

        self.context.add_message(text, response, decision.intent)
        if memory.facts:
            self.context.current_topic = memory.facts[0].target

        return ConversationResult(
            response=response,
            intent=decision.intent,
            memory_action=result.get("action", "none"),
            memories_used=[match["existing_memory"]] if match["existing_memory"] else [],
            context=self.context.to_dict(),
        )

    def _build_emotions(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        emotions = []
        for part in context.get("parts", []):
            emotion = self.memory_manager.emotion_analyzer.analyze(
                part["text"], part
            )
            emotions.append(
                {
                    "text": part["text"],
                    "emotion": emotion["emotion"],
                    "emotional_intensity": emotion["emotional_intensity"],
                    "context": part,
                }
            )
        return emotions
