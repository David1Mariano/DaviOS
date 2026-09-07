"""PromptBuilder do DaviOS.

Monta o contexto enviado ao modelo de linguagem:
    instruções de sistema + personalidade
    + contexto conversacional
    + memórias relevantes
    + mensagem atual

Os prompts vivem aqui (e na config), nunca espalhados pelo código.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from config.davios_config import DaviosConfig
from personality.personality import Personality, DEFAULT_PERSONALITY


@dataclass
class BuiltPrompt:
    """Prompt estruturado pronto para o LLMProvider."""

    system: str
    prompt: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"system": self.system, "prompt": self.prompt, "metadata": self.metadata}


class PromptBuilder:
    """Constrói prompts a partir de contexto, memórias e personalidade."""

    def __init__(
        self,
        config: Optional[DaviosConfig] = None,
        personality: Optional[Personality] = None,
    ):
        self.config = config or DaviosConfig.load()
        self.personality = personality or DEFAULT_PERSONALITY

    def build(
        self,
        user_message: str,
        *,
        context=None,
        memories: Optional[list[Any]] = None,
        intent: str = "conversation",
        resolved_context: Optional[str] = None,
    ) -> BuiltPrompt:
        """Monta o prompt final com memórias relevantes e histórico recente."""
        memory_block = self._format_memories(memories or [])
        history_block = self._format_history(context)
        topic = getattr(context, "current_topic", "") if context else ""

        sections: list[str] = []
        if memory_block:
            sections.append(f"Informacoes que voce sabe sobre o usuario:\n{memory_block}")
        if history_block:
            sections.append(f"Conversa recente:\n{history_block}")
        if topic:
            sections.append(f"Assunto atual da conversa: {topic}")
        if resolved_context:
            sections.append(f"Referencia a mensagem anterior:\n{resolved_context}")
        sections.append(f"Mensagem do usuario: {user_message}")

        prompt = "\n\n".join(sections)
        return BuiltPrompt(
            system=self._system_text(),
            prompt=prompt,
            metadata={
                "intent": intent,
                "memories_count": len(memories or []),
                "history_messages": len(getattr(context, "messages", []) or []),
            },
        )

    def _system_text(self) -> str:
        config_system = (self.config.system_prompt or "").strip()
        return "\n\n".join(
            part
            for part in (config_system, self.personality.to_system_text())
            if part
        )

    def _format_memories(self, memories: list[Any]) -> str:
        lines: list[str] = []
        for fact in memories:
            target = getattr(fact, "target", None)
            if target is None and isinstance(fact, dict):
                target = fact.get("target")
            relation = getattr(fact, "relation", None)
            if relation is None and isinstance(fact, dict):
                relation = fact.get("relation")
            negation = bool(getattr(fact, "negation", False))
            value = getattr(fact, "value", None)
            if value is None and isinstance(fact, dict):
                value = fact.get("value")

            if target == "nome" and value:
                lines.append(f"- Nome do usuario: {value}")
                continue
            if not target or not relation:
                continue
            if relation == "like":
                state = "nao gosta" if negation else "gosta"
                lines.append(f"- O usuario {state} de {target}")
            elif relation == "dislike":
                state = "nao gosta" if negation else "nao gosta"
                lines.append(f"- O usuario {state} de {target}")
            elif relation == "identity":
                lines.append(f"- {target}: {value or 'desconhecido'}")
            else:
                lines.append(f"- {target}: {relation}")

        return "\n".join(lines[: self.config.max_memories_in_prompt])

    def _format_history(self, context) -> str:
        if not context:
            return ""
        messages = getattr(context, "messages", []) or []
        recent = messages[-self.config.max_recent_messages:]
        lines = []
        for item in recent:
            user = item.get("user", "")
            response = item.get("response", "")
            lines.append(f"Usuario: {user}")
            lines.append(f"DaviOS: {response}")
        return "\n".join(lines)
