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
        new_facts: Optional[list[Any]] = None,
        style_profile: Optional[Any] = None,
    ) -> BuiltPrompt:
        """Monta o prompt final com memórias relevantes e histórico recente.

        ``new_facts`` são fatos extraídos da própria mensagem atual (já
        salvos na memória) — entram em bloco próprio para que o modelo use
        a informação recém-fornecida.

        ``style_profile`` é o perfil de estilo de fala do usuário. Quando
        fornecido e há dados suficientes, sua descrição é incluida no texto
        de sistema para orientar o modelo a responder no mesmo registro.
        """
        memory_block = self._format_memories(memories or [])
        new_facts_block = self._format_new_facts(new_facts or [])
        history_block = self._format_history(context)
        topic = getattr(context, "current_topic", "") if context else ""
        style_text = self._format_style(style_profile)

        sections: list[str] = []
        if memory_block:
            sections.append(f"Informacoes que voce sabe sobre o usuario:\n{memory_block}")
        if new_facts_block:
            sections.append(
                f"Informacoes que o usuario acabou de informar nesta mensagem:\n{new_facts_block}"
            )
        if history_block:
            sections.append(f"Conversa recente:\n{history_block}")
        if topic:
            sections.append(f"Assunto atual da conversa: {topic}")
        if resolved_context:
            sections.append(f"Referencia a mensagem anterior:\n{resolved_context}")
        sections.append(f"Mensagem do usuario: {user_message}")

        prompt = "\n\n".join(sections)
        return BuiltPrompt(
            system=self._system_text(style_text),
            prompt=prompt,
            metadata={
                "intent": intent,
                "memories_count": len(memories or []),
                "new_facts_count": len(new_facts or []),
                "history_messages": len(getattr(context, "messages", []) or []),
                "style_applied": bool(style_text),
            },
        )

    def _system_text(self, style_text: str = "") -> str:
        config_system = (self.config.system_prompt or "").strip()
        parts = [
            config_system,
            self.personality.to_system_text(),
            (style_text or "").strip(),
        ]
        return "\n\n".join(part for part in parts if part)

    @staticmethod
    def _format_style(style_profile: Optional[Any]) -> str:
        """Extrai a descricao de estilo do perfil, se houver dados suficientes.

        Retorna "" quando nao ha perfil ou o proprio perfil decide que ainda
        nao ha dados suficientes. Um perfil quebrado nunca quebra a geracao.
        """
        if style_profile is None:
            return ""
        try:
            if hasattr(style_profile, "to_instruction"):
                text = style_profile.to_instruction()
            else:
                text = style_profile.to_prompt_description()
        except Exception:
            return ""
        return text.strip() if isinstance(text, str) else ""

    def _format_fact_line(self, fact: Any) -> Optional[str]:
        """Formata um MemoryFact como linha legivel para o modelo."""
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
            return f"- Nome do usuario: {value}"
        if not target or not relation:
            return None
        if relation in ("like", "love"):
            state = "nao gosta" if negation else "gosta"
            return f"- O usuario {state} de {target}"
        if relation in ("dislike", "hate"):
            state = "nao gosta" if negation else "nao gosta"
            return f"- O usuario {state} de {target}"
        if relation == "identity":
            return f"- {target}: {value or 'desconhecido'}"
        if relation == "working_on":
            state = "nao esta trabalhando em" if negation else "esta trabalhando em"
            return f"- O usuario {state} {target}"
        if relation == "studies":
            state = "nao esta estudando" if negation else "esta estudando"
            return f"- O usuario {state} {target}"
        if relation == "lives_in":
            return f"- O usuario mora em {target}"
        if relation == "has":
            return f"- O usuario tem {target}"
        return f"- {target}: {relation}"

    def _format_memories(self, memories: list[Any]) -> str:
        lines: list[str] = []
        for fact in memories:
            line = self._format_fact_line(fact)
            if line:
                lines.append(line)
        return "\n".join(lines[: self.config.max_memories_in_prompt])

    def _format_new_facts(self, facts: list[Any]) -> str:
        lines: list[str] = []
        for fact in facts:
            line = self._format_fact_line(fact)
            if line:
                lines.append(line)
        return "\n".join(lines)

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
