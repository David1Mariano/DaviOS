"""Response Generator do DaviOS."""

from __future__ import annotations

import re
from typing import Any

from core.reasoning import ReasoningResult
from memory.memory_manager import MemoryManager


class ResponseGenerator:
    """Gera respostas baseadas no resultado da operacao de memoria."""

    def generate_greeting(self) -> str:
        return "Ola! Como posso ajudar?"

    def generate_fallback(self, text: str, context: Any) -> str:
        return "Entendido."

    def generate_from_memory_result(
        self,
        result: dict[str, Any],
        decision: ReasoningResult,
        user_input: str,
        context: Any,
    ) -> str:
        action = result.get("action", "none")
        if action == "create":
            return self._respond_to_create(result)
        if action == "update":
            return self._respond_to_update(result)
        if action == "add":
            return self._respond_to_add(result)
        if action == "reinforce":
            return self._respond_to_reinforce(result)
        if action == "ignore":
            return "Certo."
        return "Entendido."

    def _first_fact(self, result: dict[str, Any]):
        """Localiza o fato mais relevante da operacao, preferindo o alvo correto."""
        memory = result.get("memory")
        target = result.get("target")
        fact_id = result.get("fact_id")
        reported = result.get("facts") or []

        candidates = memory.facts if memory and memory.facts else []

        # Alvo re-transformado pelo proprio gerenciador (update/add/reinforce)
        if isinstance(reported, list) and reported:
            first = reported[0]
            if isinstance(first, dict) and first.get("target"):
                rt = first["target"]
                for f in candidates:
                    if (f.target or "").casefold() == str(rt).casefold():
                        return f

        if target:
            for f in candidates:
                if (f.target or "").casefold() == str(target).casefold():
                    return f
        if fact_id:
            for f in candidates:
                if f.id == fact_id:
                    return f
        return candidates[0] if candidates else None

    def _fact_target(self, result: dict[str, Any]) -> Optional[str]:
        fact = self._first_fact(result)
        if fact:
            return fact.target
        return (result.get("facts") or [{}])[0].get("target") if result.get("facts") else None

    @staticmethod
    def _capitalize_target(target: str) -> str:
        """Capitaliza o alvo para exibicao na resposta."""
        if not target:
            return target
        return target[0].upper() + target[1:]

    def _respond_to_create(self, result: dict[str, Any]) -> str:
        fact = self._first_fact(result)
        if not fact:
            return "Entendido. Vou lembrar disso."
        target = self._capitalize_target(fact.target)
        if fact.relation == "like":
            return f"Entendido. Vou lembrar que voce gosta de {target}."
        if fact.relation == "dislike":
            return f"Entendido. Vou lembrar que voce nao gosta de {target}."
        return "Entendido. Vou lembrar disso."

    def _respond_to_update(self, result: dict[str, Any]) -> str:
        fact = self._first_fact(result)
        if not fact:
            return "Entendido. Atualizei essa informacao."
        target = self._capitalize_target(fact.target)
        if fact.relation == "like":
            return f"Entendido. Agora voce gosta de {target}."
        if fact.relation == "dislike":
            return f"Entendido. Agora voce nao gosta mais de {target}."
        return "Entendido. Atualizei essa informacao."

    def _respond_to_reinforce(self, result: dict[str, Any]) -> str:
        fact = self._first_fact(result)
        if not fact:
            return "Certo."
        target = self._capitalize_target(fact.target)
        if fact.relation == "like":
            return f"Certo. Voce gosta de {target}."
        if fact.relation == "dislike":
            return f"Certo. Voce nao gosta de {target}."
        return "Certo."

    def _respond_to_add(self, result: dict[str, Any]) -> str:
        facts = result.get("facts") or []
        if not facts:
            return "Entendido. Vou lembrar disso."
        fact = facts[0]
        target = self._capitalize_target(fact.get("target", ""))
        relation = fact.get("relation")
        if relation == "like" and target:
            return f"Entendido. Vou lembrar que voce gosta de {target}."
        if relation == "dislike" and target:
            return f"Entendido. Vou lembrar que voce nao gosta de {target}."
        return "Entendido. Vou lembrar disso."

    def answer_question(
        self,
        text: str,
        memory_manager: MemoryManager,
        context: Any,
    ) -> tuple[str, list[Any]]:
        """Responde perguntas consultando a memoria."""
        text_lower = text.lower()

        if "nome" in text_lower and ("meu" in text_lower or "minha" in text_lower):
            return self._answer_about_name(memory_manager, text_lower)

        if "gosto" in text_lower and any(
            w in text_lower for w in ("do que", "o que", "que", "de que")
        ):
            return self._answer_about_preferences(memory_manager)

        match = re.search(r"eu gosto de (\w+)", text_lower)
        if match:
            target = match.group(1)
            return self._answer_confirm_preference(memory_manager, target)

        return "Nao tenho essa informacao ainda.", []

    def _answer_about_name(
        self, memory_manager: MemoryManager, text_lower: str
    ) -> tuple[str, list[Any]]:
        facts = memory_manager.database.find_facts_by_target("nome")
        if facts:
            fact = facts[0]
            if fact.value:
                return f"Seu nome e {self._capitalize_target(fact.value)}.", [fact]
            evidences = memory_manager.database.get_fact_evidence(fact.id)
            for ev in evidences:
                content = ev.get("content", "").lower()
                name_match = re.search(r"meu nome e (\w+)", content)
                if name_match:
                    return f"Seu nome e {name_match.group(1).capitalize()}.", [fact]
            return f"Seu nome e {fact.target}.", [fact]
        return "Ainda nao sei seu nome. Pode me dizer?", []

    def _answer_about_preferences(
        self, memory_manager: MemoryManager
    ) -> tuple[str, list[Any]]:
        likes = [
            f
            for f in memory_manager.database.find_active_preferences("user", "preference")
            if f.relation == "like"
        ]
        if likes:
            targets = ", ".join(self._capitalize_target(f.target) for f in likes)
            return f"Voce gosta de {targets}.", likes
        return "Ainda nao sei do que voce gosta.", []

    def _answer_confirm_preference(
        self, memory_manager: MemoryManager, target: str
    ) -> tuple[str, list[Any]]:
        facts = memory_manager.database.find_facts_for_semantic_key(
            "user", target, "preference", include_inactive=False
        )
        if facts:
            fact = facts[0]
            pretty = self._capitalize_target(fact.target)
            if fact.relation == "like":
                return f"Sim. Voce gosta de {pretty}.", [fact]
            if fact.relation == "dislike":
                return (
                    f"Nao. Pelo que voce me contou, voce nao gosta mais de {pretty}.",
                    [fact],
                )
        return f"Nao tenho informacao sobre {target}.", []
