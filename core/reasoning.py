from dataclasses import dataclass


@dataclass
class ReasoningResult:
    intent: str
    action: str
    confidence: float
    target_module: str
    reasoning_log: str
    memory_operation: str = "none"
    existing_memory_id: int | None = None
    matching_fact_id: int | None = None


class ReasoningEngine:
    """
    Responsável por interpretar a intenção do usuário,
    cruzar dados do contexto e decidir o fluxo de execução.
    """

    def __init__(self):
        pass

    def analyze_relevant_memories(self, relevant_memories):
        analysis = {"likes": [], "dislikes": [], "other": []}

        for fact in relevant_memories or []:
            if fact.relation == "like":
                analysis["likes"].append(fact)
            elif fact.relation == "dislike":
                analysis["dislikes"].append(fact)
            else:
                analysis["other"].append(fact)

        return analysis

    def compare_memory_facts(self, current_facts, relevant_memories):
        comparisons = []
        current_facts = current_facts or []
        relevant_memories = relevant_memories or []

        for current in current_facts:
            for previous in relevant_memories:
                if current.target.lower() != previous.target.lower():
                    continue

                same_fact = (
                    current.relation == previous.relation
                    and current.emotion == previous.emotion
                    and current.temporal_context == previous.temporal_context
                    and current.negation == previous.negation
                    and current.value == previous.value
                )
                if same_fact:
                    comparisons.append({
                        "target": current.target,
                        "status": "confirmation",
                        "current": current,
                        "previous": previous,
                    })
                elif current.relation == previous.relation:
                    comparisons.append({
                        "target": current.target,
                        "status": "update",
                        "current": current,
                        "previous": previous,
                    })
                elif {
                    current.relation,
                    previous.relation,
                } == {"like", "dislike"}:
                    comparisons.append({
                        "target": current.target,
                        "status": "contradiction",
                        "current": current,
                        "previous": previous,
                    })

        return comparisons

    def evaluate_input(
        self,
        user_input: str,
        memory_context=None,
        emotions=None,
        relevant_memories=None,
        current_facts=None,
        existing_memory=None,
        matching_fact=None,
    ) -> ReasoningResult:
        input_lower = user_input.lower()
        relevant_memories = relevant_memories or []
        current_facts = current_facts or []
        memory_facts = memory_context.get("facts", []) if memory_context else []

        if any(
            isinstance(fact, dict) and fact.get("conflict", False)
            for fact in memory_facts
        ):
            return ReasoningResult(
                intent="memory_update",
                action="update_memory",
                confidence=0.95,
                target_module="Memory",
                reasoning_log="Foi detectada alteração em um MemoryFact existente.",
                memory_operation="update",
            )

        comparisons = self.compare_memory_facts(
            current_facts,
            relevant_memories,
        )
        contradictions = [
            item for item in comparisons
            if item["status"] == "contradiction"
        ]
        confirmations = [
            item for item in comparisons
            if item["status"] == "confirmation"
        ]
        changes = [
            item for item in comparisons
            if item["status"] == "update"
        ]

        if contradictions:
            previous = contradictions[0]["previous"]
            return ReasoningResult(
                intent="memory_update",
                action="update_memory",
                confidence=0.95,
                target_module="Memory",
                reasoning_log="Foi detectada contradição em uma memória existente.",
                memory_operation="update",
                existing_memory_id=getattr(existing_memory, "id", None),
                matching_fact_id=getattr(previous, "id", None),
            )

        if changes:
            previous = changes[0]["previous"]
            return ReasoningResult(
                intent="memory_update",
                action="update_memory",
                confidence=0.95,
                target_module="Memory",
                reasoning_log="Foi detectada alteração nos campos de um fato existente.",
                memory_operation="update",
                existing_memory_id=getattr(existing_memory, "id", None),
                matching_fact_id=getattr(previous, "id", None),
            )

        if current_facts and existing_memory:
            existing_targets = {
                fact.target.lower()
                for fact in existing_memory.facts
            }
            if any(
                fact.target.lower() not in existing_targets
                for fact in current_facts
            ):
                return ReasoningResult(
                    intent="memory_extension",
                    action="add_memory_fact",
                    confidence=0.93,
                    target_module="Memory",
                    reasoning_log="Foi detectado um novo fato relacionado a uma memória existente.",
                    memory_operation="add",
                )

            if confirmations:
                targets = ", ".join(item["target"] for item in confirmations)
                previous = confirmations[0]["previous"]
                return ReasoningResult(
                    intent="memory_confirmation",
                    action="reinforce_memory",
                    confidence=0.98,
                    target_module="Memory",
                    reasoning_log=(
                        f"A nova informação reforça memórias existentes sobre: {targets}."
                    ),
                    memory_operation="reinforce",
                    existing_memory_id=getattr(existing_memory, "id", None),
                    matching_fact_id=getattr(previous, "id", None),
                )

            return ReasoningResult(
                intent="known_memory",
                action="ignore",
                confidence=0.98,
                target_module="Memory",
                reasoning_log="Os fatos recebidos já existem na memória.",
                memory_operation="ignore",
                existing_memory_id=getattr(existing_memory, "id", None),
                matching_fact_id=getattr(matching_fact, "id", None),
            )

        if current_facts and existing_memory is None:
            return ReasoningResult(
                intent="memory_storage",
                action="create_memory",
                confidence=0.9,
                target_module="Memory",
                reasoning_log="Detectado fato novo sem memoria existente; se crea.",
                memory_operation="create",
            )

        if any(keyword in input_lower for keyword in (
            "pesquisar", "buscar", "procurar no google"
        )):
            return ReasoningResult(
                intent="web_search",
                action="open_browser",
                confidence=0.92,
                target_module="Browser",
                reasoning_log="Detectada intenção de busca externa.",
            )

        if any(keyword in input_lower for keyword in (
            "gosto", "odeio", "adoro", "lembre", "favorito"
        )):
            return ReasoningResult(
                intent="preference_storage",
                action="update_memory",
                confidence=0.95,
                target_module="Memory",
                reasoning_log="Detectada preferência pessoal relevante para persistência.",
                memory_operation="create",
            )

        if any(keyword in input_lower for keyword in (
            "abra", "executar", "rodar", "terminal"
        )):
            return ReasoningResult(
                intent="system_execution",
                action="execute_terminal",
                confidence=0.88,
                target_module="Terminal/Windows",
                reasoning_log="Detectado comando de execução de sistema.",
            )

        return ReasoningResult(
            intent="conversational",
            action="respond_via_personality",
            confidence=0.70,
            target_module="Personality",
            reasoning_log="Nenhum gatilho operacional encontrado.",
        )

    def detect_fact_conflict(self, new_facts, existing_facts):
        for new_fact in new_facts:
            for existing_fact in existing_facts:
                if new_fact.target.lower() != existing_fact.target.lower():
                    continue

                if new_fact.relation != existing_fact.relation:
                    return True

        return False
