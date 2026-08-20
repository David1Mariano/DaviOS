from memory.database import Database
from memory.emotion_analyzer import EmotionAnalyzer
from memory.context_analyzer import ContextAnalyzer


class MemoryManager:

    def __init__(self):

        self.database = Database()

        self.emotion_analyzer = EmotionAnalyzer()
        self.context_analyzer = ContextAnalyzer()

    # ============================================================
    # RECALL
    # ============================================================

    def recall(self, target=None):

        if target is not None:
            return self.database.find_facts_by_target(target)

        return self.database.find_memories()

    # ============================================================
    # FIND MEMORY
    # ============================================================

    def find_memory_for_facts(self, facts):

        if not facts:
            return None

        memories = self.database.find_memories()

        for memory in memories:

            self.normalize_memory_facts(memory)

            for current_fact in facts:

                current_target = self.normalize_fact_target(
                    current_fact.target
                )

                if not current_target:
                    continue

                for existing_fact in memory.facts:

                    existing_target = self.normalize_fact_target(
                        existing_fact.target
                    )

                    if current_target == existing_target:
                        return memory

        return None

    # ============================================================
    # NORMALIZAÇÃO
    # ============================================================

    def normalize_memory_facts(self, memory):

        seen = set()
        normalized_facts = []

        for fact in memory.facts:

            target = self.normalize_fact_target(
                fact.target
            )

            if not target:
                continue

            fact.target = target

            key = (
                target.lower(),
            )

            if key in seen:
                continue

            seen.add(key)
            normalized_facts.append(fact)

        memory.facts = normalized_facts

        return memory

    def normalize_fact_target(self, target):

        if target is None:
            return ""

        words = target.lower().split()

        temporal_words = {
            "agora",
            "atualmente",
            "hoje",
            "antes",
            "antigamente",
        }

        filtered = [
            word
            for word in words
            if word not in temporal_words
        ]

        return " ".join(filtered).strip()

    # ============================================================
    # COMPARAÇÃO DE FATOS
    # ============================================================

    def facts_are_equal(
        self,
        existing_fact,
        new_fact,
    ):
        """
        Verifica se dois fatos representam exatamente
        o mesmo estado conhecido.
        """

        existing_target = self.normalize_fact_target(
            existing_fact.target
        )

        new_target = self.normalize_fact_target(
            new_fact.target
        )

        if existing_target != new_target:
            return False

        if existing_fact.relation != new_fact.relation:
            return False

        if existing_fact.emotion != new_fact.emotion:
            return False

        if (
            existing_fact.emotional_intensity
            != new_fact.emotional_intensity
        ):
            return False

        if (
            existing_fact.temporal_context
            != new_fact.temporal_context
        ):
            return False

        if (
            bool(existing_fact.negation)
            != bool(new_fact.negation)
        ):
            return False

        return True

    # ============================================================
    # LOCALIZAÇÃO DE FATOS
    # ============================================================

    def find_matching_fact(
        self,
        memory,
        new_fact,
    ):

        new_target = self.normalize_fact_target(
            new_fact.target
        )

        if not new_target:
            return None

        for existing_fact in memory.facts:

            existing_target = self.normalize_fact_target(
                existing_fact.target
            )

            if existing_target == new_target:
                return existing_fact

        return None

    # ============================================================
    # SINCRONIZAÇÃO DE FATOS
    # ============================================================

    def synchronize_memory_facts(
        self,
        existing_memory,
        new_memory,
    ):
        """
        Sincroniza os fatos de uma memória existente
        com os fatos da nova entrada.

        Esta é a única camada responsável por:
        - adicionar fatos;
        - atualizar fatos;
        - ignorar fatos idênticos;
        - atualizar o conteúdo da memória.
        """

        if existing_memory is None:
            return []

        if new_memory is None:
            return []

        results = []

        content_changed = False

        for new_fact in new_memory.facts:

            new_fact.target = self.normalize_fact_target(
                new_fact.target
            )

            if not new_fact.target:
                continue

            matching_fact = self.find_matching_fact(
                existing_memory,
                new_fact,
            )

            # ====================================================
            # FATO NOVO
            # ====================================================

            if matching_fact is None:

                added = self.database.add_memory_fact(
                    memory_id=existing_memory.id,
                    target=new_fact.target,
                    relation=new_fact.relation,
                    emotion=new_fact.emotion,
                    emotional_intensity=new_fact.emotional_intensity,
                    temporal_context=new_fact.temporal_context,
                    negation=new_fact.negation,
                )

                if added:

                    existing_memory.facts.append(
                        new_fact
                    )

                    content_changed = True

                    results.append({
                        "action": "add",
                        "target": new_fact.target,
                        "relation": new_fact.relation,
                    })

                    print(
                        f"[MEMORY] ADD | "
                        f"{new_fact.target} -> "
                        f"{new_fact.relation}"
                    )

                continue

            # ====================================================
            # FATO IDÊNTICO
            # ====================================================

            if self.facts_are_equal(
                matching_fact,
                new_fact,
            ):

                results.append({
                    "action": "ignore",
                    "target": new_fact.target,
                    "relation": new_fact.relation,
                })

                print(
                    f"[MEMORY] IGNORE | "
                    f"{new_fact.target} -> "
                    f"{new_fact.relation}"
                )

                continue

            # ====================================================
            # FATO ALTERADO
            # ====================================================

            old_relation = matching_fact.relation

            updated = self.database.update_memory_fact(
                memory_id=existing_memory.id,
                target=new_fact.target,
                relation=new_fact.relation,
                emotion=new_fact.emotion,
                emotional_intensity=new_fact.emotional_intensity,
                temporal_context=new_fact.temporal_context,
                negation=new_fact.negation,
            )

            if not updated:
                print(
                    f"[MEMORY] ERROR | "
                    f"Falha ao atualizar "
                    f"{new_fact.target}"
                )

                continue

            # Atualiza o objeto em RAM imediatamente.
            matching_fact.target = new_fact.target
            matching_fact.relation = new_fact.relation
            matching_fact.emotion = new_fact.emotion
            matching_fact.emotional_intensity = (
                new_fact.emotional_intensity
            )
            matching_fact.temporal_context = (
                new_fact.temporal_context
            )
            matching_fact.negation = (
                new_fact.negation
            )

            content_changed = True

            results.append({
                "action": "update",
                "target": new_fact.target,
                "old_relation": old_relation,
                "new_relation": new_fact.relation,
            })

            print(
                f"[MEMORY] UPDATE | "
                f"{new_fact.target}: "
                f"{old_relation} -> "
                f"{new_fact.relation}"
            )

        # ========================================================
        # ATUALIZA O TEXTO DA MEMÓRIA
        # ========================================================

        if content_changed:

            self.database.update_memory_content(
                memory_id=existing_memory.id,
                content=new_memory.content,
            )

            existing_memory.content = new_memory.content

            print(
                "[MEMORY] CONTENT UPDATE | "
                f"{new_memory.content}"
            )

        return results

    # ============================================================
    # APLICA OPERAÇÃO DECIDIDA PELO REASONING ENGINE
    # ============================================================

    def apply_memory_operation(
        self,
        reasoning_result,
        new_memory,
        existing_memory=None,
    ):

        if reasoning_result is None:

            print(
                "[MEMORY] ERROR | "
                "Resultado de reasoning inexistente."
            )

            return "ignored"

        operation = (
            reasoning_result.memory_operation
        )

        # ========================================================
        # CREATE
        # ========================================================

        if operation == "create":

            memory_id = self.database.save_memory(
                new_memory
            )

            if memory_id is None:

                print(
                    "[MEMORY] CREATE ignorado | "
                    "Memória já existente."
                )

                return "ignored"

            print(
                "[MEMORY] CREATE | "
                "Nova memória criada."
            )

            return "created"

        # ========================================================
        # ADD
        # ========================================================

        if operation == "add":

            if existing_memory is None:

                print(
                    "[MEMORY] ADD ignorado | "
                    "Memória existente não encontrada."
                )

                return "ignored"

            return self.synchronize_memory_facts(
                existing_memory,
                new_memory,
            )

        # ========================================================
        # UPDATE
        # ========================================================

        if operation == "update":

            if existing_memory is None:

                print(
                    "[MEMORY] UPDATE ignorado | "
                    "Memória existente não encontrada."
                )

                return "ignored"

            return self.synchronize_memory_facts(
                existing_memory,
                new_memory,
            )

        # ========================================================
        # IGNORE
        # ========================================================

        if operation == "ignore":

            print(
                "[MEMORY] IGNORE | "
                "Nenhuma alteração necessária."
            )

            return "ignored"

        # ========================================================
        # OPERAÇÃO DESCONHECIDA
        # ========================================================

        print(
            f"[MEMORY] ERROR | "
            f"Operação desconhecida: {operation}"
        )

        return "unknown"