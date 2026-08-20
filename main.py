from core.system import System
from memory.memory import Memory


def main():

    # ============================================================
    # 1. INICIALIZA O SISTEMA
    # ============================================================

    system = System()

    memory_manager = getattr(system, "memory_manager", None)

    if not memory_manager:
        from memory.memory_manager import MemoryManager
        memory_manager = MemoryManager()

    importance_analyzer = getattr(
        system,
        "importance_analyzer",
        None
    )

    if not importance_analyzer:
        from memory.ImportanceAnalyzer import ImportanceAnalyzer
        importance_analyzer = ImportanceAnalyzer()

    reasoning_engine = getattr(
        system,
        "reasoning_engine",
        None
    )

    if not reasoning_engine:
        from core.reasoning import ReasoningEngine
        reasoning_engine = ReasoningEngine()

    memory_interpreter = getattr(
        system,
        "memory_interpreter",
        None
    )

    if not memory_interpreter:
        from memory.memory_interpreter import MemoryInterpreter
        memory_interpreter = MemoryInterpreter()

    # ============================================================
    # 2. ENTRADA ATUAL
    # ============================================================

    user_input = "agora eu gosto de pizza"

    # ============================================================
    # 3. CONTEXTO E EMOÇÕES
    # ============================================================

    context = memory_manager.context_analyzer.analyze(
        user_input
    )

    emotions = []

    for part in context.get("parts", []):
        emotion = memory_manager.emotion_analyzer.analyze(
            part["text"],
            part,
        )

        emotions.append(
            {
                "text": part["text"],
                "emotion": emotion["emotion"],
                "emotional_intensity": emotion["emotional_intensity"],
                "context": part,
            }
        )
    # ============================================================
    # 4. INTERPRETA A ENTRADA
    # ============================================================

    memory_analysis = memory_interpreter.interpret(
        user_input,
        context,
        emotions,
    )

    # ============================================================
    # 5. DEFINE O TIPO DA NOVA MEMÓRIA
    # ============================================================

    memory_type = (
        "preference"
        if memory_analysis.get(
            "memory_candidate",
            False
        )
        else "episodic"
    )

    # ============================================================
    # 6. CRIA A NOVA MEMÓRIA SOMENTE EM RAM
    #
    # IMPORTANTE:
    # Ainda não existe nenhuma alteração no banco.
    # ============================================================

    new_memory = Memory(
        content=user_input,
        memory_type=memory_type,
        importance=0.0,
        emotion="neutral",
        emotional_intensity=0.0,
    )

    new_memory.facts = memory_analysis.get(
        "facts",
        []
    )

    # ============================================================
    # 7. CALCULA A IMPORTÂNCIA
    # ============================================================

    new_memory.importance = (
        importance_analyzer.analyze(
            new_memory
        )
    )

    # ============================================================
    # 8. PROCURA A MEMÓRIA EXISTENTE
    #
    # A nova memória ainda está SOMENTE EM RAM.
    #
    # Procuramos no banco usando os MemoryFacts.
    # ============================================================

    existing_memory = (
        memory_manager.find_memory_for_facts(
            new_memory.facts
        )
    )

    # ============================================================
    # 9. MEMÓRIAS RELEVANTES
    #
    # Em vez de usar recall_relevant(), utilizamos diretamente
    # os fatos da memória encontrada.
    # ============================================================

    relevant_facts = []

    if existing_memory is not None:
        relevant_facts = [
            fact
            for fact in existing_memory.facts
            if any(
                current_fact.target.lower()
                == fact.target.lower()
                for current_fact in new_memory.facts
            )
        ]

    print("=== EXISTING MEMORY ===")
    print(existing_memory)

    print("=== RELEVANT MEMORY ===")
    print(relevant_facts)

    # ============================================================
    # 10. RACIOCÍNIO
    # ============================================================

    decision_reasoning = (
        reasoning_engine.evaluate_input(
            user_input,
            memory_context=context,
            emotions=emotions,
            relevant_memories=relevant_facts,
            current_facts=new_memory.facts,
            existing_memory=existing_memory,
        )
    )

    print("=== REASONING ===")

    print(
        f"INTENT: "
        f"{decision_reasoning.intent}"
    )

    print(
        f"ACTION: "
        f"{decision_reasoning.action}"
    )

    print(
        f"OPERATION: "
        f"{decision_reasoning.memory_operation}"
    )

    print(
        f"CONFIDENCE: "
        f"{decision_reasoning.confidence}"
    )

    print(
        f"REASONING: "
        f"{decision_reasoning.reasoning_log}"
    )

    # ============================================================
    # 11. MOSTRA O HISTÓRICO ANTES DA OPERAÇÃO
    # ============================================================

    if existing_memory is not None:

        history = (
            memory_manager.database
            .get_memory_fact_history(
                existing_memory.id
            )
        )

        print("=== MEMORY FACT HISTORY ===")

        for item in history:
            print(item)

    # ============================================================
    # 12. EXECUTA A OPERAÇÃO DECIDIDA
    #
    # SOMENTE AQUI o banco pode ser alterado.
    # ============================================================

    result = (
        memory_manager.apply_memory_operation(
            reasoning_result=decision_reasoning,
            new_memory=new_memory,
            existing_memory=existing_memory,
        )
    )

    print(
        "=== MEMORY OPERATION RESULT ==="
    )

    print(result)

    # ============================================================
    # 13. BOOT
    # ============================================================

    system.boot()


if __name__ == "__main__":
    main()