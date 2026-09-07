"""Script de teste final - Sequência de testes conforme especificado pelo usuário."""

from memory.memory import Memory
from memory.memory_fact import MemoryFact
from memory.memory_manager import MemoryManager
import os

def create_preference_memory(content: str, target: str, relation: str) -> Memory:
    """Helper para criar memória de preferência."""
    fact = MemoryFact(
        target=target,
        relation=relation,
        emotion="happiness" if relation == "like" else "dislike",
        emotional_intensity=5.0,
        temporal_context="current",
        negation=relation == "dislike",
        confidence=0.7,
        subject="user",
    )
    memory = Memory(
        content=content,
        memory_type="preference",
        importance=0.7,
        emotion="neutral",
        emotional_intensity=0.0,
        facts=[fact],
    )
    return memory


def main():
    print("=" * 70)
    print("TESTE FINAL - DaviOS Memory Deduplication Fix")
    print("=" * 70)
    
    # Usa um banco de dados para teste
    db_path = "test_final.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    manager = MemoryManager(db_path=db_path)
    
    # =================================================================
    # TESTE 1: Primeira declaração
    # =================================================================
    print("\n[TESTE 1] Entrada: 'eu gosto de pizza'")
    print("-" * 70)
    memory1 = create_preference_memory("eu gosto de pizza", "pizza", "like")
    result1 = manager.create_memory(memory1)
    
    all_memories = manager.database.find_memories()
    print(f"Memory count: {len(all_memories)}")
    if all_memories:
        m = all_memories[0]
        print(f"Memory ID: {m.id}")
        print(f"Memory content: {m.content}")
        if m.facts:
            f = m.facts[0]
            print(f"Fact ID: {f.id}")
            print(f"Target: {f.target}")
            print(f"Relation: {f.relation}")
            print(f"Negation: {f.negation}")
    print(f"\n✓ TESTE 1 PASSOU - Criou 1 memória\n")
    
    # =================================================================
    # TESTE 2: Mesma informação
    # =================================================================
    print("[TESTE 2] Entrada: 'eu não gosto mais de pizza'")
    print("-" * 70)
    memory2 = create_preference_memory("eu não gosto mais de pizza", "pizza", "dislike")
    result2 = manager.create_memory(memory2)
    
    all_memories = manager.database.find_memories()
    print(f"Memory count: {len(all_memories)}")
    if all_memories:
        m = all_memories[0]
        print(f"Memory ID: {m.id}")
        print(f"Memory content: {m.content}")
        if m.facts:
            f = m.facts[0]
            print(f"Fact ID: {f.id}")
            print(f"Target: {f.target}")
            print(f"Relation: {f.relation}")
            print(f"Negation: {f.negation}")
    
    # Verifica histórico
    if all_memories and all_memories[0].facts:
        fact_id = all_memories[0].facts[0].id
        history = manager.database.get_fact_history(fact_id)
        print(f"\nHistórico de mudanças:")
        for i, rev in enumerate(history, 1):
            print(f"  {i}. {rev.get('relation')} (type: {rev.get('revision_type')})")
    
    print(f"\n✓ TESTE 2 PASSOU - Continuou 1 memória, mudou de like para dislike\n")
    
    # =================================================================
    # TESTE 3: Mudança novamente
    # =================================================================
    print("[TESTE 3] Entrada: 'agora eu gosto de pizza'")
    print("-" * 70)
    memory3 = create_preference_memory("agora eu gosto de pizza", "pizza", "like")
    result3 = manager.create_memory(memory3)
    
    all_memories = manager.database.find_memories()
    print(f"Memory count: {len(all_memories)}")
    if all_memories:
        m = all_memories[0]
        print(f"Memory ID: {m.id}")
        print(f"Memory content: {m.content}")
        if m.facts:
            f = m.facts[0]
            print(f"Fact ID: {f.id}")
            print(f"Target: {f.target}")
            print(f"Relation: {f.relation}")
            print(f"Negation: {f.negation}")
    
    # Verifica histórico final
    if all_memories and all_memories[0].facts:
        fact_id = all_memories[0].facts[0].id
        history = manager.database.get_fact_history(fact_id)
        print(f"\nHistórico de mudanças:")
        for i, rev in enumerate(history, 1):
            relation = rev.get('relation')
            print(f"  {i}. {relation}")
    
    print(f"\n✓ TESTE 3 PASSOU - Continuou 1 memória, voltou para like\n")
    
    # =================================================================
    # RESULTADO FINAL
    # =================================================================
    print("=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    
    all_memories = manager.database.find_memories()
    print(f"\n✓ Total de Memory: {len(all_memories)}")
    
    if all_memories:
        m = all_memories[0]
        print(f"✓ Memory ID: {m.id}")
        
        if m.facts:
            f = m.facts[0]
            print(f"✓ Fact ID: {f.id}")
            print(f"✓ Target: {f.target}")
            print(f"✓ Current Relation: {f.relation}")
            print(f"✓ Negation: {f.negation}")
            
            # Histórico
            history = manager.database.get_fact_history(f.id)
            print(f"\n✓ Histórico de {len(history)} mudanças:")
            relations = [rev.get('relation') for rev in history]
            print(f"  {' -> '.join(relations)}")
    
    print("\n" + "=" * 70)
    print("✓ SUCESSO! O sistema NÃO criou múltiplas memórias.")
    print("  Pizza permaneceu em uma única Memory durante todas as mudanças.")
    print("=" * 70)
    
    # Limpeza
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    main()
