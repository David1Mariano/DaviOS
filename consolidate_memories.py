"""Script de consolidação segura de memórias duplicadas.

Este script:
1. Identifica memórias duplicadas
2. Escolhe a memória canônica
3. Move evidências
4. Marca duplicatas como inativas (não apaga)
5. Valida o resultado
"""

import os
import shutil
from datetime import datetime
from memory.memory_manager import MemoryManager


def main():
    db_path = "memory.db"
    backup_path = f"memory.db.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("=" * 70)
    print("CONSOLIDAÇÃO SEGURA DE MEMÓRIAS DUPLICADAS - DaviOS")
    print("=" * 70)
    
    # Step 1: Backup adicional
    print(f"\n[1/5] Criando backup adicional...")
    if os.path.exists(db_path):
        shutil.copy2(db_path, backup_path)
        print(f"✓ Backup criado: {backup_path}")
    
    # Step 2: Conectar ao banco
    print(f"\n[2/5] Conectando ao banco...")
    manager = MemoryManager(db_path=db_path)
    
    # Step 3: Estado antes
    print(f"\n[3/5] Analisando estado atual...")
    all_memories = manager.database.find_memories(include_archived=True)
    print(f"Total de memórias: {len(all_memories)}")
    
    semantic_map = {}
    for mem in all_memories:
        for fact in mem.facts:
            target = manager.normalize_fact_target(fact.target)
            subject = manager.normalize_subject(fact.subject)
            fact_type = fact.fact_type or mem.memory_type
            key = (subject, target, fact_type)
            if key not in semantic_map:
                semantic_map[key] = []
            semantic_map[key].append((mem.id, mem.status, fact.relation))
    
    print("\nMemórias por conceito semântico:")
    for key, mems in semantic_map.items():
        subject, target, fact_type = key
        print(f"  {subject} + {target} ({fact_type}): {len(mems)} memórias")
        for mem_id, status, relation in mems:
            print(f"    - Memory {mem_id}: status={status}, relation={relation}")
    
    # Step 4: Executar consolidação
    print(f"\n[4/5] Executando consolidação...")
    result = manager.database.consolidate_duplicate_memories()
    
    # Step 5: Validar resultado
    print(f"\n[5/5] Validando resultado...")
    
    new_memories = manager.database.find_memories(include_archived=False)
    print(f"\nMemórias ativas: {len(new_memories)}")
    
    for mem in new_memories:
        print(f"\n✓ Memory ID: {mem.id}")
        print(f"  Content: {mem.content}")
        print(f"  Status: {mem.status}")
        print(f"  Fatos:")
        for fact in mem.facts:
            print(f"    - Target: {fact.target}")
            print(f"      Relation: {fact.relation}")
            print(f"      Status: {fact.status}")
            
            # Mostra histórico
            history = manager.database.get_fact_history(fact.id)
            print(f"      Histórico ({len(history)} mudanças):")
            for i, rev in enumerate(history[-5:], 1):  # Últimas 5
                print(f"        {i}. {rev.get('relation')} ({rev.get('revision_type')})")
    
    # Resultado da consolidação
    print(f"\n" + "=" * 70)
    print("RESULTADO DA CONSOLIDAÇÃO")
    print("=" * 70)
    
    if result["consolidated"]:
        print(f"\n✓ Consolidações realizadas: {len(result['consolidated'])}")
        for item in result["consolidated"]:
            key = item["semantic_key"]
            print(f"\n  Conceito: {key['subject']} + {key['target']} ({key['fact_type']})")
            print(f"  Memória canônica: {item['canonical_memory_id']}")
            print(f"  Duplicatas marcadas como inativas: {item['duplicate_memory_ids']}")
    else:
        print("\n✓ Nenhuma duplicata encontrada!")
    
    print(f"\n" + "=" * 70)
    print("✓ CONSOLIDAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)
    print(f"\nBackup completo: {backup_path}")
    print(f"Para restaurar: copy {backup_path} {db_path}")
    
    return result


if __name__ == "__main__":
    main()
