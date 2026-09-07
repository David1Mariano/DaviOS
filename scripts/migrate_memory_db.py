"""Migração segura do banco de memórias existente.

Fluxo: backup -> correção de targets inválidos -> consolidação de
duplicatas -> índice de unicidade semântica -> relatório final.

Uso (a partir da raiz do projeto):
    python scripts/migrate_memory_db.py
    python scripts/migrate_memory_db.py caminho/para/memory.db
"""

import shutil
import sys
import time
from pathlib import Path

# Console Windows (cp1252) não consegue imprimir os caracteres usados nos
# logs internos do subsistema de memória.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.memory_manager import MemoryManager  # noqa: E402


def migrate(db_path: str = "memory.db") -> None:
    if not Path(db_path).exists():
        print(f"[MIGRATION] banco não encontrado: {db_path}")
        return

    backup_path = f"{db_path}.backup-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(db_path, backup_path)
    print(f"[BACKUP] {db_path} -> {backup_path}")

    manager = MemoryManager(db_path=db_path)
    result = manager.consolidate_memories()

    print(f"[MIGRATION] targets corrigidos: {result['fixed_targets']}")
    for item in result["consolidated"]:
        print(
            "[MIGRATION] conceito "
            f"({item['semantic_key']['subject']}, {item['semantic_key']['target']}, "
            f"{item['semantic_key']['fact_type']}): canônica memory_id="
            f"{item['canonical_memory_id']}, duplicatas inativadas="
            f"{item['duplicate_memory_ids']}"
        )
    print(f"[MIGRATION] índice de unicidade aplicado: {result['unique_index']}")

    print("\n[RELATORIO FINAL]")
    for memory in manager.database.find_memories(include_archived=True):
        print(
            f"  Memory id={memory.id} status={memory.status} "
            f"type={memory.memory_type} content='{memory.content}'"
        )
        for fact in memory.facts:
            print(
                f"    Fact id={fact.id} subject={fact.subject} target={fact.target} "
                f"relation={fact.relation} negation={fact.negation} "
                f"status={fact.status} fact_type={fact.fact_type}"
            )

    manager.database.close()
    print(f"\n[MIGRATION] concluída. Backup preservado em: {backup_path}")


if __name__ == "__main__":
    migrate(sys.argv[1] if len(sys.argv) > 1 else "memory.db")
