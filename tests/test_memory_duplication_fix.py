"""Testes para validar a correção da duplicação de memórias.

Os 6 testes obrigatórios garantem que o sistema não cria múltiplas Memory
para o mesmo conceito semântico (subject + target + relation_family).
"""

import pytest
from datetime import datetime, timezone
from memory.memory import Memory
from memory.memory_fact import MemoryFact
from memory.memory_manager import MemoryManager
from memory.database import Database
import tempfile
import os


@pytest.fixture
def temp_db():
    """Cria um banco de dados temporário para testes."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def manager(temp_db):
    """Cria um MemoryManager com banco temporário."""
    manager = MemoryManager(db_path=temp_db)
    yield manager
    manager.database.close()


def create_preference_memory(content: str, target: str, relation: str, confidence: float = 0.7) -> Memory:
    """Helper para criar uma memória de preferência."""
    fact = MemoryFact(
        target=target,
        relation=relation,
        emotion="happiness" if relation == "like" else "dislike",
        emotional_intensity=5.0 if relation == "like" else 5.0,
        temporal_context="current",
        negation=relation == "dislike",
        confidence=confidence,
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


class TestMemoryDeduplication:
    """Testes da correção de duplicação de memória."""

    def test_01_primeira_declaracao(self, manager):
        """TESTE 1: Primeira declaração cria uma memória.
        
        Entrada: "eu gosto de pizza"
        Esperado:
        - 1 memory
        - 1 fact
        - pizza -> like
        """
        memory = create_preference_memory(
            "eu gosto de pizza",
            target="pizza",
            relation="like"
        )
        
        result = manager.create_memory(memory)
        
        # Validações
        assert result["action"] == "create", f"Esperado 'create', obtido '{result['action']}'"
        assert result["memory_id"] is not None
        assert result["memory"] is not None
        
        # Verifica memória
        saved_memory = result["memory"]
        assert len(manager.database.find_memories()) == 1
        assert saved_memory.content == "eu gosto de pizza"
        assert saved_memory.memory_type == "preference"
        
        # Verifica fato
        facts = saved_memory.facts
        assert len(facts) == 1
        assert facts[0].target == "pizza"
        assert facts[0].relation == "like"

    def test_02_mesma_informacao_novamente(self, manager):
        """TESTE 2: Mesma informação novamente reforça a memória.
        
        Primeira entrada: "eu gosto de pizza"
        Segunda entrada: "agora eu gosto de pizza"
        
        Esperado:
        - MESMA 1 memory
        - MESMA 1 fact
        - operation = reinforce
        - access_count incrementado
        """
        # Primeira declaração
        memory1 = create_preference_memory(
            "eu gosto de pizza",
            target="pizza",
            relation="like"
        )
        result1 = manager.create_memory(memory1)
        memory_id_1 = result1["memory_id"]
        fact_id_1 = result1["memory"].facts[0].id
        
        # Segunda declaração (semelhante)
        memory2 = create_preference_memory(
            "agora eu gosto de pizza",
            target="pizza",
            relation="like"
        )
        result2 = manager.create_memory(memory2)
        
        # Validações
        assert result2["action"] == "reinforce", f"Esperado 'reinforce', obtido '{result2['action']}'"
        
        # Deve retornar a mesma memória
        assert result2["memory"].id == memory_id_1
        assert result2["memory"].facts[0].id == fact_id_1
        
        # Deve haver apenas 1 memória no banco
        assert len(manager.database.find_memories()) == 1
        
        # access_count deve ter incrementado
        refreshed = manager.database.get_memory(memory_id_1)
        assert refreshed.access_count >= 1

    def test_03_mudanca_de_preferencia(self, manager):
        """TESTE 3: Mudança de preferência atualiza a MESMA memória.
        
        Primeira entrada: "eu gosto de pizza"
        Segunda entrada: "eu não gosto mais de pizza"
        
        Esperado:
        - MESMA 1 memory
        - MESMA 1 fact (mas com relation mudada)
        - operation = update
        - pizza -> dislike
        """
        # Primeira: gosto
        memory1 = create_preference_memory(
            "eu gosto de pizza",
            target="pizza",
            relation="like"
        )
        result1 = manager.create_memory(memory1)
        memory_id_1 = result1["memory_id"]
        
        # Segunda: não gosto
        memory2 = create_preference_memory(
            "eu não gosto mais de pizza",
            target="pizza",
            relation="dislike"
        )
        result2 = manager.create_memory(memory2)
        
        # Validações
        assert result2["action"] == "update", f"Esperado 'update', obtido '{result2['action']}'"
        
        # Deve ser MESMA memória
        assert result2["memory"].id == memory_id_1
        
        # Deve haver apenas 1 memória no banco
        assert len(manager.database.find_memories()) == 1
        
        # Fato deve ter mudado para dislike
        refreshed = manager.database.get_memory(memory_id_1)
        facts = refreshed.facts
        assert len(facts) == 1
        assert facts[0].relation == "dislike"

    def test_04_mudanca_novamente(self, manager):
        """TESTE 4: Mudança novamente continua na MESMA memória.
        
        Primeira: "eu gosto de pizza" (like)
        Segunda: "eu não gosto mais de pizza" (dislike)
        Terceira: "agora eu gosto de pizza" (like novamente)
        
        Esperado:
        - MESMA 1 memory
        - MESMA 1 fact
        - operation = update
        - pizza -> like
        - Histórico preservado: [like, dislike, like]
        """
        # Primeira: like
        memory1 = create_preference_memory(
            "eu gosto de pizza",
            target="pizza",
            relation="like"
        )
        result1 = manager.create_memory(memory1)
        memory_id_1 = result1["memory_id"]
        
        # Segunda: dislike
        memory2 = create_preference_memory(
            "eu não gosto mais de pizza",
            target="pizza",
            relation="dislike"
        )
        result2 = manager.create_memory(memory2)
        
        # Terceira: like novamente
        memory3 = create_preference_memory(
            "agora eu gosto de pizza",
            target="pizza",
            relation="like"
        )
        result3 = manager.create_memory(memory3)
        
        # Validações
        assert result3["action"] == "update", f"Esperado 'update', obtido '{result3['action']}'"
        
        # Deve ser MESMA memória
        assert result3["memory"].id == memory_id_1
        
        # Deve haver apenas 1 memória
        assert len(manager.database.find_memories()) == 1
        
        # Estado atual deve ser like
        refreshed = manager.database.get_memory(memory_id_1)
        facts = refreshed.facts
        assert len(facts) == 1
        assert facts[0].relation == "like"
        
        # Histórico deve ter 3 entradas
        history = manager.database.get_fact_history(facts[0].id)
        assert len(history) >= 3, f"Esperado pelo menos 3 revisões, obteve {len(history)}"

    def test_05_frase_incerta(self, manager):
        """TESTE 5: Frase incerta continua na MESMA memória.
        
        Primeira: "eu gosto de pizza" (like, confidence=0.7)
        Segunda: "eu acho que não gosto mais de pizza" (dislike, confidence=0.65, uncertain)
        
        Esperado:
        - MESMA 1 memory
        - MESMA 1 fact
        - Confiança reduzida, mas MESMA memória
        - NÃO cria nova memória por causa da incerteza
        """
        # Primeira: certain like
        memory1 = create_preference_memory(
            "eu gosto de pizza",
            target="pizza",
            relation="like",
            confidence=0.7
        )
        result1 = manager.create_memory(memory1)
        memory_id_1 = result1["memory_id"]
        
        # Segunda: uncertain dislike
        memory2 = create_preference_memory(
            "eu acho que não gosto mais de pizza",
            target="pizza",
            relation="dislike",
            confidence=0.65
        )
        result2 = manager.create_memory(memory2)
        
        # Validações
        # Deve ser update (relação mudou) mesmo com incerteza
        assert result2["action"] in ["update", "reinforce"], \
            f"Esperado 'update' ou 'reinforce', obtido '{result2['action']}'"
        
        # Deve ser MESMA memória
        assert result2["memory"].id == memory_id_1
        
        # Deve haver apenas 1 memória
        assert len(manager.database.find_memories()) == 1
        
        # Confiança deve estar reduzida
        refreshed = manager.database.get_memory(memory_id_1)
        facts = refreshed.facts
        assert facts[0].confidence < 0.7

    def test_06_frases_semanticamente_iguais(self, manager):
        """TESTE 6: Frases semanticamente iguais não criam duplicatas.
        
        Três frases diferentes, MESMA preferência:
        - "eu gosto de pizza"
        - "eu adoro pizza"
        - "pizza é uma das minhas comidas favoritas"
        
        Esperado:
        - TODAS na MESMA 1 memory
        - MESMA 1 fact (ou múltiplos fatos da mesma memória)
        - NÃO 3 memórias diferentes
        """
        # Primeira
        memory1 = create_preference_memory(
            "eu gosto de pizza",
            target="pizza",
            relation="like"
        )
        result1 = manager.create_memory(memory1)
        memory_id_1 = result1["memory_id"]
        
        # Segunda (variation)
        memory2 = create_preference_memory(
            "eu adoro pizza",
            target="pizza",
            relation="like"
        )
        result2 = manager.create_memory(memory2)
        
        # Terceira (variation)
        memory3 = create_preference_memory(
            "pizza é uma das minhas comidas favoritas",
            target="pizza",
            relation="like"
        )
        result3 = manager.create_memory(memory3)
        
        # Validações - todos devem estar na MESMA memória
        assert result2["action"] == "reinforce", \
            f"Segunda deveria ser 'reinforce', obteve '{result2['action']}'"
        assert result3["action"] == "reinforce", \
            f"Terceira deveria ser 'reinforce', obteve '{result3['action']}'"
        
        # Todos apontam para a mesma memória
        assert result2["memory"].id == memory_id_1
        assert result3["memory"].id == memory_id_1
        
        # Deve haver apenas 1 memória no banco
        all_memories = manager.database.find_memories()
        assert len(all_memories) == 1, \
            f"Esperado 1 memória, encontrou {len(all_memories)}"


class TestMemoryConsolidation:
    """Testes para garantir consolidação correta de múltiplos fatos."""

    def test_multiplos_fatos_mesma_memoria(self, manager):
        """Quando a mesma memória tem múltiplos fatos, todos são preservados."""
        memory = Memory(
            content="Eu gosto de pizza e odeio brócolis",
            memory_type="preference",
            importance=0.7,
            emotion="mixed",
            emotional_intensity=4.0,
            facts=[
                MemoryFact(
                    target="pizza",
                    relation="like",
                    emotion="happiness",
                    emotional_intensity=5.0,
                ),
                MemoryFact(
                    target="brócolis",
                    relation="dislike",
                    emotion="dislike",
                    emotional_intensity=5.0,
                ),
            ],
        )
        
        result = manager.create_memory(memory)
        assert result["action"] == "create"
        
        saved = result["memory"]
        assert len(saved.facts) == 2
        
        # Verifica que ambos os fatos existem
        targets = {fact.target for fact in saved.facts}
        assert targets == {"pizza", "brócolis"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
