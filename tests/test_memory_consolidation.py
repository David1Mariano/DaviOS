"""Testes de consolidação de memórias duplicadas e estabilidade do pipeline.

Cobre:
- Parte 10: sequência completa de preferências sobre pizza -> 1 memória/1 fato.
- Parte 11: variações linguísticas reconhecidas como o mesmo conceito.
- Parte 12: alvos diferentes geram memórias independentes.
- Parte 14: migração de dados legados (backup testado à parte, via script).
- Parte 15: pós-migração, nova declaração idêntica gera reinforce.
- Parte 13: índice de unicidade semântica impede fato ativo duplicado.
"""

import os
import sqlite3
import tempfile

import pytest

from memory.memory import Memory
from memory.memory_manager import MemoryManager
from memory.memory_interpreter import MemoryInterpreter


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            pass  # conexão ainda aberta em algum SO/lock; ignora no teardown


@pytest.fixture
def manager(temp_db):
    manager = MemoryManager(db_path=temp_db)
    yield manager
    manager.database.close()


def process_input(manager: MemoryManager, text: str) -> dict:
    """Pipeline completo: ContextAnalyzer -> EmotionAnalyzer -> Fact -> decisão.

    O matching acontece ANTES da criação da memória (dentro de
    create_memory -> _determine_operation), nunca depois.
    """
    context = manager.context_analyzer.analyze(text)
    emotions = []
    for part in context["parts"]:
        emotion = manager.emotion_analyzer.analyze(part["text"], part)
        emotions.append(
            {
                "text": part["text"],
                "emotion": emotion["emotion"],
                "emotional_intensity": emotion["emotional_intensity"],
                "context": part,
            }
        )
    interpretation = MemoryInterpreter().interpret(text, context, emotions)
    if not interpretation["facts"]:
        return {"action": "no_facts"}
    memory = Memory(
        content=text,
        memory_type="preference",
        importance=0.7,
        emotion="neutral",
        emotional_intensity=0.0,
        facts=interpretation["facts"],
    )
    return manager.create_memory(memory)


def active_memory_ids_for(manager, target: str) -> set:
    facts = manager.database.find_facts_for_semantic_key(
        "user", target, "preference", include_inactive=False
    )
    return {fact.memory_id for fact in facts}


def active_facts_for(manager, target: str) -> list:
    return manager.database.find_facts_for_semantic_key(
        "user", target, "preference", include_inactive=False
    )


class TestSequenciaCompletaPizza:
    """Parte 10: uma única memória conceitual para pizza."""

    def test_sequencia_de_seis_declaracoes(self, manager):
        sequence = [
            "eu gosto de pizza",
            "agora eu gosto de pizza",
            "eu nao gosto mais de pizza",
            "agora eu nao gosto mais de pizza",
            "eu gosto de pizza",
            "eu nao gosto de pizza",
        ]
        operations = [process_input(manager, text)["action"] for text in sequence]

        assert operations[0] == "create"
        assert operations[1] == "reinforce"
        assert operations[2] == "update"
        assert operations[3] == "reinforce"
        assert operations[4] == "update"
        assert operations[5] == "update"

        pizza_memory_ids = active_memory_ids_for(manager, "pizza")
        assert pizza_memory_ids == {1}, f"memórias ativas: {pizza_memory_ids}"

        facts = active_facts_for(manager, "pizza")
        assert len(facts) == 1
        fact = facts[0]
        assert fact.target == "pizza"
        assert fact.relation == "dislike"
        assert fact.negation is True
        assert fact.status == "active"
        assert fact.fact_type == "preference"

        # Parte 7: histórico com transições preservado
        history = manager.database.get_fact_history(fact.id)
        revision_types = {entry["revision_type"] for entry in history}
        assert "created" in revision_types
        assert "updated" in revision_types


class TestVariacaoLinguistica:
    """Parte 11: frase diferente, mesmo conceito."""

    def test_variacoes_reforcam_ou_atualizam_a_mesma_memoria(self, manager):
        first = process_input(manager, "eu gosto de pizza")
        memory_id = first["memory_id"]
        fact_id = first["memory"].facts[0].id

        for text in (
            "eu adoro pizza",
            "pizza é uma das minhas comidas favoritas",
            "eu curto pizza",
        ):
            result = process_input(manager, text)
            assert result["action"] in {"reinforce", "update"}, text
            fact = active_facts_for(manager, "pizza")[0]
            assert fact.memory_id == memory_id
            assert fact.id == fact_id
            assert fact.relation == "like"

        # Mudança de estado na mesma memória
        result = process_input(manager, "eu não gosto de pizza")
        assert result["action"] == "update"
        fact = active_facts_for(manager, "pizza")[0]
        assert fact.memory_id == memory_id
        assert fact.id == fact_id
        assert fact.relation == "dislike"
        assert fact.negation is True


class TestOutrosAlvos:
    """Parte 12: cada conceito tem sua própria memória."""

    def test_alvos_independentes(self, manager):
        for text, target in [
            ("eu gosto de pizza", "pizza"),
            ("eu gosto de hambúrguer", "hambúrguer"),
            ("eu gosto de programação", "programação"),
            ("eu gosto de Python", "python"),
        ]:
            result = process_input(manager, text)
            assert result["action"] == "create", text
            facts = active_facts_for(manager, target)
            assert len(facts) == 1
            assert facts[0].relation == "like"

        assert len(active_memory_ids_for(manager, "pizza")) == 1

        # Atualiza SOMENTE pizza
        process_input(manager, "eu nao gosto de pizza")
        assert active_facts_for(manager, "pizza")[0].relation == "dislike"
        for target in ("hambúrguer", "programação", "python"):
            assert active_facts_for(manager, target)[0].relation == "like"


class TestTargetNormalization:
    """Parte 3: normalização nunca produz alvos inválidos."""

    def test_target_hedge_words_removidos(self, manager):
        assert (
            manager.normalize_fact_target("acho que nao pizza na real") == "pizza"
        )

    def test_target_valido_preservado(self, manager):
        assert manager.normalize_fact_target("pizza") == "pizza"
        assert manager.normalize_fact_target("programação") == "programação"


class TestConsolidacaoLegado:
    """Partes 2 e 14: consolida duplicatas legadas preservando histórico."""

    def _seed_legacy_database(self, manager):
        # Memory 1: pizza -> like (ficará conflicted, estado antigo)
        m1 = process_input(manager, "agora eu gosto de pizza")
        fact1 = m1["memory"].facts[0]
        manager.database.set_fact_status(fact1.id, "conflicted", reason="legado")

        # Memory 2: target inválido da normalização antiga
        m2 = process_input(manager, "eu acho que nao gosto mais de pizza na real")
        # A extração atual pode já normalizar; força o resíduo legado:
        manager.database.cursor.execute(
            "UPDATE memory_facts SET target = ? WHERE id = ?",
            ("acho que nao pizza na real", m2["memory"].facts[0].id),
        )
        manager.database.connection.commit()
        manager.database.add_fact_evidence(
            m2["memory"].facts[0].id,
            {
                "content": "eu acho que nao gosto mais de pizza na real",
                "source": "user_statement",
                "confidence": 0.65,
            },
        )

        # Memory 3: estado atual correto
        m3 = process_input(manager, "eu nao gosto mais de pizza")
        return m1, m2, m3

    def test_consolidacao_escolhe_memoria_mais_recente(self, manager):
        m1, m2, m3 = self._seed_legacy_database(manager)

        result = manager.consolidate_memories()

        assert result["fixed_targets"] >= 1
        assert result["unique_index"] is True

        # Apenas a memória 3 permanece ativa para pizza
        assert active_memory_ids_for(manager, "pizza") == {m3["memory_id"]}

        # Duplicatas marcadas como inativas, não apagadas
        for memory in manager.database.find_memories(include_archived=True):
            if memory.id in (m1["memory_id"], m2["memory_id"]):
                assert memory.status == "inactive"

        # Fato canônico: estado atual
        facts = active_facts_for(manager, "pizza")
        assert len(facts) == 1
        fact = facts[0]
        assert fact.memory_id == m3["memory_id"]
        assert fact.relation == "dislike"
        assert fact.negation is True
        assert fact.status == "active"

        # Evidências da duplicata foram movidas para o fato canônico
        evidences = manager.database.get_fact_evidence(fact.id)
        contents = " ".join(item["content"] for item in evidences)
        assert "nao gosto mais de pizza" in contents

        # Histórico preservado (nada foi removido)
        total_memories = len(manager.database.find_memories(include_archived=True))
        assert total_memories == 3

    def test_pos_migracao_reinforce_nao_cria_duplicata(self, manager):
        self._seed_legacy_database(manager)
        manager.consolidate_memories()

        result = process_input(manager, "eu nao gosto mais de pizza")

        assert result["action"] == "reinforce"
        assert active_memory_ids_for(manager, "pizza") == {result["memory"].id}
        assert len(active_facts_for(manager, "pizza")) == 1


class TestUnicidadeSemantica:
    """Parte 13: restrição de unicidade no nível do banco."""

    def test_indice_impede_fato_ativo_duplicado(self, manager):
        process_input(manager, "eu gosto de pizza")
        assert manager.database.ensure_semantic_uniqueness_index() is True

        with pytest.raises(sqlite3.IntegrityError):
            manager.database.cursor.execute(
                """
                INSERT INTO memory_facts (
                    memory_id, target, relation, subject, status, fact_type
                ) VALUES (1, 'pizza', 'like', 'user', 'active', 'preference')
                """
            )
            manager.database.connection.commit()
        manager.database.connection.rollback()

    def test_historico_inativo_nao_viola_unicidade(self, manager):
        process_input(manager, "eu gosto de pizza")
        fact = active_facts_for(manager, "pizza")[0]
        manager.database.set_fact_status(fact.id, "superseded", reason="teste")
        assert manager.database.ensure_semantic_uniqueness_index() is True
