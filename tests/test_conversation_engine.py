"""Testes para o Conversation Engine do DaviOS."""

import os
import tempfile

import pytest

from brain.conversation_engine import ConversationEngine, ConversationResult
from memory.memory_manager import MemoryManager


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
            pass


@pytest.fixture
def engine(temp_db):
    manager = MemoryManager(db_path=temp_db)
    engine = ConversationEngine(memory_manager=manager)
    yield engine
    manager.database.close()


class TestConversationEngineCreation:
    def test_engine_creates_with_default_manager(self):
        engine = ConversationEngine()
        assert engine.memory_manager is not None
        assert engine.context is not None
        engine.memory_manager.database.close()

    def test_engine_creates_with_custom_manager(self, engine):
        assert engine.memory_manager is not None
        assert isinstance(engine.context.current_topic, str)


class TestGreeting:
    def test_greeting_oi(self, engine):
        result = engine.process("oi")
        assert result.response == "Ola! Como posso ajudar?"
        assert result.intent == "greeting"
        assert result.should_exit is False

    def test_greeting_ola(self, engine):
        result = engine.process("ola")
        assert result.intent == "greeting"

    def test_greeting_bom_dia(self, engine):
        result = engine.process("bom dia")
        assert result.intent == "greeting"


class TestExit:
    def test_exit_sair(self, engine):
        result = engine.process("sair")
        assert result.response == "Ate mais!"
        assert result.intent == "exit"
        assert result.should_exit is True

    def test_exit_tchau(self, engine):
        result = engine.process("tchau")
        assert result.should_exit is True

    def test_exit_encerrar(self, engine):
        result = engine.process("encerrar")
        assert result.should_exit is True

    def test_exit_quit(self, engine):
        result = engine.process("quit")
        assert result.should_exit is True


class TestEmptyInput:
    def test_empty_string(self, engine):
        result = engine.process("")
        assert result.intent == "empty_input"

    def test_whitespace_only(self, engine):
        result = engine.process("   ")
        assert result.intent == "empty_input"

    def test_none_input(self, engine):
        result = engine.process(None)
        assert result.intent == "empty_input"


class TestMemoryStorage:
    def test_store_preference(self, engine):
        result = engine.process("eu gosto de Python")
        assert result.memory_action == "create"
        assert "Python" in result.response

    def test_store_name(self, engine):
        result = engine.process("meu nome e Davi")
        assert result.memory_action in ("create", "reinforce")


class TestMemoryRetrieval:
    def test_recall_name(self, engine):
        engine.process("meu nome e Davi")
        result = engine.process("qual e meu nome?")
        assert result.intent == "question"
        assert "Davi" in result.response

    def test_recall_preference(self, engine):
        engine.process("eu gosto de Python")
        result = engine.process("do que eu gosto?")
        assert result.intent == "question"
        assert "Python" in result.response

    def test_confirm_preference_positive(self, engine):
        engine.process("eu gosto de Python")
        result = engine.process("eu gosto de Python?")
        assert result.intent == "question"
        assert "Sim" in result.response or "gosta" in result.response.lower()

    def test_confirm_preference_negative(self, engine):
        engine.process("eu gosto de pizza")
        engine.process("eu nao gosto mais de pizza")
        result = engine.process("eu gosto de pizza?")
        assert result.intent == "question"
        assert "nao" in result.response.lower() or "Nao" in result.response


class TestMemoryContradiction:
    def test_contradiction_updates_same_memory(self, engine):
        engine.process("eu gosto de pizza")
        result = engine.process("eu nao gosto mais de pizza")
        assert result.memory_action == "update"
        memories = engine.memory_manager.recall()
        pizza_memories = [m for m in memories if "pizza" in m.content.lower()]
        assert len(pizza_memories) == 1


class TestContext:
    def test_context_tracks_topic(self, engine):
        engine.process("eu gosto de programacao")
        assert engine.context.current_topic == "programacao"

    def test_multiple_messages(self, engine):
        engine.process("eu gosto de Python")
        engine.process("eu gosto de programacao")
        assert engine.context.current_topic == "programacao"


class TestReinforcement:
    def test_reinforce_does_not_create_duplicate(self, engine):
        engine.process("eu gosto de pizza")
        result = engine.process("agora eu gosto de pizza")
        assert result.memory_action in ("reinforce", "update")
        memories = engine.memory_manager.recall()
        pizza_memories = [m for m in memories if "pizza" in m.content.lower()]
        assert len(pizza_memories) == 1


class TestMultipleTargets:
    def test_different_targets_create_separate_memories(self, engine):
        engine.process("eu gosto de pizza")
        engine.process("eu gosto de hamburguer")
        engine.process("eu gosto de programacao")
        result = engine.process("do que eu gosto?")
        assert "pizza" in result.response.lower()
        assert "hamburguer" in result.response.lower()
        assert "programacao" in result.response.lower()


class TestPersistence:
    def test_memory_persists_after_new_engine(self, temp_db):
        manager1 = MemoryManager(db_path=temp_db)
        engine1 = ConversationEngine(memory_manager=manager1)
        engine1.process("eu gosto de Python")
        manager1.database.close()

        manager2 = MemoryManager(db_path=temp_db)
        engine2 = ConversationEngine(memory_manager=manager2)
        result = engine2.process("do que eu gosto?")
        assert "Python" in result.response
        manager2.database.close()


class TestNoDuplication:
    def test_repeated_same_statement_no_duplicates(self, engine):
        engine.process("eu gosto de pizza")
        engine.process("eu gosto de pizza")
        engine.process("eu gosto de pizza")
        facts = engine.memory_manager.database.find_active_preferences()
        pizza_facts = [f for f in facts if f.target.lower() == "pizza"]
        assert len(pizza_facts) == 1
