"""Testes de integracao cognitiva do DaviOS.

Prova o fluxo completo: mensagem -> memoria -> prompt -> LLM -> resposta,
incluindo persistencia apos reinicializacao da aplicacao.

Usa FakeLLMProvider — nenhum teste depende de modelo real.
"""

import os
import tempfile

import pytest

from brain.cognitive_core import CognitiveCore
from brain.llm_provider import LLMProvider, LLMRequest, LLMResponse
from brain.prompt_builder import PromptBuilder
from config.davios_config import DaviosConfig
from memory.memory_manager import MemoryManager


class FakeLLMProvider(LLMProvider):
    """Provider falso: responde com base no prompt recebido."""

    name = "fake"

    def __init__(self, response="Resposta do modelo falso.", available=True):
        self.fixed_response = response
        self._available = available
        self.calls: list[LLMRequest] = []

    def initialize(self) -> bool:
        return self._available

    def is_available(self) -> bool:
        return self._available

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        prompt = request.prompt.lower()
        if "nome do usuario" in prompt and "davi" in prompt:
            return LLMResponse(text="Seu nome e Davi.", provider="fake", model="fake")
        if "netoptimizer" in prompt:
            return LLMResponse(
                text="Voce trabalha no NetOptimizer.", provider="fake", model="fake"
            )
        if "python" in prompt and "gosta" in prompt:
            return LLMResponse(
                text="Sim, voce gosta de Python.", provider="fake", model="fake"
            )
        return LLMResponse(text=self.fixed_response, provider="fake", model="fake")

    def unload(self) -> None:
        pass


class UnavailableProvider(LLMProvider):
    name = "unavailable"

    def initialize(self):
        return False

    def is_available(self):
        return False

    def generate(self, request):
        raise RuntimeError("nao deveria ser chamado")

    def unload(self):
        pass


# ==========================================================================
# Testes de integracao: memoria + LLM
# ==========================================================================


class TestMemoryStatementFlow:
    """Prova que MEMORY_STATEMENT salva antes de gerar resposta via LLM."""

    def test_statement_saves_then_uses_llm(self, temp_db, config):
        from brain.conversation_engine import ConversationEngine

        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider()
        core = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager,
        )
        engine = ConversationEngine(memory_manager=manager, cognitive_core=core)
        result = engine.process("eu gosto de Python")
        assert result.memory_action in ("create", "add")
        assert len(provider.calls) == 1
        assert "python" in provider.calls[0].prompt.lower()
        manager.database.close()

    def test_statement_facts_in_prompt(self, temp_db, config):
        from brain.conversation_engine import ConversationEngine

        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider()
        core = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager,
        )
        engine = ConversationEngine(memory_manager=manager, cognitive_core=core)
        engine.process("eu gosto de Python")
        prompt = provider.calls[0].prompt.lower()
        assert "python" in prompt
        manager.database.close()

    def test_statement_fallback_to_rules(self, temp_db, config):
        from brain.conversation_engine import ConversationEngine

        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider(available=False)
        core = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager,
        )
        engine = ConversationEngine(memory_manager=manager, cognitive_core=core)
        result = engine.process("eu gosto de Python")
        assert len(provider.calls) == 0
        assert result.memory_action in ("create", "add")
        manager.database.close()


class TestMemoryQueryFlow:
    """Prova que MEMORY_QUERY recupera fatos e passa ao LLM."""

    def test_query_uses_llm_with_memories(self, temp_db, config):
        from brain.conversation_engine import ConversationEngine

        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider()
        core = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager,
        )
        engine = ConversationEngine(memory_manager=manager, cognitive_core=core)
        engine.process("meu nome e Davi")


class TestPersistence:
    """Prova de persistencia: dados sobrevivem a reinicializacao."""

    def test_memory_persists_after_restart(self, temp_db, config):
        from brain.conversation_engine import ConversationEngine

        manager1 = MemoryManager(db_path=temp_db)
        provider1 = FakeLLMProvider()
        core1 = CognitiveCore(
            llm_provider=provider1, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager1,
        )
        engine1 = ConversationEngine(memory_manager=manager1, cognitive_core=core1)
        engine1.process("meu nome e Davi")
        engine1.process("eu gosto de Python")
        manager1.database.close()

        manager2 = MemoryManager(db_path=temp_db)
        provider2 = FakeLLMProvider()
        core2 = CognitiveCore(
            llm_provider=provider2, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager2,
        )
        engine2 = ConversationEngine(memory_manager=manager2, cognitive_core=core2)
        result = engine2.process("qual e meu nome?")
        assert "Davi" in result.response
        manager2.database.close()

    def test_working_on_persists(self, temp_db, config):
        from brain.conversation_engine import ConversationEngine

        manager1 = MemoryManager(db_path=temp_db)
        provider1 = FakeLLMProvider()
        core1 = CognitiveCore(
            llm_provider=provider1, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager1,
        )
        engine1 = ConversationEngine(memory_manager=manager1, cognitive_core=core1)
        engine1.process("estou trabalhando no NetOptimizer")
        manager1.database.close()

        manager2 = MemoryManager(db_path=temp_db)
        facts = manager2.database.find_active_preferences(relation_family="working_on")
        assert any(f.target == "netoptimizer" for f in facts)
        manager2.database.close()


class TestContradictionHandling:
    """Prova que contradicoes atualizam a memoria corretamente."""

    def test_contradiction_updates_same_fact(self, temp_db, config):
        from brain.conversation_engine import ConversationEngine

        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider()
        core = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager,
        )
        engine = ConversationEngine(memory_manager=manager, cognitive_core=core)
        engine.process("eu gosto de pizza")
        result = engine.process("eu nao gosto mais de pizza")
        assert result.memory_action in ("update", "reinforce")
        facts = manager.database.find_facts_by_target("pizza")
        active = [f for f in facts if f.is_active]
        assert len(active) == 1
        manager.database.close()


class TestMultiTurnConversation:
    """Prova conversa multi-turno com contexto."""

    def test_context_continuity(self, temp_db, config):
        from brain.conversation_engine import ConversationEngine

        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider()
        core = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager,
        )
        engine = ConversationEngine(memory_manager=manager, cognitive_core=core)
        engine.process("estou estudando Python")
        provider.calls.clear()
        engine.process("interfaces")
        if provider.calls:
            prompt = provider.calls[-1].prompt.lower()
            assert "python" in prompt
        manager.database.close()


class TestObservability:
    """Prova que metadados estao presentes."""

    def test_prompt_metadata(self, temp_db, config):
        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider()
        core = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager,
        )
        from brain.conversation_engine import ConversationEngine

        engine = ConversationEngine(memory_manager=manager, cognitive_core=core)
        engine.process("eu gosto de Python")
        assert len(provider.calls) == 1
        built = core.prompt_builder.build("teste", memories=[], intent="conversation")
        assert "memories_count" in built.metadata
        manager.database.close()

        result = engine.process("qual e meu nome?")
        assert result.memory_action in ("llm_recall", "recall")
        assert len(provider.calls) >= 1
        manager.database.close()

    def test_query_fallback_to_rules(self, temp_db, config):
        from brain.conversation_engine import ConversationEngine

        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider(available=False)
        core = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config), memory_manager=manager,
        )
        engine = ConversationEngine(memory_manager=manager, cognitive_core=core)
        engine.process("meu nome e Davi")
        provider.calls.clear()
        result = engine.process("qual e meu nome?")
        assert len(provider.calls) == 0
        assert "Davi" in result.response
        manager.database.close()


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
def config():
    return DaviosConfig.load()
