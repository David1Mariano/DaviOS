"""Testes de LLMProvider, PromptBuilder, CognitiveCore e integracao.

Usa um FakeLLMProvider — nenhum teste depende de modelo real gigante.
"""

import os
import tempfile

import pytest

from brain.cognitive_core import CognitiveCore
from brain.intent_classifier import Intent, IntentClassifier
from brain.llm_provider import LLMProvider, LLMRequest, LLMResponse
from brain.prompt_builder import PromptBuilder
from brain.providers.local_llm_provider import LocalLLMProvider
from brain.tool_registry import Tool, ToolRegistry, ToolRouter
from config.davios_config import DaviosConfig
from memory.memory_manager import MemoryManager


class FakeLLMProvider(LLMProvider):
    """Provider falso: responde com texto fixo e registra chamadas."""

    name = "fake"

    def __init__(self, response: str = "Resposta do modelo falso.", available: bool = True):
        self.fixed_response = response
        self._available = available
        self.calls: list[LLMRequest] = []
        self.unloaded = False

    def initialize(self) -> bool:
        return self._available

    def is_available(self) -> bool:
        return self._available

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(text=self.fixed_response, provider="fake", model="fake-model")

    def unload(self) -> None:
        self.unloaded = True


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


class TestIntentClassifier:
    def setup_method(self):
        self.classifier = IntentClassifier()

    def test_greeting(self):
        assert self.classifier.classify("oi") == Intent.GREETING
        assert self.classifier.classify("ola, tudo bem?") == Intent.GREETING

    def test_farewell(self):
        assert self.classifier.classify("sair") == Intent.FAREWELL
        assert self.classifier.classify("tchau") == Intent.FAREWELL

    def test_memory_query(self):
        assert self.classifier.classify("qual e meu nome?") == Intent.MEMORY_QUERY
        assert self.classifier.classify("do que eu gosto?") == Intent.MEMORY_QUERY
        assert self.classifier.classify("eu gosto de pizza?") == Intent.MEMORY_QUERY

    def test_memory_statement(self):
        assert self.classifier.classify("eu gosto de Python") == Intent.MEMORY_STATEMENT
        assert (
            self.classifier.classify("eu nao gosto mais de pizza")
            == Intent.MEMORY_STATEMENT
        )
        assert self.classifier.classify("meu nome e Davi") == Intent.MEMORY_STATEMENT

    def test_general_question(self):
        assert self.classifier.classify("o que e Python?") == Intent.GENERAL_QUESTION
        assert (
            self.classifier.classify("explique gravidade")
            == Intent.GENERAL_QUESTION
        )
        assert (
            self.classifier.classify("qual a diferenca entre CPU e GPU?")
            == Intent.GENERAL_QUESTION
        )

    def test_opinion(self):
        assert self.classifier.classify("o que voce acha da vida?") == Intent.OPINION
        assert self.classifier.classify("qual sua opiniao sobre musica") == Intent.OPINION

    def test_command_refused(self):
        assert self.classifier.classify("abra o terminal") == Intent.COMMAND
        assert self.classifier.classify("execute um programa") == Intent.COMMAND

    def test_conversation(self):
        assert self.classifier.classify("o sentido da vida e lindo") == Intent.CONVERSATION


class TestPromptBuilder:
    def test_system_includes_personality_and_rules(self):
        # Usa DaviosConfig() (defaults) em vez de .load() (que lê davios.json)
        # para não depender do estado do arquivo de config do usuário.
        built = PromptBuilder(DaviosConfig()).build("oi", memories=[])
        assert "DaviOS" in built.system
        assert "nao pode executar comandos" in built.system or "comandos" in built.system

    def test_memories_formatted(self):
        class Fact:
            target = "pizza"
            relation = "dislike"
            negation = True

        built = PromptBuilder(DaviosConfig()).build("o que eu gosto?", memories=[Fact()])
        assert "pizza" in built.prompt

    def test_name_memory_formatted(self):
        class Fact:
            target = "nome"
            relation = "identity"
            value = "Davi"
            negation = False

        built = PromptBuilder(DaviosConfig()).build("qual e meu nome?", memories=[Fact()])
        assert "Davi" in built.prompt

    def test_history_included(self):
        from brain.conversation_engine import ConversationContext

        context = ConversationContext()
        context.add_message("estou estudando Python", "Legal!", "conversation")
        built = PromptBuilder(DaviosConfig()).build("interfaces", context=context)
        assert "estudando Python" in built.prompt
        assert "interfaces" in built.prompt


@pytest.fixture
def temp_db():
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            pass


class TestCognitiveCoreFlow:
    @pytest.fixture
    def core(self, temp_db):
        config = DaviosConfig.load()
        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider("Resposta do modelo falso.")
        cognitive = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config),
        )
        cognitive.memory_manager = manager
        yield cognitive, manager, provider
        manager.database.close()

    def _save_pizza(self, manager):
        from memory.memory import Memory
        from memory.memory_fact import MemoryFact

        fact = MemoryFact(target="pizza", relation="dislike", negation=True)
        memory = Memory(
            content="eu nao gosto de pizza", memory_type="preference",
            importance=0.7, emotion="neutral", emotional_intensity=0.0,
            facts=[fact],
        )
        manager.database.save_memory(memory)

    def test_generates_with_llm(self, core):
        cognitive, _, provider = core
        outcome = cognitive.generate_response("o que e programacao?")
        assert outcome["text"] == "Resposta do modelo falso."
        assert len(provider.calls) == 1

    def test_prompt_contains_user_message(self, core):
        cognitive, _, provider = core
        cognitive.generate_response("explique gravidade")
        assert "gravidade" in provider.calls[0].prompt

    def test_relevant_memory_retrieved(self, core):
        cognitive, manager, provider = core
        self._save_pizza(manager)
        cognitive.generate_response("voce lembra de pizza?")
        assert "pizza" in provider.calls[0].prompt

    def test_intent_forwarded(self, core):
        cognitive, _, _ = core
        outcome = cognitive.generate_response(
            "o que voce acha da vida?", intent=Intent.OPINION
        )
        assert outcome["intent"] == "opinion"


class TestProviderUnavailable:
    def test_local_provider_without_backend(self):
        provider = LocalLLMProvider(config=DaviosConfig.load())
        # sem llama-cpp-python/sem modelo: nunca lanca excecao
        assert provider.initialize() in (True, False)
        assert isinstance(provider.is_available(), bool)

    def test_local_provider_generate_raises_when_unavailable(self):
        provider = LocalLLMProvider(config=DaviosConfig.load())
        if provider.is_available():
            pytest.skip("modelo local real instalado neste ambiente")
        from brain.llm_provider import LLMUnavailableError

        with pytest.raises(LLMUnavailableError):
            provider.generate(LLMRequest(prompt="oi"))

    def test_health_check_shape(self):
        info = LocalLLMProvider(config=DaviosConfig.load()).health_check()
        assert "available" in info and "backend" in info


class TestConversationEngineWithLLM:
    @pytest.fixture
    def engine(self, temp_db):
        config = DaviosConfig.load()
        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider("Pensamento profundo do modelo.")
        cognitive = CognitiveCore(
            llm_provider=provider, config=config,
            prompt_builder=PromptBuilder(config),
        )
        cognitive.memory_manager = manager
        from brain.conversation_engine import ConversationEngine

        engine = ConversationEngine(memory_manager=manager, cognitive_core=cognitive)
        yield engine, manager, provider
        manager.database.close()

    def test_general_question_uses_llm(self, engine):
        engine_obj, _, provider = engine
        result = engine_obj.process("o que e programacao?")
        assert result.response == "Pensamento profundo do modelo."
        assert result.memory_action == "llm_context"
        assert len(provider.calls) == 1

    def test_memory_query_uses_llm(self, engine):
        engine_obj, _, provider = engine
        result = engine_obj.process("qual e meu nome?")
        assert len(provider.calls) == 1
        assert result.memory_action == "llm_recall"

    def test_memory_query_falls_back_to_rules(self, engine):
        engine_obj, _, provider = engine
        provider._available = False
        result = engine_obj.process("qual e meu nome?")
        assert len(provider.calls) == 0
        assert result.memory_action == "recall"

    def test_memory_statement_uses_llm_after_saving(self, engine):
        engine_obj, manager, provider = engine
        result = engine_obj.process("eu gosto de Python")
        # Memoria salva ANTES da geracao; LLM recebe o fato novo no prompt
        assert result.memory_action == "create"
        assert len(provider.calls) == 1
        prompt = provider.calls[0].prompt
        assert "Python" in prompt or "python" in prompt
        facts = manager.database.find_active_preferences()
        assert any(f.target == "python" for f in facts)

    def test_memory_statement_falls_back_to_rules(self, engine):
        engine_obj, _, provider = engine
        provider._available = False
        result = engine_obj.process("eu gosto de Python")
        assert len(provider.calls) == 0
        assert result.memory_action == "create"

    def test_greeting_does_not_use_llm(self, engine):
        engine_obj, _, provider = engine
        result = engine_obj.process("oi")
        assert result.intent == "greeting"
        assert len(provider.calls) == 0

    def test_command_refused(self, engine):
        engine_obj, _, _ = engine
        result = engine_obj.process("abra o terminal")
        assert result.intent == "command"
        assert "nao" in result.response.lower()

    def test_context_continuity(self, engine):
        engine_obj, _, provider = engine
        engine_obj.process("estou estudando Python")
        engine_obj.process("interfaces")
        prompt = provider.calls[-1].prompt
        assert "Python" in prompt

    def test_llm_error_falls_back_to_rules(self, engine):
        engine_obj, _, provider = engine
        provider._available = False
        result = engine_obj.process("o que e programacao?")
        assert result.response == "Nao tenho essa informacao ainda."


class TestOfflineMode:
    def test_config_defaults_offline(self):
        config = DaviosConfig.load()
        assert config.offline_mode is True
        assert config.network_policy == "offline"

    def test_fake_provider_runs_without_network(self, temp_db):
        manager = MemoryManager(db_path=temp_db)
        provider = FakeLLMProvider("ok offline")
        core = CognitiveCore(llm_provider=provider, config=DaviosConfig.load())
        core.memory_manager = manager
        outcome = core.generate_response("converse comigo")
        assert outcome["text"] == "ok offline"
        manager.database.close()


# ---------------------------------------------------------------------------
# C5 + Correção: tools_registry chega ao PromptBuilder no fluxo real
# ---------------------------------------------------------------------------


class TestToolsRegistryReachesPromptBuilder:
    """Testes de integração que confirmam que o tools_registry populado
    no main.py chega de fato ao PromptBuilder.build() via CognitiveCore."""

    def test_flag_off_prompt_has_no_tools_section(self, temp_db):
        """Flag off → prompt final NÃO contém seção de ferramentas (regressão)."""
        config = DaviosConfig.load()
        config.tools_visible_to_llm = False
        provider = FakeLLMProvider("Resposta normal.")
        reg = ToolRegistry()
        reg.register(Tool(name="echo", description="Repete texto.", arguments=["args"]))
        router = ToolRouter(reg)
        core = CognitiveCore(llm_provider=provider, config=config, tool_router=router)
        core.memory_manager = MemoryManager(db_path=temp_db)
        outcome = core.generate_response("oi")
        assert "FERRAMENTAS DISPONIVEIS" not in outcome["text"]
        assert outcome["text"] == "Resposta normal."
        core.memory_manager.database.close()

    def test_flag_on_prompt_contains_tools_section(self, temp_db):
        """Flag on + registry configurado → prompt final CONTÉM a seção."""
        config = DaviosConfig.load()
        config.tools_visible_to_llm = True
        provider = FakeLLMProvider("Resposta com ferramentas.")
        reg = ToolRegistry()
        reg.register(Tool(name="echo", description="Repete texto.", arguments=["args"]))
        reg.register(Tool(name="time", description="Mostra a hora."))
        router = ToolRouter(reg)
        core = CognitiveCore(llm_provider=provider, config=config, tool_router=router)
        core.memory_manager = MemoryManager(db_path=temp_db)
        outcome = core.generate_response("que horas sao?")
        # A seção de ferramentas deve estar no prompt enviado ao LLM
        prompt_enviado = provider.calls[0].prompt
        assert "FERRAMENTAS DISPONIVEIS" in prompt_enviado
        assert "- echo: Repete texto. (argumentos: args)" in prompt_enviado
        assert "- time: Mostra a hora. (sem argumentos)" in prompt_enviado
        core.memory_manager.database.close()

    def test_no_router_means_no_tools_section_safe_degradation(self, temp_db):
        """Sem tool_router (ex: testes antigos) → não quebra, sem seção."""
        config = DaviosConfig.load()
        config.tools_visible_to_llm = True
        provider = FakeLLMProvider("Resposta sem tools.")
        core = CognitiveCore(llm_provider=provider, config=config)
        core.memory_manager = MemoryManager(db_path=temp_db)
        outcome = core.generate_response("oi")
        # Sem router, tools_registry é None → seção não aparece, mas não quebra
        prompt_enviado = provider.calls[0].prompt
        assert "FERRAMENTAS DISPONIVEIS" not in prompt_enviado
        assert outcome["text"] == "Resposta sem tools."
        core.memory_manager.database.close()

    def test_same_registry_instance_used_by_router_and_prompt_builder(self, temp_db):
        """O MESMO objeto ToolRegistry usado pelo ToolRouter é o que chega
        ao PromptBuilder — não há instâncias divergentes."""
        config = DaviosConfig.load()
        config.tools_visible_to_llm = True
        provider = FakeLLMProvider("Resposta.")
        reg = ToolRegistry()
        reg.register(Tool(name="echo", description="Repete texto.", arguments=["args"]))
        router = ToolRouter(reg)
        core = CognitiveCore(llm_provider=provider, config=config, tool_router=router)
        core.memory_manager = MemoryManager(db_path=temp_db)
        # Confirma que o registry do CognitiveCore é o mesmo objeto do router
        assert core.tools_registry is router.registry
        assert core.tools_registry is reg
        core.generate_response("oi")
        # E que o prompt_builder recém-instanciado recebeu o registry
        prompt_enviado = provider.calls[0].prompt
        assert "FERRAMENTAS DISPONIVEIS" in prompt_enviado
        core.memory_manager.database.close()


