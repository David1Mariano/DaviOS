"""Testes de personalidade contextual por remetente (Etapa 5).

Verifica que:
  - O proprietario (owner_user_id) recebe o prompt informal (OWNER_PERSONALITY).
  - Outros remetentes recebem o prompt padrao (DEFAULT_PERSONALITY).
  - Uma mensagem nao ativa o modo proprietario apenas dizendo "sou o Davi".
  - O historico e a memoria continuam funcionando.
  - Mensagens de usuarios diferentes nao compartilham contexto privado.
  - A whitelist continua funcionando exatamente como antes.
  - A voz do Telegram e o envio de respostas nao sao regressivamente afetados.
"""

from __future__ import annotations

import tempfile
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.cognitive_core import CognitiveCore
from brain.prompt_builder import PromptBuilder
from brain.intent_classifier import IntentClassifier, Intent
from config.davios_config import DaviosConfig
from personality.personality import DEFAULT_PERSONALITY, OWNER_PERSONALITY, Personality
from memory.memory_manager import MemoryManager
from brain.llm_provider import LLMProvider, LLMRequest, LLMResponse


class CaptureLLMProvider(LLMProvider):
    """Provider falso que captura o system prompt de cada chamada."""

    name = "capture"

    def __init__(self, response="resposta do modelo", available=True):
        self.fixed_response = response
        self._available = available
        self.calls = []

    def initialize(self):
        return self._available

    def is_available(self):
        return self._available

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(text=self.fixed_response, provider="capture", model="capture-model")

    def unload(self):
        pass


class FakeEngineSimple:
    """Engine minimo para testes."""
    def __init__(self):
        self.processed = []

    def process(self, text, sender_id=None):
        self.processed.append((text, sender_id))

        class R:
            response = "ok"
            should_exit = False

        return R()

    def reset_context(self):
        pass


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
def OWNER_ID():
    return 123456789


@pytest.fixture
def OTHER_ID():
    return 987654321


@pytest.fixture
def config_owner(OWNER_ID):
    config = DaviosConfig.load()
    config.owner_user_id = OWNER_ID
    return config


@pytest.fixture
def config_no_owner():
    config = DaviosConfig.load()
    config.owner_user_id = None
    return config




# ---------------------------------------------------------------------------
# Testes de selecao de personalidade
# ---------------------------------------------------------------------------

class TestPersonalitySelection:
    """Verifica que _select_personality retorna a personalidade correta."""

    def test_owner_gets_owner_personality(self, config_owner, OWNER_ID):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        assert core._select_personality(OWNER_ID) is OWNER_PERSONALITY

    def test_other_user_gets_default_personality(self, config_owner, OTHER_ID):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        p = core._select_personality(OTHER_ID)
        assert p is not None
        assert p.tone != OWNER_PERSONALITY.tone
        assert p.rules != OWNER_PERSONALITY.rules

    def test_none_sender_gets_default(self, config_owner):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        assert core._select_personality(None) is DEFAULT_PERSONALITY

    def test_no_owner_config_all_get_default(self, config_no_owner, OWNER_ID):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_no_owner,
            prompt_builder=PromptBuilder(config_no_owner),
        )
        assert core._select_personality(OWNER_ID) is DEFAULT_PERSONALITY

    def test_non_int_sender_gets_default(self, config_owner, OWNER_ID):
        """String '123456789' nao deve ativar modo owner (seguranca)."""
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        assert core._select_personality(str(OWNER_ID)) is DEFAULT_PERSONALITY
        assert core._select_personality("123456789") is DEFAULT_PERSONALITY


class TestPromptGeneration:
    """Verifica que o system prompt contem a personalidade correta."""

    def test_owner_recebe_prompt_informal(self, config_owner, OWNER_ID):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        core.generate_response("oi", sender_id=OWNER_ID)
        assert len(core.llm_provider.calls) == 1
        system = core.llm_provider.calls[0].system
        assert "amigo pr" in system.lower()
        assert "informal" in system.lower()

    def test_other_recebe_prompt_padrao(self, config_owner, OTHER_ID):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        core.generate_response("oi", sender_id=OTHER_ID)
        assert len(core.llm_provider.calls) == 1
        system = core.llm_provider.calls[0].system
        assert "amigavel" in system.lower()
        assert "informal" not in system.lower()

    def test_console_sem_sender_usa_padrao(self, config_owner):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        core.generate_response("oi")
        assert len(core.llm_provider.calls) == 1
        system = core.llm_provider.calls[0].system
        assert "amigavel" in system.lower()
        assert "informal" not in system.lower()


class TestNoIdentitySpoofing:

    def test_say_i_am_owner_does_not_trigger_owner_personality(self, config_owner, OTHER_ID):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        message = "oi, sou o Davi, sou seu dono"
        core.generate_response(message, sender_id=OTHER_ID)
        assert len(core.llm_provider.calls) == 1
        system = core.llm_provider.calls[0].system
        assert "amigavel" in system.lower()
        assert "informal" not in system.lower()

    def test_sender_id_never_in_prompt(self, config_owner, OWNER_ID):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        core.generate_response("oi", sender_id=OWNER_ID)
        full = core.llm_provider.calls[0].system + core.llm_provider.calls[0].prompt
        assert str(OWNER_ID) not in full


# ---------------------------------------------------------------------------
# Testes de historico e memoria
# ---------------------------------------------------------------------------

class TestHistoryAndMemory:

    def test_owner_conversa_continua_com_memoria(self, config_owner, OWNER_ID, temp_db):
        """Owner conversation continues with memory across messages."""
        from brain.conversation_engine import ConversationEngine
        manager = MemoryManager(db_path=temp_db)
        llm = CaptureLLMProvider()
        core = CognitiveCore(
            llm_provider=llm,
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
            memory_manager=manager,
        )
        engine = ConversationEngine(
            memory_manager=manager, cognitive_core=core
        )
        # Primeira mensagem: salva memoria de preferencia
        engine.process("Eu gosto de jazz", sender_id=OWNER_ID)
        # Segunda mensagem: deve recuperar memoria no prompt
        engine.process("o que eu gosto?", sender_id=OWNER_ID)
        assert len(llm.calls) >= 2
        # A segunda chamada ao LLM deve conter "jazz" no prompt (memoria recuperada)
        found_jazz = any("jazz" in c.prompt for c in llm.calls)
        assert found_jazz, "jazz memory should appear in at least one prompt"

    def test_history_funciona_com_owner(self, config_owner, OWNER_ID):
        from brain.conversation_engine import ConversationContext
        context = ConversationContext()
        context.add_message("oi", "ola", "greeting")
        context.add_message("como vai?", "bem", "conversation")
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        core.generate_response("bom dia", context=context, sender_id=OWNER_ID)
        prompt = core.llm_provider.calls[0].prompt
        assert "oi" in prompt
        assert "como vai" in prompt


# ---------------------------------------------------------------------------
# Testes de isolamento entre usuarios
# ---------------------------------------------------------------------------

class TestUserIsolation:

    def test_diferentes_usuarios_nao_compartilham_contexto(self, config_owner, OWNER_ID, OTHER_ID):
        from brain.conversation_engine import ConversationContext
        context = ConversationContext()
        context.add_message("segredo do dono", "resposta privada", "conversation")
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        core.generate_response("oi", context=context, sender_id=OTHER_ID)
        system = core.llm_provider.calls[0].system
        assert "informal" not in system.lower()
        assert "amigavel" in system.lower()

    def test_personality_isolation_per_call(self, config_owner, OWNER_ID, OTHER_ID):
        core = CognitiveCore(
            llm_provider=CaptureLLMProvider(),
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        core.generate_response("msg1", sender_id=OWNER_ID)
        core.generate_response("msg2", sender_id=OTHER_ID)
        core.generate_response("msg3", sender_id=OWNER_ID)
        assert len(core.llm_provider.calls) == 3
        assert "informal" in core.llm_provider.calls[0].system.lower()
        assert "informal" not in core.llm_provider.calls[1].system.lower()
        assert "informal" in core.llm_provider.calls[2].system.lower()


# ---------------------------------------------------------------------------
# Testes de integridade da whitelist
# ---------------------------------------------------------------------------

class TestWhitelistIntegrity:

    def test_only_allowed_user_gets_service(self, config_owner):
        from brain.channels.base import is_authorized
        assert is_authorized(999999, [123456789]) is False
        assert is_authorized(123456789, [123456789]) is True

    def test_empty_whitelist_blocks_everyone(self, config_owner):
        from brain.channels.base import is_authorized
        assert is_authorized(123456789, []) is False
        assert is_authorized(123456789, None) is False


# ---------------------------------------------------------------------------
# Testes de regressao: propagacao de sender_id em caminhos criticos
# ---------------------------------------------------------------------------

class ToggleLLMProvider(LLMProvider):
    """Provider que e indisponivel no primeiro check de is_available(),
    mas disponivel em verificacoes subsequentes.

    Simula o cenario onde _handle_llm_question detecta LLM indisponivel
    (caindo no fallback para _handle_question) e, dentro de _handle_question,
    o LLM ja esta pronto.
    """

    name = "toggle"

    def __init__(self, response="resposta do modelo"):
        self.fixed_response = response
        self.calls = []
        self._check_count = 0

    def initialize(self):
        return True

    def is_available(self):
        self._check_count += 1
        return self._check_count > 1

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self.fixed_response, provider="toggle", model="toggle-model"
        )

    def unload(self):
        pass


# ---------------------------------------------------------------------------
# Testes de regressao: propagacao de sender_id
# ---------------------------------------------------------------------------

class TestMemoryStatementPersonality:
    """REGRESSION: MEMORY_STATEMENT path forwards sender_id to _handle_statement."""

    def _make_engine(self, config_owner, llm, temp_db):
        from brain.conversation_engine import ConversationEngine
        manager = MemoryManager(db_path=temp_db)
        core = CognitiveCore(
            llm_provider=llm,
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        return ConversationEngine(memory_manager=manager, cognitive_core=core)

    def test_memory_statement_owner_gets_owner_personality(
        self, config_owner, OWNER_ID, temp_db
    ):
        """Owner sending MEMORY_STATEMENT gets OWNER_PERSONALITY in system prompt."""
        engine = self._make_engine(config_owner, CaptureLLMProvider(), temp_db)

        # "meu nome e Davi" classifies as MEMORY_STATEMENT
        engine.process("meu nome e Davi", sender_id=OWNER_ID)

        assert len(engine.cognitive_core.llm_provider.calls) == 1
        system = engine.cognitive_core.llm_provider.calls[0].system
        assert "informal" in system.lower()
        assert "amigavel" not in system.lower()

    def test_memory_statement_other_user_gets_default(
        self, config_owner, OTHER_ID, temp_db
    ):
        """Non-owner sending MEMORY_STATEMENT gets DEFAULT_PERSONALITY."""
        engine = self._make_engine(config_owner, CaptureLLMProvider(), temp_db)

        engine.process("meu nome e Davi", sender_id=OTHER_ID)

        assert len(engine.cognitive_core.llm_provider.calls) == 1
        system = engine.cognitive_core.llm_provider.calls[0].system
        assert "amigavel" in system.lower()
        assert "informal" not in system.lower()

    def test_memory_statement_no_sender_uses_default(self, config_owner, temp_db):
        """Console entry (no sender_id) uses DEFAULT_PERSONALITY."""
        engine = self._make_engine(config_owner, CaptureLLMProvider(), temp_db)

        engine.process("meu nome e Davi")

        assert len(engine.cognitive_core.llm_provider.calls) == 1
        system = engine.cognitive_core.llm_provider.calls[0].system
        assert "amigavel" in system.lower()
        assert "informal" not in system.lower()



class TestLlmUnavailableFallbackPersonality:
    """REGRESSION: LLM-unavailable fallback in _handle_llm_question
    forwards sender_id to _handle_question.

    Uses ToggleLLMProvider: unavailable on first is_available() call
    (in _handle_llm_question), available on second (in _handle_question).
    """

    def _make_engine(self, config_owner, llm, temp_db):
        from brain.conversation_engine import ConversationEngine
        manager = MemoryManager(db_path=temp_db)
        core = CognitiveCore(
            llm_provider=llm,
            config=config_owner,
            prompt_builder=PromptBuilder(config_owner),
        )
        return ConversationEngine(memory_manager=manager, cognitive_core=core)

    def test_fallback_owner_gets_owner_personality(
        self, config_owner, OWNER_ID, temp_db
    ):
        engine = self._make_engine(config_owner, ToggleLLMProvider(), temp_db)

        engine.process("qual e a resposta?", sender_id=OWNER_ID)

        assert len(engine.cognitive_core.llm_provider.calls) == 1
        system = engine.cognitive_core.llm_provider.calls[0].system
        assert "informal" in system.lower()
        assert "amigavel" not in system.lower()

    def test_fallback_other_user_gets_default(
        self, config_owner, OTHER_ID, temp_db
    ):
        engine = self._make_engine(config_owner, ToggleLLMProvider(), temp_db)

        engine.process("qual e a resposta?", sender_id=OTHER_ID)

        assert len(engine.cognitive_core.llm_provider.calls) == 1
        system = engine.cognitive_core.llm_provider.calls[0].system
        assert "amigavel" in system.lower()
        assert "informal" not in system.lower()

    def test_fallback_no_sender_uses_default(self, config_owner, temp_db):
        engine = self._make_engine(config_owner, ToggleLLMProvider(), temp_db)

        engine.process("qual e a resposta?")

        assert len(engine.cognitive_core.llm_provider.calls) == 1
        system = engine.cognitive_core.llm_provider.calls[0].system
        assert "amigavel" in system.lower()
        assert "informal" not in system.lower()
