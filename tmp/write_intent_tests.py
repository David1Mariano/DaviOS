# -*- coding: utf-8 -*-
"""Genera tests/test_intent_classifier.py (escritura atomica).
Evita los limites de caracteres del editor de archivos.
"""

from pathlib import Path

CONTENT = '''"""Testes de la inteligencia conversacional (etapa 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.intent_classifier import (
    Intent,
    IntentClassifier,
    resolve_context_reference,
)
from brain.prompt_builder import PromptBuilder
from config.davios_config import DaviosConfig
from personality.personality import DEFAULT_PERSONALITY


class _Ctx:
    """Contexto minimo: misma forma que ConversationContext."""

    def __init__(self, messages=None, current_topic=""):
        self.messages = messages or []
        self.current_topic = current_topic


class TestSocial:
    def test_laughter_k(self):
        c = IntentClassifier()
        assert c.classify("kkkk") == Intent.SOCIAL
        assert c.classify("kkk") == Intent.SOCIAL

    def test_reaction_legal(self):
        assert IntentClassifier().classify("legal") == Intent.SOCIAL

    def test_reaction_boa(self):
        assert IntentClassifier().classify("boa") == Intent.SOCIAL

    def test_slang_eai(self):
        assert IntentClassifier().classify("eai") == Intent.SOCIAL


class TestSocialReciprocal:
    def test_e_voce(self):
        c = IntentClassifier()
        assert c.classify("e você?") == Intent.SOCIAL_RECIPROCAL
        assert c.classify("e vc?") == Intent.SOCIAL_RECIPROCAL

    def test_full_reciprocal(self):
        c = IntentClassifier()
        assert c.classify("eu tô bem e você?") == Intent.SOCIAL_RECIPROCAL
        assert c.classify("to bem, e voce?") == Intent.SOCIAL_RECIPROCAL

    def test_como_voce_esta(self):
        assert (
            IntentClassifier().classify("como você está?")
            == Intent.SOCIAL_RECIPROCAL
        )

    def test_no_false_positive_que_voce(self):
        assert (
            IntentClassifier().classify("o que voce acha da vida?")
            == Intent.OPINION
        )


class TestFollowUp:
    def test_por_que_with_history(self):
        ctx = _Ctx(
            messages=[{"user": "ele tá travando", "response": "me conta mais"}]
        )
        assert (
            IntentClassifier().classify("por quê?", context=ctx)
            == Intent.FOLLOW_UP
        )

    def test_por_que_without_history_is_general(self):
        assert (
            IntentClassifier().classify("por quê?")
            == Intent.GENERAL_QUESTION
        )

    def test_como_assim_with_history(self):
        ctx = _Ctx(messages=[{"user": "o café está frio", "response": "uhm"}])
        assert (
            IntentClassifier().classify("como assim?", context=ctx)
            == Intent.FOLLOW_UP
        )

    def test_o_que_queres_saber_sobre(self):
        ctx = _Ctx(
            messages=[
                {"user": "tô com problema no caixa", "response": "conta mais"}
            ]
        )
        assert (
            IntentClassifier().classify("o que você quer saber sobre?", context=ctx)
            == Intent.FOLLOW_UP
        )

PART2 = '''class TestFileRequest:
    def test_que_tem_no_main(self):
        assert (
            IntentClassifier().classify("o que tem no main.py?")
            == Intent.FILE_REQUEST
        )

    def test_leia_archivo(self):
        assert IntentClassifier().classify("leia o main.py") == Intent.FILE_REQUEST

    def test_abre_esse_archivo(self):
        assert (
            IntentClassifier().classify("abre esse arquivo")
            == Intent.FILE_REQUEST
        )

    def test_terminal_command_still_command(self):
        assert (
            IntentClassifier().classify("abra o terminal") == Intent.COMMAND
        )


class TestTimeRequest:
    def test_que_horas_son(self):
        assert IntentClassifier().classify("que horas são?") == Intent.TIME_REQUEST

    def test_me_fala_as_horas(self):
        assert IntentClassifier().classify("me fala as horas") == Intent.TIME_REQUEST

    def test_qual_hora_agora(self):
        assert (
            IntentClassifier().classify("qual a hora agora?")
            == Intent.TIME_REQUEST
        )


class TestPreservedContracts:
    def test_greeting_preserved(self):
        c = IntentClassifier()
        assert c.classify("oi") == Intent.GREETING
        assert c.classify("ola, tudo bem?") == Intent.GREETING

    def test_memory_query_preserved(self):
        c = IntentClassifier()
        assert c.classify("qual e meu nome?") == Intent.MEMORY_QUERY
        assert c.classify("do que eu gosto?") == Intent.MEMORY_QUERY

    def test_general_question_preserved(self):
        c = IntentClassifier()
        assert c.classify("o que e Python?") == Intent.GENERAL_QUESTION


class TestResolveContextReference:
    def test_follow_up_uses_last_message(self):
        ctx = _Ctx(
            messages=[
                {
                    "user": "ele tá travando e não contabiliza",
                    "response": "puede ser config",
                }
            ]
        )
        out = resolve_context_reference("por quê?", ctx)
        assert out is not None
        assert "ele tá travando" in out
        assert "puede ser config" in out

    def test_explicit_queres_saber_uses_last_message(self):
        ctx = _Ctx(
            messages=[{"user": "tô com problema no caixa", "response": "conta mais"}]
        )
        out = resolve_context_reference("o que você quer saber sobre?", ctx)
        assert out is not None
        assert "caixa" in out

    def test_reciprocal_no_topic_resolution(self):
        """'e você?' NO se trata como referencia a topico anterior."""
        ctx = _Ctx(
            messages=[{"user": "tô com problema no caixa", "response": "me conta"}],
            current_topic="caixa",
        )
        assert resolve_context_reference("e você?", ctx) is None

    def test_full_question_no_resolution(self):
        ctx = _Ctx(messages=[{"user": "a", "response": "b"}])
        assert resolve_context_reference("por que o céu é azul?", ctx) is None
'''

PART3 = '''class TestConversationPolicy:
    def _builder(self) -> PromptBuilder:
        return PromptBuilder(DaviosConfig(), DEFAULT_PERSONALITY)

    def test_policy_follow_up_present(self):
        built = self._builder().build("por quê?", intent="follow_up")
        assert "CONTINUACION" in built.prompt
        assert "Mensagem do usuario: por quê?" in built.prompt

    def test_policy_social_reciprocal_present(self):
        built = self._builder().build("e você?", intent="social_reciprocal")
        assert "reciproca" in built.prompt

    def test_policy_file_request_mentions_read_file(self):
        built = self._builder().build(
            "o que tem no main.py?", intent="file_request"
        )
        assert "read_file" in built.prompt

    def test_policy_time_request_mentions_time(self):
        built = self._builder().build(
            "que horas são?", intent="time_request"
        )
        assert "herramienta time" in built.prompt

    def test_no_policy_for_conversation(self):
        built = self._builder().build("oi", intent="conversation")
        assert "CONTINUACION" not in built.prompt
        assert "reciproca" not in built.prompt
        assert built.prompt == "Mensagem do usuario: oi"


class TestToolIntegration:
    def test_time_request_tool_available_in_registry(self):
        """'que horas son?' = TIME_REQUEST y la herramienta `time` existe."""
        from brain.tool_adapters import register_default_tools
        from brain.tool_registry import ToolRegistry

        reg = ToolRegistry()
        register_default_tools(reg)
        tool = reg.find("time")
        assert tool is not None
        assert callable(getattr(tool, "execute", None))
        assert IntentClassifier().classify("que horas são?") == Intent.TIME_REQUEST

    def test_file_request_prompt_orients_to_read_file_tool(self):
        """FILE_REQUEST + registry real → prompt con orientacion read_file."""
        from brain.tool_adapters import register_default_tools
        from brain.tool_registry import ToolRegistry

        reg = ToolRegistry()
        register_default_tools(reg)
        names = {t.name for t in reg.list_tools()}
        assert "read_file" in names

        config = DaviosConfig()
        config.tools_visible_to_llm = True
        builder = PromptBuilder(config, DEFAULT_PERSONALITY)
        built = builder.build(
            "o que tem no main.py?",
            intent="file_request",
            tools_registry=reg,
        )
        assert "read_file" in built.prompt
        assert (
            IntentClassifier().classify("o que tem no main.py?")
            == Intent.FILE_REQUEST
        )
'''

with open("tests/test_intent_classifier.py", "a", encoding="utf-8") as fh:
    fh.write(PART2)
    fh.write(PART3)
print("PARTS_OK")

    def test_full_questions_not_follow_up(self):
        c = IntentClassifier()
        ctx = _Ctx(messages=[{"user": "a", "response": "b"}])
        assert (
            c.classify("por que o céu é azul?", context=ctx)
            == Intent.GENERAL_QUESTION
        )
        assert (
            c.classify("como funciona Python?", context=ctx)
            == Intent.GENERAL_QUESTION
        )
        assert (
            c.classify("qual é a capital do Brasil?", context=ctx)
            == Intent.GENERAL_QUESTION
        )
'''

Path("tests/test_intent_classifier.py").write_text(CONTENT, encoding="utf-8")
print("PART1_OK")