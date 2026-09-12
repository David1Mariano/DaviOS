"""Testes do fechamento do loop de ferramentas (C5).

Cobrem:
    1. flag off → comportamento idêntico ao pré-C5 (1 chamada ao LLM);
    2. flag on + texto sem pedido → resposta normal, sem execução;
    3. pedido válido autorizado → execute() com args corretos + 2ª chamada;
    4. categoria flag off → negação, execute() nunca chamado;
    5. ferramenta inexistente → mesma negação, sem crash;
    6. pedido malformado → texto cru, sem 2ª chamada (decisão 2c);
    7. REGRA DE OURO: cada condição falhando sozinha bloqueia;
    8. no máximo 1 execução por resposta (segunda resposta nunca re-parseada);
    9. logging verificável de todo pedido reconhecido.
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.cognitive_core import CognitiveCore
from brain.llm_provider import LLMProvider, LLMRequest, LLMResponse
from brain.tool_execution_flow import ToolExecutionOrchestrator
from brain.tool_registry import Tool, ToolRegistry, ToolRouter
from config.davios_config import DaviosConfig


class ScriptedProvider(LLMProvider):
    """Provider fake: devolve respostas roteirizadas e registra chamadas."""

    name = "scripted"

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[LLMRequest] = []

    def initialize(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return LLMResponse(text=self.responses[idx], provider="scripted",
                           model="fake-model")

    def unload(self) -> None:
        pass


def _recording_tool(name: str, category: str, output):
    """Tool cujo execute grava chamadas e devolve output fixo."""
    calls: list[dict] = []

    def _execute(**kwargs):
        calls.append(dict(kwargs))
        return output

    tool = Tool(
        name=name,
        description=f"Ferramenta de teste {name}.",
        arguments=[],
        execute=_execute,
        category=category,
    )
    return tool, calls


def _make_registry():
    """Registry com 1 tool 'actions', 1 'web' e os logs de chamadas."""
    reg = ToolRegistry()
    echo_tool, echo_calls = _recording_tool("echo", "actions", "ECO: <args>")
    web_tool, web_calls = _recording_tool(
        "web_fetch", "web", {"ok": True, "text": "pagina"}
    )
    reg.register(echo_tool)
    reg.register(web_tool)
    return reg, echo_calls, web_calls


def _config(**flags) -> DaviosConfig:
    config = DaviosConfig()
    for key, value in flags.items():
        setattr(config, key, value)
    return config


def _core(provider, config, registry):
    return CognitiveCore(
        llm_provider=provider,
        config=config,
        tool_router=ToolRouter(registry) if registry is not None else None,
    )


def _block(tool: str, args_json: str = '{"args": "oi"}') -> str:
    inner = f'{{"tool": "{tool}", "arguments": {args_json}}}'
    return f"<<<TOOL_REQUEST>>>\n{inner}\n<<<END_TOOL_REQUEST>>>"


# ---------------------------------------------------------------------------
# 1. Flag off → comportamento idêntico ao pré-C5
# ---------------------------------------------------------------------------


def test_flag_off_identical_behavior_even_with_tool_request_in_text():
    block = _block("echo", '{"args": "oi"}')
    provider = ScriptedProvider([block])  # bloco no texto do Qwen
    config = _config(tools_visible_to_llm=False, actions_enabled=True)
    reg, echo_calls, _ = _make_registry()
    core = _core(provider, config, reg)

    out = core.generate_response("repita oi")

    assert len(provider.calls) == 1  # UMA chamada ao LLM, como antes
    # C6: sanitização remove o bloco residual mesmo com flag off (rede de segurança).
    assert "<<<TOOL_REQUEST>>>" not in out["text"]
    assert "<<<END_TOOL_REQUEST>>>" not in out["text"]
    assert "tool_flow" not in out    # sem metadata do fluxo
    assert echo_calls == []          # nada executado


# ---------------------------------------------------------------------------
# 2. Flag on + texto sem pedido → resposta normal
# ---------------------------------------------------------------------------


def test_flag_on_no_request_single_llm_call_no_execution():
    provider = ScriptedProvider(["Resposta normal ao usuario."])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, echo_calls, web_calls = _make_registry()
    core = _core(provider, config, reg)

    out = core.generate_response("oi, tudo bem?")

    assert len(provider.calls) == 1  # sem segunda chamada
    assert out["text"] == "Resposta normal ao usuario."
    assert echo_calls == [] and web_calls == []
    assert out["tool_flow"]["request_found"] is False
    assert out["tool_flow"]["tool_executed"] is False


# ---------------------------------------------------------------------------
# 3. Pedido válido autorizado → execute + segunda chamada com resultado
# ---------------------------------------------------------------------------


def test_valid_request_executes_and_second_llm_call_includes_result():
    first = "Vou verificar. " + _block("echo", '{"args": "teste"}')
    provider = ScriptedProvider([first, "O resultado foi: teste."])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, echo_calls, _ = _make_registry()
    core = _core(provider, config, reg)

    out = core.generate_response("repita teste")

    assert len(provider.calls) == 2
    assert echo_calls == [{"args": "teste"}]  # execute com args corretos
    followup_prompt = provider.calls[1].prompt
    assert "Resultado da execucao da ferramenta 'echo'" in followup_prompt
    assert "ECO: <args>" in followup_prompt  # saída da tool no follow-up
    assert out["text"] == "O resultado foi: teste."  # resposta final da 2ª chamada
    assert out["tool_flow"]["tool_executed"] is True
    assert out["tool_flow"]["execution_success"] is True
    assert out["tool_flow"]["second_llm_call"] is True


def test_valid_web_request_result_dict_serialized_in_followup():
    first = _block("web_fetch", '{"url": "http://exemplo.com"}')
    provider = ScriptedProvider([first, "A pagina diz: pagina."])
    config = _config(tools_visible_to_llm=True, web_enabled=True)
    reg, _, web_calls = _make_registry()
    core = _core(provider, config, reg)

    out = core.generate_response("busque a pagina")

    assert len(provider.calls) == 2
    assert web_calls == [{"url": "http://exemplo.com"}]
    followup_prompt = provider.calls[1].prompt
    assert "Resultado da execucao da ferramenta 'web_fetch'" in followup_prompt
    assert '"text": "pagina"' in followup_prompt  # dict serializado
    assert out["text"] == "A pagina diz: pagina."


# ---------------------------------------------------------------------------
# 3b. Followup_prompt NÃO contém seção de ferramentas (correção do bug
#     "bloco cru aparece + resultado atrasado")
# ---------------------------------------------------------------------------


def test_followup_prompt_does_not_contain_tools_section():
    """O followup_prompt NÃO deve conter a seção de ferramentas, para não
    induzir o segundo LLM a pedir outra ferramenta."""
    first = _block("echo", '{"args": "teste"}')
    provider = ScriptedProvider([first, "O resultado foi: teste."])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, _, _ = _make_registry()
    core = _core(provider, config, reg)

    core.generate_response("repita teste")

    followup_prompt = provider.calls[1].prompt
    assert "FERRAMENTAS DISPONIVEIS" not in followup_prompt
    assert "<<<TOOL_REQUEST>>>" not in followup_prompt
    assert "<<<END_TOOL_REQUEST>>>" not in followup_prompt


def test_followup_prompt_contains_tool_result():
    """O followup_prompt DEVE conter o resultado da execução da ferramenta."""
    first = _block("echo", '{"args": "teste"}')
    provider = ScriptedProvider([first, "O resultado foi: teste."])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, _, _ = _make_registry()
    core = _core(provider, config, reg)

    core.generate_response("repita teste")

    followup_prompt = provider.calls[1].prompt
    assert "Resultado da execucao da ferramenta 'echo'" in followup_prompt
    assert "ECO: <args>" in followup_prompt


def test_followup_prompt_contains_no_tool_request_instruction():
    """O followup_prompt DEVE conter instrução explícita de NÃO pedir outra ferramenta."""
    first = _block("echo", '{"args": "teste"}')
    provider = ScriptedProvider([first, "O resultado foi: teste."])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, _, _ = _make_registry()
    core = _core(provider, config, reg)

    core.generate_response("repita teste")

    followup_prompt = provider.calls[1].prompt
    assert "NAO peca nenhuma ferramenta" in followup_prompt


def test_followup_prompt_contains_user_message():
    """O followup_prompt DEVE conter a mensagem original do usuário para contexto."""
    first = _block("echo", '{"args": "teste"}')
    provider = ScriptedProvider([first, "O resultado foi: teste."])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, _, _ = _make_registry()
    core = _core(provider, config, reg)

    core.generate_response("repita teste")

    followup_prompt = provider.calls[1].prompt
    assert "Mensagem original do usuario: repita teste" in followup_prompt


def test_followup_for_denied_request_also_has_no_tools_section():
    """Followup de negação (ferramenta inexistente) também NÃO deve conter
    a seção de ferramentas."""
    first = _block("ferramenta_fantasma", '{"x": 1}')
    provider = ScriptedProvider([first, "Essa funcao nao existe."])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, _, _ = _make_registry()
    core = _core(provider, config, reg)

    core.generate_response("use ferramenta fantasma")

    followup_prompt = provider.calls[1].prompt
    assert "FERRAMENTAS DISPONIVEIS" not in followup_prompt
    assert "<<<TOOL_REQUEST>>>" not in followup_prompt


# ---------------------------------------------------------------------------
# 4. Categoria flag off → negação, execute() nunca chamado
# ---------------------------------------------------------------------------


def test_denied_when_actions_flag_off():
    first = _block("echo", '{"args": "oi"}')
    provider = ScriptedProvider([first, "Nao consigo fazer isso agora."])
    config = _config(tools_visible_to_llm=True, actions_enabled=False)
    reg, echo_calls, _ = _make_registry()
    core = _core(provider, config, reg)

    out = core.generate_response("repita oi")

    assert echo_calls == []  # PROVA: execute() nunca foi chamado
    assert len(provider.calls) == 2  # negação volta ao Qwen p/ resposta final
    followup_prompt = provider.calls[1].prompt
    assert "nao foi autorizado" in followup_prompt
    # Negacao GENERICA: sem vazar detalhes internos de configuracao:
    assert "actions_enabled" not in followup_prompt
    assert "tools_visible_to_llm" not in followup_prompt
    assert "actions" not in followup_prompt
    assert out["text"] == "Nao consigo fazer isso agora."
    assert out["tool_flow"]["tool_executed"] is False
    assert out["tool_flow"]["authorized"] is False


def test_denied_when_web_flag_off():
    first = _block("web_fetch", '{"url": "http://exemplo.com"}')
    provider = ScriptedProvider([first, "Sem acesso a web."])
    config = _config(tools_visible_to_llm=True, web_enabled=False)
    reg, _, web_calls = _make_registry()
    core = _core(provider, config, reg)

    out = core.generate_response("busque a pagina")

    assert web_calls == []
    assert len(provider.calls) == 2
    assert "nao foi autorizado" in provider.calls[1].prompt
    assert out["tool_flow"]["authorized"] is False


# ---------------------------------------------------------------------------
# 5. Ferramenta inexistente → mesma negação, sem crash
# ---------------------------------------------------------------------------


def test_denied_when_tool_does_not_exist():
    first = _block("ferramenta_fantasma", '{"x": 1}')
    provider = ScriptedProvider([first, "Essa funcao nao existe."])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, echo_calls, _ = _make_registry()
    core = _core(provider, config, reg)

    out = core.generate_response("use a ferramenta fantasma")

    assert echo_calls == []  # nada foi executado
    assert len(provider.calls) == 2
    followup_prompt = provider.calls[1].prompt
    assert "ferramenta_fantasma" in followup_prompt  # nome do pedido
    assert "nao foi autorizado" in followup_prompt
    assert out["text"] == "Essa funcao nao existe."
    assert out["tool_flow"]["tool_executed"] is False


# ---------------------------------------------------------------------------
# 6. Pedido malformado → decisão 2c: texto cru, sem segunda chamada
# ---------------------------------------------------------------------------


def test_malformed_request_raw_text_unchanged_no_second_call():
    first = (
        "Texto com bloco quebrado:\n"
        "<<<TOOL_REQUEST>>>\n"
        '{"tool": "echo", "arguments": {"args" \n'
        "<<<END_TOOL_REQUEST>>>\n"
        "Fim da resposta."
    )
    provider = ScriptedProvider([first])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, echo_calls, _ = _make_registry()
    core = _core(provider, config, reg)

    out = core.generate_response("repita oi")

    assert len(provider.calls) == 1  # SEM segunda chamada ao LLM
    # C6: sanitização remove o bloco malformado residual (comportamento defensivo).
    assert "<<<TOOL_REQUEST>>>" not in out["text"]
    assert "<<<END_TOOL_REQUEST>>>" not in out["text"]
    assert "Fim da resposta." in out["text"]
    assert echo_calls == []
    assert out["tool_flow"]["request_found"] is True   # tentativa reconhecida
    assert out["tool_flow"]["tool_executed"] is False


# ---------------------------------------------------------------------------
# 7. REGRA DE OURO — cada condição falhando SOZINHA bloqueia a execução
# ---------------------------------------------------------------------------


def _recording_run_llm(calls: list):
    def _run(prompt: str, system: str) -> str:
        calls.append(prompt)
        return "resposta da segunda chamada"
    return _run


def test_golden_rule_condition1_tools_visible_off_blocks():
    # Condição 1 falhando sozinha: flag off → nem parse, nem LLM, nem execução.
    reg, _, _ = _make_registry()
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=False, actions_enabled=True),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    result = orchestrator.handle_first_response(
        _block("echo", '{"args": "oi"}'),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert result.final_text == _block("echo", '{"args": "oi"}')
    assert result.tool_executed is False
    assert result.request_found is False  # nem chegou a parsear
    assert llm_calls == []                # segunda chamada NUNCA acontece


def test_golden_rule_condition2_category_flag_off_blocks():
    # Condição 2 falhando sozinha: tool existe, mas actions_enabled=False.
    reg, echo_calls, _ = _make_registry()
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, actions_enabled=False),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    result = orchestrator.handle_first_response(
        _block("echo", '{"args": "oi"}'),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert echo_calls == []               # PROVA: não executou
    assert result.tool_executed is False
    assert result.authorized is False
    assert len(llm_calls) == 1            # 1 chamada: a da negação
    assert "nao foi autorizado" in llm_calls[0]


def test_golden_rule_condition3_tool_not_in_registry_blocks():
    # Condição 3 falhando sozinha: registry existe, mas a tool pedida não.
    reg, echo_calls, _ = _make_registry()
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, actions_enabled=True),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    result = orchestrator.handle_first_response(
        _block("inexistente", '{"x": 1}'),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert echo_calls == []
    assert result.tool_executed is False
    assert result.authorized is False
    assert len(llm_calls) == 1
    assert "nao foi autorizado" in llm_calls[0]


def test_golden_rule_tool_without_category_never_executes():
    # Default seguro: categoria vazia/desconhecida NUNCA executa.
    reg = ToolRegistry()
    tool_calls: list = []

    def _execute(**kwargs):
        tool_calls.append(kwargs)
        return "ok"

    reg.register(Tool(name="sem_categoria", description="Sem categoria.",
                      execute=_execute))  # category="" (default)
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, actions_enabled=True, web_enabled=True),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    result = orchestrator.handle_first_response(
        _block("sem_categoria", "{}"),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert tool_calls == []
    assert result.tool_executed is False
    assert result.authorized is False


def test_golden_rule_all_conditions_met_executes():
    # Contraprova: as 3 condições verdadeiras → executa.
    reg, echo_calls, _ = _make_registry()
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, actions_enabled=True),
        tool_router=ToolRouter(reg),
    )
    result = orchestrator.handle_first_response(
        _block("echo", '{"args": "ok"}'),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm([]),
    )
    assert echo_calls == [{"args": "ok"}]
    assert result.tool_executed is True
    assert result.authorized is True


# ---------------------------------------------------------------------------
# 8. Limite de segurança: no máximo 1 execução por resposta
# ---------------------------------------------------------------------------


def test_max_one_execution_even_if_second_response_contains_request():
    # Mesmo que a 2ª resposta do Qwen contenha outro bloco <<<TOOL_REQUEST>>>,
    # ele NUNCA é re-parseado → só 1 execução por resposta do usuário.
    first = _block("echo", '{"args": "um"}')
    second_with_block = (
        "Veja outro pedido: "
        + _block("echo", '{"args": "dois"}')
        + " fim."
    )
    provider = ScriptedProvider([first, second_with_block])
    config = _config(tools_visible_to_llm=True, actions_enabled=True)
    reg, echo_calls, _ = _make_registry()
    core = _core(provider, config, reg)

    core.generate_response("repita um")

    # Apenas 1 execução (o segundo bloco no texto da 2ª resposta é ignorado).
    assert echo_calls == [{"args": "um"}]
    assert len(provider.calls) == 2  # 1ª + follow-up; sem 3ª chamada


def test_max_one_execution_orchestrator_only_parses_first_response():
    # O orquestrador nunca re-parseia a saída da segunda chamada ao LLM.
    reg, echo_calls, _ = _make_registry()
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, actions_enabled=True),
        tool_router=ToolRouter(reg),
    )

    def _llm_with_another_request(prompt: str, system: str) -> str:
        # Segunda resposta contém novo bloco — deve ser ignorado.
        return _block("echo", '{"args": "nao_deve_executar"}')

    result = orchestrator.handle_first_response(
        _block("echo", '{"args": "primeiro"}'),
        original_prompt="P",
        original_system="S",
        run_llm=_llm_with_another_request,
    )
    # Só a primeira execução aconteceu.
    assert echo_calls == [{"args": "primeiro"}]
    assert result.tool_executed is True
    assert result.second_llm_call is True


# ---------------------------------------------------------------------------
# 9. Logging de todo pedido reconhecido
# ---------------------------------------------------------------------------


def test_recognized_request_logs_tool_and_args_keys(caplog):
    reg, _, _ = _make_registry()
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, actions_enabled=True),
        tool_router=ToolRouter(reg),
    )
    with caplog.at_level(logging.INFO, logger="davios.toolflow"):
        orchestrator.handle_first_response(
            _block("echo", '{"args": "oi", "extra": 1}'),
            original_prompt="P",
            original_system="S",
            run_llm=_recording_run_llm([]),
        )
    # Pedido reconhecido logado com tool + CHAVES dos args (nunca valores).
    messages = " ".join(rec.message for rec in caplog.records)
    assert "echo" in messages
    assert "args" in messages  # chave
    assert "extra" in messages  # chave
    assert '"oi"' not in messages  # valor NUNCA é logado


def test_malformed_request_logs_error(caplog):
    reg, _, _ = _make_registry()
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, actions_enabled=True),
        tool_router=ToolRouter(reg),
    )
    with caplog.at_level(logging.INFO, logger="davios.toolflow"):
        orchestrator.handle_first_response(
            "<<<TOOL_REQUEST>>>\\n{nao eh json\\n<<<END_TOOL_REQUEST>>>",
            original_prompt="P",
            original_system="S",
            run_llm=_recording_run_llm([]),
        )
    messages = " ".join(rec.message for rec in caplog.records)
    assert "malformado" in messages


def test_denied_request_logs_internal_reason(caplog):
    reg, _, _ = _make_registry()
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, actions_enabled=False),
        tool_router=ToolRouter(reg),
    )
    with caplog.at_level(logging.INFO, logger="davios.toolflow"):
        orchestrator.handle_first_response(
            _block("echo", '{"args": "oi"}'),
            original_prompt="P",
            original_system="S",
            run_llm=_recording_run_llm([]),
        )
    messages = " ".join(rec.message for rec in caplog.records)
    assert "negado" in messages
    assert "categoria" in messages


def test_execution_result_logged(caplog):
    reg, _, _ = _make_registry()
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, actions_enabled=True),
        tool_router=ToolRouter(reg),
    )
    with caplog.at_level(logging.INFO, logger="davios.toolflow"):
        orchestrator.handle_first_response(
            _block("echo", '{"args": "oi"}'),
            original_prompt="P",
            original_system="S",
            run_llm=_recording_run_llm([]),
        )
    messages = " ".join(rec.message for rec in caplog.records)
    assert "autorizado" in messages
    assert "execucao concluida" in messages
    assert "success=True" in messages
