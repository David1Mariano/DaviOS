"""Testes unitários para phone_adapters.

Seguem o mesmo padrão dos testes atuais: registry + router + stubs, sem unittest.mock.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.phone_adapters import register_phone_tools
from brain.phone_client import PhoneClient
from brain.tool_execution_flow import ToolExecutionOrchestrator
from brain.tool_registry import Tool, ToolRegistry, ToolRouter
from config.davios_config import DaviosConfig


# ---------------------------------------------------------------------------
# Stubs do PhoneClient
# ---------------------------------------------------------------------------


class _StubPhoneOK:
    """Simula PhoneClient com todas as chamadas OK."""

    def get_battery(self) -> dict:
        return {"ok": True, "level": 85, "charging": False}

    def get_location(self) -> dict:
        return {"ok": True, "lat": -23.5, "lon": -46.6, "accuracy": 10}

    def get_device_info(self) -> dict:
        return {"ok": True, "model": "SM-A307F", "android_version": "11"}

    def speak(self, text: str) -> dict:
        return {"ok": True, "spoken": True}

    def vibrate(self, ms: int) -> dict:
        return {"ok": True, "vibrated": True}


class _StubPhoneFail:
    """Simula PhoneClient com falha nas chamadas."""

    def get_battery(self) -> dict:
        return {"ok": False, "error": "Falha ao ler bateria: timeout"}

    def get_location(self) -> dict:
        return {"ok": False, "error": "GPS desligado"}

    def get_device_info(self) -> dict:
        return {"ok": False, "error": "Dispositivo não respondeu"}

    def speak(self, text: str) -> dict:
        return {"ok": False, "error": "TTS indisponível"}

    def vibrate(self, ms: int) -> dict:
        return {"ok": False, "error": "Vibração falhou"}


# ---------------------------------------------------------------------------
# Helpers para testes do ToolExecutionOrchestrator
# ---------------------------------------------------------------------------


def _config(**flags) -> DaviosConfig:
    config = DaviosConfig()
    for key, value in flags.items():
        setattr(config, key, value)
    return config


def _block(tool: str, args_json: str = "{}") -> str:
    inner = f'{{"tool": "{tool}", "arguments": {args_json}}}'
    return f"<<<TOOL_REQUEST>>>\n{inner}\n<<<END_TOOL_REQUEST>>>"


def _recording_run_llm(calls: list):
    def _run(prompt: str, system: str) -> str:
        calls.append(prompt)
        return "resposta da segunda chamada"
    return _run


# ---------------------------------------------------------------------------
# 1. Registro e categorias
# ---------------------------------------------------------------------------


def test_phone_adapters_registra_todas_as_ferramentas():
    reg = ToolRegistry()
    registered = register_phone_tools(reg, _StubPhoneOK())
    assert len(registered) == 5
    names = {t.name for t in reg.list_tools()}
    assert names == {
        "phone_battery",
        "phone_location",
        "phone_device_info",
        "phone_speak",
        "phone_vibrate",
    }
    # Categorias corretas
    for name in ("phone_battery", "phone_location", "phone_device_info"):
        tool = reg.get(name)
        assert tool.category == "phone_read"
    for name in ("phone_speak", "phone_vibrate"):
        tool = reg.get(name)
        assert tool.category == "phone_act"
    # Descrições existem e são restritivas
    for tool in reg.list_tools():
        assert tool.description
        assert "apenas quando o usuário pedir explicitamente" in tool.description


def test_phone_adapters_idempotente_sem_duplicar():
    reg = ToolRegistry()
    first = register_phone_tools(reg, _StubPhoneOK())
    second = register_phone_tools(reg, _StubPhoneOK())
    assert first == ["phone_battery", "phone_location", "phone_device_info", "phone_speak", "phone_vibrate"]
    assert second == []
    assert reg.count() == 5


# ---------------------------------------------------------------------------
# 2. Execução com sucesso
# ---------------------------------------------------------------------------


def test_phone_battery_execucao_sucesso():
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    router = ToolRouter(registry=reg)
    result = router.execute("phone_battery")
    assert result.success is True
    assert result.output == {"ok": True, "level": 85, "charging": False}
    assert result.error == ""


def test_phone_location_execucao_sucesso():
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    router = ToolRouter(registry=reg)
    result = router.execute("phone_location")
    assert result.success is True
    assert result.output["lat"] == -23.5
    assert result.output["lon"] == -46.6


def test_phone_device_info_execucao_sucesso():
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    router = ToolRouter(registry=reg)
    result = router.execute("phone_device_info")
    assert result.success is True
    assert result.output["model"] == "SM-A307F"


def test_phone_speak_execucao_sucesso_com_args():
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    router = ToolRouter(registry=reg)
    result = router.execute("phone_speak", {"text": "Olá do DaviOS"})
    assert result.success is True
    assert result.output == {"ok": True, "spoken": True}


def test_phone_vibrate_execucao_sucesso_com_args():
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    router = ToolRouter(registry=reg)
    result = router.execute("phone_vibrate", {"duration_ms": 500})
    assert result.success is True
    assert result.output == {"ok": True, "vibrated": True}


# ---------------------------------------------------------------------------
# 3. Falha do client vira erro padronizado (ToolExecutionResult success=False)
# ---------------------------------------------------------------------------


def test_phone_falha_vira_erro_padronizado():
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneFail())
    router = ToolRouter(registry=reg)

    for name in ("phone_battery", "phone_location", "phone_device_info", "phone_speak", "phone_vibrate"):
        args = {"text": "oi"} if name == "phone_speak" else ({"duration_ms": 100} if name == "phone_vibrate" else {})
        result = router.execute(name, args)
        assert result.success is False
        assert result.output is None
        assert result.error != ""
        # ToolRouter wraps ToolError as "ToolError: <message>"
        assert "toolerror" in result.error.lower()


# ---------------------------------------------------------------------------
# 4. REGRA DE OURO — cada condição falhando isoladamente
# ---------------------------------------------------------------------------


def test_regra_ouro_phone_read_flag_off_bloqueia():
    """phone_read_enabled=False → negação, execute() nunca chamado."""
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, phone_read_enabled=False, phone_act_enabled=True),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    outcome = orchestrator.handle_first_response(
        _block("phone_battery", "{}"),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert outcome.authorized is False
    assert outcome.tool_executed is False
    assert len(llm_calls) == 1  # 1 chamada: a da negação
    assert "nao foi autorizado" in llm_calls[0]
    assert outcome.denial_reason == "categoria 'phone_read' desabilitada"


def test_regra_ouro_phone_act_flag_off_bloqueia():
    """phone_act_enabled=False → negação, execute() nunca chamado."""
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, phone_read_enabled=True, phone_act_enabled=False),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    outcome = orchestrator.handle_first_response(
        _block("phone_speak", '{"text": "oi"}'),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert outcome.authorized is False
    assert outcome.tool_executed is False
    assert len(llm_calls) == 1
    assert "nao foi autorizado" in llm_calls[0]
    assert outcome.denial_reason == "categoria 'phone_act' desabilitada"


def test_regra_ouro_tools_visible_off_bloqueia_tudo():
    """tools_visible_to_llm=False → nem parse, nem execução (idêntico ao pré-C5)."""
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=False, phone_read_enabled=True, phone_act_enabled=True),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    outcome = orchestrator.handle_first_response(
        _block("phone_battery", "{}"),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert outcome.request_found is False  # nem chegou a parsear
    assert outcome.tool_executed is False
    assert llm_calls == []  # segunda chamada NUNCA acontece


def test_regra_ouro_ferramenta_inexistente_bloqueia():
    """Ferramenta não registrada → negação genérica, sem crash."""
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, phone_read_enabled=True, phone_act_enabled=True),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    outcome = orchestrator.handle_first_response(
        _block("phone_inexistente", '{"x": 1}'),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert outcome.tool_executed is False
    assert outcome.authorized is False
    assert len(llm_calls) == 1
    assert "nao foi autorizado" in llm_calls[0]


def test_regra_ouro_categoria_vazia_nunca_executa():
    """Categoria vazia/desconhecida → default seguro (nunca executa)."""
    reg = ToolRegistry()
    tool_calls: list = []

    def _execute(**kwargs):
        tool_calls.append(kwargs)
        return "ok"

    # Registra uma tool com category="" (default)
    reg.register(Tool(name="sem_categoria", description="Sem categoria.", execute=_execute))
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, phone_read_enabled=True, phone_act_enabled=True, actions_enabled=True, web_enabled=True),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    outcome = orchestrator.handle_first_response(
        _block("sem_categoria", "{}"),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert tool_calls == []
    assert outcome.tool_executed is False
    assert outcome.authorized is False
    assert "nao auto-executavel" in outcome.denial_reason


def test_regra_ouro_todas_condicoes_phone_read_executa():
    """Contraprova: phone_read ligado + tools_visible + tool existe → executa."""
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, phone_read_enabled=True),
        tool_router=ToolRouter(reg),
    )
    outcome = orchestrator.handle_first_response(
        _block("phone_battery", "{}"),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm([]),
    )
    assert outcome.tool_executed is True
    assert outcome.authorized is True
    assert outcome.execution_success is True


def test_regra_ouro_todas_condicoes_phone_act_executa():
    """Contraprova: phone_act ligado + tools_visible + tool existe → executa."""
    reg = ToolRegistry()
    register_phone_tools(reg, _StubPhoneOK())
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, phone_act_enabled=True),
        tool_router=ToolRouter(reg),
    )
    outcome = orchestrator.handle_first_response(
        _block("phone_speak", '{"text": "teste"}'),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm([]),
    )
    assert outcome.tool_executed is True
    assert outcome.authorized is True
    assert outcome.execution_success is True


# ---------------------------------------------------------------------------
# 5. PhoneClient com servidor indisponível/timeout/JSON inválido/401
#    (já coberto em test_phone_client.py, mas verifica integração aqui)
# ---------------------------------------------------------------------------


def test_phone_client_erro_rede_vira_toolerror_no_adapter():
    """Adapter captura falha de rede do client e vira ToolError padronizado."""
    class _StubRede:
        def get_battery(self):
            raise ConnectionError("rede fora")

    reg = ToolRegistry()
    # Monkey-patch: usa stub que levanta exceção no método
    from brain.phone_adapters import make_battery_tool
    tool = make_battery_tool(_StubRede())  # type: ignore
    reg.register(tool)
    router = ToolRouter(registry=reg)
    result = router.execute("phone_battery")
    assert result.success is False
    assert "rede" in result.error.lower() or "conectar" in result.error.lower()


# ---------------------------------------------------------------------------
# 6. Validação de speak/vibrate no adapter (já validado no client)
# ---------------------------------------------------------------------------


def _make_registry_with_real_client():
    """Cria registry com PhoneClient real e _request mockado para sucesso."""
    from brain.phone_client import PhoneClient

    def mock_request(method: str, path: str, json_body: dict | None = None) -> dict:
        return {"ok": True}

    client = PhoneClient("http://fake", token="t")
    client._request = mock_request  # type: ignore
    reg = ToolRegistry()
    register_phone_tools(reg, client)
    return reg


def test_phone_speak_argumentos_invalidos():
    """speak com texto muito longo → erro antes de chamar client (validação real)."""
    reg = _make_registry_with_real_client()
    router = ToolRouter(registry=reg)
    result = router.execute("phone_speak", {"text": "a" * 301})
    assert result.success is False
    assert "longo" in result.error.lower()


def test_phone_vibrate_argumentos_invalidos():
    """vibrate fora do range → erro antes de chamar client (validação real)."""
    reg = _make_registry_with_real_client()
    router = ToolRouter(registry=reg)
    result = router.execute("phone_vibrate", {"duration_ms": 49})
    assert result.success is False
    assert "50" in result.error
    result = router.execute("phone_vibrate", {"duration_ms": 2001})
    assert result.success is False
    assert "2000" in result.error


# ---------------------------------------------------------------------------
# 7. Ferramentas phone_* com category vazia continuam bloqueadas
# ---------------------------------------------------------------------------


def test_phone_tool_category_vazia_bloqueada():
    """Se por algum motivo uma tool phone_* tiver category='', bloqueia."""
    reg = ToolRegistry()
    tool_calls: list = []

    def _execute(**kwargs):
        tool_calls.append(kwargs)
        return "ok"

    # Simula tool com category vazia
    reg.register(Tool(name="phone_bad", description="Ruim", execute=_execute))
    orchestrator = ToolExecutionOrchestrator(
        _config(tools_visible_to_llm=True, phone_read_enabled=True, phone_act_enabled=True),
        tool_router=ToolRouter(reg),
    )
    llm_calls: list = []
    outcome = orchestrator.handle_first_response(
        _block("phone_bad", "{}"),
        original_prompt="P",
        original_system="S",
        run_llm=_recording_run_llm(llm_calls),
    )
    assert tool_calls == []
    assert outcome.tool_executed is False
    assert outcome.authorized is False