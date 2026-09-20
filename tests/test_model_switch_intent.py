"""Testes da troca de modelo por linguagem natural (etapa 6).

Escopo: a intencao MODEL_SWITCH e o caminho que ela percorre ate o provider.

O que este arquivo prova:
- ordem explicita vira `ModelSwitchRequest` estruturado, com o alvo AINDA em
  linguagem natural (o classificador nao resolve modelo nenhum);
- consulta, duvida, negacao e interrogacao NAO viram comando de troca;
- resolucao e execucao pertencem ao ModelManager
  (`resolve_model` + `switch_active_model`), nunca ao classificador;
- modelo inexistente nao e inventado, modelo nao instalado nao dispara
  download, nada e escolhido "no lugar do usuario" e falha do gerenciador nao
  e mascarada como sucesso.

Nenhum recurso externo:
- catalogo, config e arquivos .gguf vivem em tmp_path (nao tocam o projeto);
- os .gguf sao criados pelo proprio teste (nada e baixado);
- o provider e fake e apenas registra chamadas (nunca sobe llama-server);
- o MemoryManager usa um SQLite temporario.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from brain.conversation_engine import ConversationEngine
from brain.intent_classifier import (
    Intent,
    IntentClassifier,
    ModelSwitchRequest,
    extract_model_switch_request,
)
from brain.model_catalog import ModelCatalog
from brain.model_manager import ModelManager
from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareProfile
from memory.memory_manager import MemoryManager

GIB = 1024 ** 3

# IDs escolhidos para espelhar o catalogo real (qwen3-<tamanho>-<quant>).
SMALL = "qwen3-0.6b-q4km"    # instalado
MEDIUM = "qwen3-4b-q4km"     # instalado
BIG = "qwen3-8b-q4km"        # instalado
HUGE = "qwen3-14b-q4km"      # catalogado, NAO instalado

INSTALLED = (SMALL, MEDIUM, BIG)


def _catalog_payload() -> dict:
    """Catalogo sintetico do TESTE (nao depende do catalogo do projeto)."""
    base = {"family": "Qwen3", "format": "GGUF", "quantization": "Q4_K_M"}
    return {
        "version": 1,
        "models": [
            {
                **base,
                "id": SMALL, "name": "Qwen3 0.6B", "tier": "light",
                "parameters": "0.6B",
                "filename": "Qwen3-0.6B-Q4_K_M.gguf",
                "path": f"models/light/{SMALL}.gguf",
                "size_bytes": int(0.4 * GIB),
                "min_ram_gb": 1.0, "recommended_ram_gb": 2.0,
                "context_window": 8192, "enabled": True,
            },
            {
                **base,
                "id": MEDIUM, "name": "Qwen3 4B", "tier": "balanced",
                "parameters": "4B",
                "filename": "Qwen3-4B-Q4_K_M.gguf",
                "path": f"models/balanced/{MEDIUM}.gguf",
                "size_bytes": int(2.5 * GIB),
                "min_ram_gb": 4.0, "recommended_ram_gb": 6.0,
                "context_window": 8192, "enabled": True,
            },
            {
                **base,
                "id": BIG, "name": "Qwen3 8B", "tier": "strong",
                "parameters": "8B",
                "filename": "Qwen3-8B-Q4_K_M.gguf",
                "path": f"models/strong/{BIG}.gguf",
                "size_bytes": int(5.0 * GIB),
                "min_ram_gb": 8.0, "recommended_ram_gb": 10.0,
                "context_window": 8192, "enabled": True,
            },
            {
                **base,
                "id": HUGE, "name": "Qwen3 14B", "tier": "very_strong",
                "parameters": "14B",
                "filename": "Qwen3-14B-Q4_K_M.gguf",
                "path": f"models/very_strong/{HUGE}.gguf",
                "size_bytes": int(9.0 * GIB),
                "min_ram_gb": 12.0, "recommended_ram_gb": 16.0,
                "context_window": 8192, "enabled": True,
            },
        ],
    }


def _write_bytes(path: Path, size: int = 1024) -> None:
    """Cria o .gguf: arquivo com conteudo (tamanho 0 nao conta como instalado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def _hardware(
    ram_total_gb: float = 32.0,
    ram_available_gb: float = 24.0,
    disk_free_gb: float | None = 200.0,
) -> HardwareProfile:
    """Hardware fake, injetado: nenhum teste sonda a maquina real."""
    return HardwareProfile(
        os="Teste",
        cpu_name="CPU Fake",
        cpu_cores=8,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        gpu_name="GPU Fake",
        gpu_vram_gb=None,
        gpu_vendor="Fake",
        inference_backend="llama_cpp",
        disk_free_gb=disk_free_gb,
    )


class RecordingProvider:
    """Provider fake: registra chamadas; nunca inicia llama-server."""

    def __init__(self, switch_result: bool = True, switch_results=None) -> None:
        self.calls: list[str] = []
        self.switch_result = switch_result
        self.switch_results = switch_results
        self.switch_selections: list = []
        self.loaded_model: str | None = None

    def initialize(self) -> bool:
        self.calls.append("initialize")
        return True

    def unload(self) -> None:
        self.calls.append("unload")

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return True

    def switch_model(self, selection) -> bool:
        self.calls.append("switch_model")
        self.switch_selections.append(selection)
        if self.switch_results is not None:
            result = self.switch_results.pop(0) if self.switch_results else self.switch_result
        else:
            result = self.switch_result
        if result is True and selection is not None and selection.model is not None:
            # Simula o que o provider real faz: publica a identidade carregada.
            self.loaded_model = str(selection.model.path)
        return result


# ------------------------------------------------------------------------- #
# Fixtures
# ------------------------------------------------------------------------- #

@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Raiz isolada: catalogo, davios.json e .gguf ficam todos em tmp_path."""
    (tmp_path / "config").mkdir()
    for tier in ("light", "balanced", "strong", "very_strong"):
        (tmp_path / "models" / tier).mkdir(parents=True)

    (tmp_path / "config" / "model_catalog.json").write_text(
        json.dumps(_catalog_payload(), indent=2), encoding="utf-8"
    )
    # `active_model_id: ""` deixa explicito que NADA esta gravado: o teste nao
    # herda o active_model_id do config real do projeto.
    (tmp_path / "config" / "davios.json").write_text(
        json.dumps({
            "offline_mode": True,
            "backend": "llama_cpp_standalone",
            "active_model_id": "",
        }),
        encoding="utf-8",
    )
    payload = {m["id"]: m for m in _catalog_payload()["models"]}
    for model_id in INSTALLED:
        _write_bytes(tmp_path / payload[model_id]["path"])
    return tmp_path


@pytest.fixture
def temp_db():
    """SQLite temporario para o MemoryManager (nunca toca a memoria real)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except PermissionError:
            pass


def _make_manager(
    workspace: Path,
    provider=None,
    hardware: HardwareProfile | None = None,
) -> ModelManager:
    """ModelManager isolado em tmp_path, com provider fake ja anexado."""
    config = DaviosConfig.load()
    config.models_dir = str(workspace / "models")
    config.active_model_id = None
    catalog = ModelCatalog(
        config=config,
        catalog_path=str(workspace / "config" / "model_catalog.json"),
        project_root=workspace,
    )
    manager = ModelManager(
        config=config,
        catalog=catalog,
        config_path=str(workspace / "config" / "davios.json"),
    )
    manager._hardware = hardware or _hardware()
    if provider is not None:
        manager.attach_provider(provider)
    return manager


def _read_config(workspace: Path) -> dict:
    return json.loads((workspace / "config" / "davios.json").read_text("utf-8"))


def _make_engine(manager, temp_db):
    """Engine real, com memoria temporaria e o ModelManager sob teste."""
    memory = MemoryManager(db_path=temp_db)
    engine = ConversationEngine(memory_manager=memory, model_manager=manager)
    return engine, memory


# ------------------------------------------------------------------------- #
# TESTE 1-3 + negativos: camada de intencao (unit, sem execucao)
# ------------------------------------------------------------------------- #

class TestSwitchIntent:
    """A intencao e o alvo, isolados de qualquer execucao."""

    def test_explicit_command_produces_switch_intent(self):
        """TESTE 1: 'troca para o Qwen 8B' gera intencao de troca."""
        text = "troca para o Qwen 8B"
        classifier = IntentClassifier()

        assert classifier.classify(text) == Intent.MODEL_SWITCH
        request = extract_model_switch_request(text)
        assert isinstance(request, ModelSwitchRequest)
        assert request.action == "switch_model"
        # O alvo continua em LINGUAGEM NATURAL: resolver "8b" e do ModelManager.
        assert request.target == "qwen 8b"

    def test_vocative_prefix_is_accepted(self):
        """'Davi, troca para o Qwen 8B' e a mesma ordem, com vocativo."""
        request = extract_model_switch_request("Davi, troca para o Qwen 8B")
        assert request is not None
        assert request.action == "switch_model"
        assert request.target == "qwen 8b"

    def test_explicit_id_is_preserved(self):
        """TESTE 2: 'troque para qwen3-8b-q4km' preserva o ID."""
        text = "troque para qwen3-8b-q4km"
        request = extract_model_switch_request(text)
        assert request is not None
        assert request.target == "qwen3-8b-q4km"
        assert IntentClassifier().classify(text) == Intent.MODEL_SWITCH

    def test_uppercase_is_accepted(self):
        """TESTE 3: capitalizacao nao muda a intencao."""
        request = extract_model_switch_request("QUERO USAR O QWEN 8B")
        assert request is not None
        assert request.action == "switch_model"
        assert request.target == "qwen 8b"

    @pytest.mark.parametrize("text", [
        "troca para o Qwen 8B",
        "muda para o Qwen 4B",
        "quero usar o Qwen 14B",
        "usa o 0.6B",
        "troque o modelo para o 4B",
        "usa o Qwen 0.6B",
        "use o Qwen3 8B",
        "utilize o qwen3-8b-q4km",
    ])
    def test_recognized_orders(self, text):
        """As variacoes dos requisitos viram ModelSwitchRequest."""
        request = extract_model_switch_request(text)
        assert request is not None, text
        assert request.action == "switch_model"
        assert request.target, text

    @pytest.mark.parametrize("text", [
        "qual modelo esta usando?",
        "qual modelo e esse?",
        "me fale sobre o Qwen 8B",
        "qual e o melhor modelo?",
        "quais modelos estao disponiveis?",
        "me mostre os modelos instalados",
        "como trocar de modelo?",
        "e possivel trocar para o Qwen 8B?",
        "voce pode trocar o modelo para o Qwen 8B?",
    ])
    def test_questions_never_produce_switch_intent(self, text):
        """TESTE 6-8: consulta nao e ordem (nem no classificador)."""
        assert extract_model_switch_request(text) is None, text
        assert IntentClassifier().classify(text) != Intent.MODEL_SWITCH

    @pytest.mark.parametrize("text", [
        "acho que talvez eu devesse usar o Qwen 8B",
        "talvez seja melhor mudar para o Qwen 8B",
        "nao troca o modelo",
        "nao quero usar o Qwen 8B",
        "gostaria de saber se posso usar o Qwen 8B",
        "voce recomenda mudar para o Qwen 8B?",
    ])
    def test_doubt_and_negation_do_not_switch(self, text):
        """Prioridade de seguranca: na duvida, NAO se executa."""
        assert extract_model_switch_request(text) is None, text

    @pytest.mark.parametrize("text", [
        "troca de assunto",
        "usa o bom senso",
        "muda de ideia",
        "vamos trocar uma ideia",
    ])
    def test_unrelated_verbs_do_not_switch(self, text):
        """Sem evidencia de modelo, verbo generico nao vira comando."""
        assert extract_model_switch_request(text) is None, text

    @pytest.mark.parametrize("text", [
        "quero um modelo grande",
        "troca para um modelo grande",
        "quero usar um modelo melhor",
    ])
    def test_vague_target_asks_for_clarification(self, text):
        """Alvo vago nao escolhe modelo: gera pedido COM alvo vazio."""
        request = extract_model_switch_request(text)
        assert request is not None, text
        assert request.action == "switch_model"
        assert request.target == ""

    def test_unknown_qualifier_is_not_guessed(self):
        """'mais forte possivel' nao vira escolha: o alvo chega literal e quem
        decide se existe e o catalogo (resolucao, nao adivinhacao)."""
        request = extract_model_switch_request("muda para o modelo mais forte possivel")
        assert request is not None
        assert request.target == "mais forte possivel"

    def test_request_payload_is_serializable(self):
        """O pedido estruturado carrega acao, alvo e texto original."""
        request = extract_model_switch_request("muda para o Qwen 4B")
        payload = request.to_dict()
        assert payload["action"] == "switch_model"
        assert payload["target"] == "qwen 4b"
        assert payload["raw_text"] == "muda para o Qwen 4B"

    def test_classifier_does_not_resolve_models(self):
        """O classificador nao resolve IDs: o alvo sai como o usuario disse."""
        request = extract_model_switch_request("quero usar o Qwen 14B")
        assert request is not None
        assert request.target == "qwen 14b"
        assert request.target != HUGE



# ------------------------------------------------------------------------- #
# TESTE 4, 5, 9, 10: execucao ponta a ponta (intent -> ModelManager -> provider)
# ------------------------------------------------------------------------- #

class TestModelSwitchExecution:
    """A ordem do usuario passa pelo ModelManager ate o provider fake."""

    def test_success_reaches_manager_and_provider(self, workspace, temp_db):
        """TESTE 9: a troca bem-sucedida chega a switch_active_model()."""
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("Davi, troca para o Qwen 8B")

        assert result.intent == "model_switch"
        outcome = result.model_switch
        assert outcome["success"] is True
        assert outcome["status"] == "switched"
        assert outcome["requested_model"] == "qwen 8b"      # como o usuario disse
        assert outcome["resolved_model"] == BIG             # resolvido no catalogo
        assert outcome["previous_model"] == MEDIUM
        assert outcome["error"] is None
        # Uma UNICA troca fisica, e ela veio do ModelManager (nao do classifier).
        assert provider.calls == ["switch_model"]
        assert len(provider.switch_selections) == 1
        assert provider.switch_selections[0].model.path.name == f"{BIG}.gguf"
        assert _read_config(workspace)["active_model_id"] == BIG
        assert manager.get_active_model().id == BIG
        assert result.response == "Modelo alterado para Qwen3 8B."
        memory.database.close()

    def test_switch_goes_through_manager_entrypoint(self, workspace, temp_db):
        """TESTE 9 (explicito): a execucao passa por switch_active_model()."""
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        seen: list = []
        original = manager.switch_active_model

        def _spy(request):
            seen.append(request)
            return original(request)

        manager.switch_active_model = _spy          # espiao, nao substituicao
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("usa o Qwen 8B")

        # O pedido chegou ao metodo oficial do ModelManager, uma unica vez...
        assert len(seen) == 1
        assert seen[0].id == BIG
        assert result.model_switch["success"] is True
        # ...e o provider nao foi chamado por fora (o classifier nao executa).
        assert provider.calls == ["switch_model"]
        assert provider.switch_selections[0].model.path.name == f"{BIG}.gguf"
        memory.database.close()

    def test_explicit_id_order_also_switches(self, workspace, temp_db):
        """ID explicito percorre o mesmo caminho, sem atalhos."""
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("troque para qwen3-8b-q4km")

        assert result.model_switch["success"] is True
        assert result.model_switch["resolved_model"] == BIG
        assert provider.calls == ["switch_model"]
        memory.database.close()

    def test_uppercase_order_switches(self, workspace, temp_db):
        """TESTE 3 (ponta a ponta): capitalizacao nao impede a troca."""
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("QUERO USAR O QWEN 8B")

        assert result.model_switch["success"] is True
        assert result.model_switch["resolved_model"] == BIG
        assert provider.calls == ["switch_model"]
        memory.database.close()

    def test_unknown_model_never_calls_provider(self, workspace, temp_db):
        """TESTE 4: modelo inexistente nao e inventado nem baixado."""
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("troca para Qwen 100B")

        outcome = result.model_switch
        assert outcome["success"] is False
        assert outcome["status"] == "not_found"
        assert outcome["resolved_model"] is None
        assert outcome["requested_model"] == "qwen 100b"
        assert outcome["error"]
        assert provider.calls == []
        assert provider.switch_selections == []
        # Nada mudou: nem registro, nem modelo ativo.
        assert _read_config(workspace)["active_model_id"] == MEDIUM
        assert manager.get_active_model().id == MEDIUM
        assert "nao encontrei esse modelo" in result.response.lower()
        memory.database.close()

    def test_catalogued_but_not_installed_never_calls_provider(
        self, workspace, temp_db
    ):
        """TESTE 5: catalogado e ausente no disco -> reconhecido, sem troca."""
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("troca para Qwen 14B")

        outcome = result.model_switch
        assert outcome["success"] is False
        assert outcome["status"] == "not_installed"
        assert outcome["resolved_model"] == HUGE      # reconhecido no catalogo
        assert outcome["previous_model"] is None      # nem chegou a consultar
        assert provider.calls == []
        assert _read_config(workspace)["active_model_id"] == MEDIUM
        assert "nao esta instalado" in result.response
        memory.database.close()

    def test_vague_order_asks_instead_of_choosing(self, workspace, temp_db):
        """Nada e escolhido pelo usuario: pedido vago vira esclarecimento."""
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("quero um modelo grande")

        outcome = result.model_switch
        assert outcome["success"] is False
        assert outcome["status"] == "need_target"
        assert outcome["resolved_model"] is None
        assert provider.calls == []
        assert _read_config(workspace)["active_model_id"] in ("", None)
        memory.database.close()

    def test_provider_failure_is_not_masked(self, workspace, temp_db):
        """TESTE 10: falha do gerenciador chega intacta a camada superior."""
        provider = RecordingProvider(switch_result=False)
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("troca para o Qwen 8B")

        outcome = result.model_switch
        assert outcome["success"] is False
        assert outcome["status"] == "switch_failed"
        assert outcome["resolved_model"] == BIG
        assert outcome["previous_model"] == MEDIUM
        # O motivo REAL do ModelManager, sem texto generico por cima.
        assert outcome["error"] == manager.last_error
        assert "recusada pelo provider" in outcome["error"]
        # Uma unica tentativa fisica; o restante das chamadas e diagnostico.
        assert provider.calls.count("switch_model") == 1
        assert "unload" not in provider.calls
        assert "initialize" not in provider.calls
        assert _read_config(workspace)["active_model_id"] == MEDIUM
        assert manager.get_active_model().id == MEDIUM
        assert "anterior foi preservado" in result.response
        memory.database.close()

    def test_provider_exception_is_not_masked(self, workspace, temp_db):
        """Falha fisica com excecao: erro preservado, modelo anterior mantido."""
        provider = RecordingProvider()
        provider.switch_model = _raising_switch(provider)
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("usa o Qwen 8B")

        outcome = result.model_switch
        assert outcome["success"] is False
        assert outcome["status"] == "switch_failed"
        assert "provider falhou durante a troca" in outcome["error"]
        assert _read_config(workspace)["active_model_id"] == MEDIUM
        memory.database.close()

    def test_divergence_after_failed_persist_and_rollback_is_reported(
        self, workspace, temp_db
    ):
        """Persistencia falhou E rollback recusado: divergencia explicitada."""
        provider = RecordingProvider(switch_results=[True, False])
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        # Falha de persistencia simulada: a troca fisica acontece, o registro
        # nao; o rollback e recusado pelo provider.
        manager._write_persisted_active_id = lambda model_id: False
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process("muda para o Qwen 8B")

        outcome = result.model_switch
        assert outcome["success"] is False
        assert outcome["status"] == "divergent"
        assert "divergente" in outcome["error"]
        assert provider.calls.count("switch_model") == 2  # troca + rollback tentado
        assert provider.calls[:2] == ["switch_model", "switch_model"]
        assert manager.get_status()["active_model_matches_loaded"] is False
        assert "divergentes" in result.response
        memory.database.close()

    def test_engine_without_manager_reports_unavailable(self, temp_db):
        """Sem ModelManager na sessao, nada e executado nem inventado."""
        memory = MemoryManager(db_path=temp_db)
        engine = ConversationEngine(memory_manager=memory)

        result = engine.process("troca para o Qwen 8B")

        assert result.intent == "model_switch"
        assert result.model_switch["success"] is False
        assert result.model_switch["status"] == "unavailable"
        assert result.model_switch["error"]
        memory.database.close()


def _raising_switch(provider: RecordingProvider):
    """switch_model que levanta excecao, mantendo o registro da chamada."""

    def _switch(selection):
        provider.calls.append("switch_model")
        provider.switch_selections.append(selection)
        raise RuntimeError("backend caiu no meio da troca")

    return _switch


# ------------------------------------------------------------------------- #
# TESTE 6-8: consultas informativas nao executam troca
# ------------------------------------------------------------------------- #

class TestInformationalMessagesDoNotSwitch:
    """Consulta != acao: perguntar sobre modelo nao muda o runtime."""

    @pytest.mark.parametrize("text", [
        "qual modelo esta usando?",
        "qual modelo e esse?",
        "me fale sobre o Qwen 8B",
        "qual e o melhor modelo?",
        "quais modelos estao disponiveis?",
        "me mostre os modelos instalados",
        "como trocar de modelo?",
        "acho que talvez eu devesse usar o Qwen 8B",
    ])
    def test_query_does_not_touch_manager_or_provider(self, workspace, temp_db, text):
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        engine, memory = _make_engine(manager, temp_db)

        result = engine.process(text)

        assert result.intent != "model_switch", text
        assert result.model_switch is None, text
        assert provider.calls == [], text
        assert _read_config(workspace)["active_model_id"] == MEDIUM
        assert manager.get_active_model().id == MEDIUM
        memory.database.close()


# ------------------------------------------------------------------------- #
# O LLM nunca escolhe nem executa a troca
# ------------------------------------------------------------------------- #

class _StubLLMProvider:
    """Provider minimo: apenas diz que esta disponivel."""

    name = "stub"

    def is_available(self) -> bool:
        return True


class RecordingCognitiveCore:
    """CognitiveCore fake: registra se o LLM foi consultado."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.llm_provider = _StubLLMProvider()

    def generate_response(self, *args, **kwargs) -> dict:
        self.calls.append("generate_response")
        return {"text": "Resposta do LLM falso.", "memories_used": []}


class TestLlmNeverDrivesTheSwitch:
    """A ordem vira intencao estruturada; o LLM nao gera chamada de troca."""

    def test_llm_is_not_consulted_for_switch_order(self, workspace, temp_db):
        """Ordem explicita e atendida pela camada de modelos, sem LLM."""
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        memory = MemoryManager(db_path=temp_db)
        core = RecordingCognitiveCore()
        engine = ConversationEngine(
            memory_manager=memory, cognitive_core=core, model_manager=manager
        )

        result = engine.process("muda para o Qwen 8B")

        assert result.model_switch["status"] == "switched"
        assert provider.calls == ["switch_model"]
        # O LLM nao foi perguntado "qual modelo usar": nao houve eleicao.
        assert core.calls == []
        memory.database.close()

    def test_regular_conversation_still_reaches_the_llm(self, workspace, temp_db):
        """Regressao: a etapa 6 nao transforma toda mensagem em comando."""
        provider = RecordingProvider()
        manager = _make_manager(workspace, provider=provider)
        manager.set_active_model(MEDIUM)
        memory = MemoryManager(db_path=temp_db)
        core = RecordingCognitiveCore()
        engine = ConversationEngine(
            memory_manager=memory, cognitive_core=core, model_manager=manager
        )

        result = engine.process("quem foi alan turing?")

        assert result.intent != "model_switch"
        assert result.model_switch is None
        assert core.calls == ["generate_response"]
        # Nenhuma troca de modelo foi disparada por uma pergunta.
        assert provider.calls == []
        memory.database.close()


