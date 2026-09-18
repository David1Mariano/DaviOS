"""Testes do ModelManager (camada de decisão sobre modelos).

Escopo: catálogo carregado, listagem, resolução de id/alias, detecção de
instalado, modelo ativo + persistência, compatibilidade e os estados
inconsistentes (ativo inexistente / não instalado / indefinido).

Nenhum recurso externo é usado:
- catálogo, config e arquivos .gguf vivem em tmp_path (não tocam o projeto);
- o hardware é um HardwareProfile FAKE, injetado (não sonda a máquina real);
- NÃO se inicia llama-server, NÃO se baixa nada e NÃO se usa rede;
- o provider é um fake que apenas registra chamadas.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from brain.model_catalog import ModelCatalog
from brain.model_manager import (
    ACTIVE_STATE_AUTO,
    ACTIVE_STATE_MISSING,
    ACTIVE_STATE_NONE,
    ACTIVE_STATE_NOT_INSTALLED,
    ACTIVE_STATE_PERSISTED,
    LEVEL_INCOMPATIBLE,
    LEVEL_POSSIBLE,
    LEVEL_RECOMMENDED,
    LEVEL_RISKY,
    LEVEL_UNKNOWN,
    ModelManager,
)
from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareProfile

GIB = 1024 ** 3

SMALL = "modelo-pequeno-q4km"      # instalado, cabe folgado
BIG = "modelo-grande-q4km"         # pede 24 GB de RAM
HUGE = "modelo-enorme-q4km"        # 50 GB: não cabe no disco
MYSTERY = "modelo-sem-metadados"   # sem size/requisitos -> unknown


def _catalog_payload() -> dict:
    """Catálogo sintético com exatamente os casos que interessam.

    Os valores são do TESTE, não do projeto: assim a asserção não depende de
    como o catálogo real for editado no futuro.
    """
    return {
        "version": 1,
        "models": [
            {
                "id": SMALL,
                "name": "Modelo Pequeno",
                "family": "Teste",
                "tier": "light",
                "quantization": "Q4_K_M",
                "parameters": "1B",
                "filename": "modelo-pequeno-q4km.gguf",
                "path": "models/light/modelo-pequeno-q4km.gguf",
                "size_bytes": int(1.0 * GIB),
                "min_ram_gb": 2.0,
                "recommended_ram_gb": 4.0,
                "context_window": 8192,
                "enabled": True,
            },
            {
                "id": BIG,
                "name": "Modelo Grande",
                "family": "Teste",
                "tier": "strong",
                "quantization": "Q4_K_M",
                "parameters": "8B",
                "filename": "modelo-grande-q4km.gguf",
                "path": "models/strong/modelo-grande-q4km.gguf",
                "size_bytes": int(5.0 * GIB),
                "min_ram_gb": 24.0,
                "recommended_ram_gb": 32.0,
                "enabled": True,
            },
            {
                "id": HUGE,
                "name": "Modelo Enorme",
                "family": "Teste",
                "tier": "very_strong",
                "quantization": "Q4_K_M",
                "parameters": "32B",
                "filename": "modelo-enorme-q4km.gguf",
                "path": "models/very_strong/modelo-enorme-q4km.gguf",
                "size_bytes": int(50.0 * GIB),
                "min_ram_gb": 4.0,
                "recommended_ram_gb": 8.0,
                "enabled": True,
            },
            {
                "id": MYSTERY,
                "name": "Modelo Sem Metadados",
                "family": "Teste",
                "tier": "balanced",
                "filename": "modelo-sem-metadados.gguf",
                "path": "models/balanced/modelo-sem-metadados.gguf",
                "enabled": True,
            },
        ],
    }


# --------------------------------------------------------------------- #
# Helpers e fixtures
# --------------------------------------------------------------------- #

def _write_bytes(path: Path, size_bytes: int = 64 * 1024) -> Path:
    """Cria um arquivo do tamanho pedido, sem alocar tudo em memória."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"\0" * size_bytes)
    return path


def _hardware(
    ram_total_gb: float = 16.0,
    ram_available_gb: float = 12.0,
    disk_free_gb: float | None = 100.0,
    backend: str = "llama_cpp",
) -> HardwareProfile:
    """Hardware fake: valores explícitos, sem sondar a máquina real."""
    return HardwareProfile(
        os="Teste",
        cpu_name="CPU Fake",
        cpu_cores=8,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
        gpu_name="GPU Fake",
        gpu_vram_gb=None,
        gpu_vendor="Fake",
        inference_backend=backend,
        disk_free_gb=disk_free_gb,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Raiz isolada: catálogo, davios.json e .gguf ficam todos em tmp_path."""
    (tmp_path / "config").mkdir()
    for tier in ("light", "balanced", "strong", "very_strong"):
        (tmp_path / "models" / tier).mkdir(parents=True)

    (tmp_path / "config" / "model_catalog.json").write_text(
        json.dumps(_catalog_payload(), indent=2), encoding="utf-8"
    )
    (tmp_path / "config" / "davios.json").write_text(
        json.dumps({"offline_mode": True, "backend": "llama_cpp_standalone"}),
        encoding="utf-8",
    )
    return tmp_path


def _catalog_for(workspace: Path) -> ModelCatalog:
    config = DaviosConfig.load()
    config.models_dir = str(workspace / "models")
    return ModelCatalog(
        config=config,
        catalog_path=str(workspace / "config" / "model_catalog.json"),
        project_root=workspace,
    )


def _make_manager(
    workspace: Path,
    hardware: HardwareProfile | None = None,
    install: tuple[str, ...] = (),
) -> ModelManager:
    """ModelManager isolado em tmp_path.

    `install` cria os arquivos .gguf indicados, controlando o que conta como
    instalado sem depender de nada que exista no disco real.
    """
    payload = {m["id"]: m for m in _catalog_payload()["models"]}
    for model_id in install:
        _write_bytes(workspace / payload[model_id]["path"])

    config = DaviosConfig.load()
    config.models_dir = str(workspace / "models")
    manager = ModelManager(
        config=config,
        catalog=_catalog_for(workspace),
        config_path=str(workspace / "config" / "davios.json"),
    )
    manager._hardware = hardware or _hardware()
    return manager


def _read_config(workspace: Path) -> dict:
    return json.loads((workspace / "config" / "davios.json").read_text("utf-8"))


class RecordingProvider:
    """Provider fake: registra chamadas para provar que nada é iniciado."""

    def __init__(self, available: bool = True) -> None:
        self.calls: list[str] = []
        self.available = available

    def initialize(self) -> bool:
        self.calls.append("initialize")
        return True

    def unload(self) -> None:
        self.calls.append("unload")

    def is_available(self) -> bool:
        self.calls.append("is_available")
        return self.available

# --------------------------------------------------------------------- #
# Catálogo carregado, listagens e resolução
# --------------------------------------------------------------------- #

class TestCatalogIntegration:
    def test_catalog_is_the_shared_instance(self, workspace):
        """O ModelManager CONSOME o catálogo; não cria um catálogo paralelo."""
        manager = _make_manager(workspace)
        assert manager.get_catalog() is manager.catalog
        assert manager.catalog.catalog_path == (
            workspace / "config" / "model_catalog.json"
        )

    def test_list_models_returns_all_known(self, workspace):
        manager = _make_manager(workspace)
        assert len(manager.list_models()) == 4

    def test_list_models_sorted_by_size(self, workspace):
        manager = _make_manager(workspace)
        assert [m.id for m in manager.list_models()] == [MYSTERY, SMALL, BIG, HUGE]

    def test_list_models_only_installed(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        assert [m.id for m in manager.list_models(only_installed=True)] == [SMALL]

    def test_get_model_by_id(self, workspace):
        manager = _make_manager(workspace)
        assert manager.get_model(BIG).name == "Modelo Grande"

    def test_get_model_unknown_id_returns_none(self, workspace):
        manager = _make_manager(workspace)
        assert manager.get_model("nao-existe") is None

    def test_resolve_model_accepts_modelinfo(self, workspace):
        manager = _make_manager(workspace)
        model = manager.get_model(SMALL)
        assert manager.resolve_model(model) is model

    def test_resolve_model_by_exact_id(self, workspace):
        manager = _make_manager(workspace)
        assert manager.resolve_model(SMALL).id == SMALL

    def test_resolve_model_by_tier_alias_installed_wins(self, workspace):
        """'leve' com o modelo leve instalado devolve o instalado."""
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.resolve_model("leve").id == SMALL

    def test_resolve_model_by_tier_alias_without_install(self, workspace):
        """'forte' sem nada instalado ainda devolve o menor do tier (candidato)."""
        manager = _make_manager(workspace)
        assert manager.resolve_model("forte").id == BIG

    def test_resolve_model_by_parameter_and_family(self, workspace):
        """'qwen3 8b' é genérico: aqui o equivalente é 'qwen3' + '8b'."""
        manager = _make_manager(workspace)
        assert manager.resolve_model("8b").id == BIG

    def test_resolve_model_unknown_returns_none(self, workspace):
        """Sem casamento no catálogo, a resposta é None — nada é inventado."""
        manager = _make_manager(workspace)
        assert manager.resolve_model("modelo alienígena") is None

    def test_resolve_model_empty_returns_none(self, workspace):
        manager = _make_manager(workspace)
        assert manager.resolve_model("") is None
        assert manager.resolve_model(None) is None


class TestInstalledDetection:
    def test_installed_model_detected(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.catalog.is_model_installed(manager.get_model(SMALL))
        assert [m.id for m in manager.list_installed_models()] == [SMALL]

    def test_not_installed_model_detected(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        assert not manager.catalog.is_model_installed(manager.get_model(BIG))

    def test_empty_file_is_not_installed(self, workspace):
        """Arquivo de 0 byte NÃO conta como instalado."""
        _write_bytes(
            workspace / "models" / "strong" / "modelo-grande-q4km.gguf", size_bytes=0
        )
        manager = _make_manager(workspace)
        assert not manager.catalog.is_model_installed(manager.get_model(BIG))

    def test_partial_download_is_not_installed(self, workspace):
        """O .part do downloader fica fora do caminho final e não conta."""
        _write_bytes(workspace / "models" / "strong" / "modelo-grande-q4km.gguf.part")
        manager = _make_manager(workspace)
        assert manager.list_installed_models() == []

    def test_known_but_not_installed_listing(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        assert [m.id for m in manager.list_available_to_install()] == [
            MYSTERY, BIG, HUGE
        ]

# --------------------------------------------------------------------- #
# Compatibilidade com o hardware (estimativas != requisitos absolutos)
# --------------------------------------------------------------------- #

class TestCompatibility:
    def test_recommended_when_everything_fits(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        compat = manager.check_compatibility(SMALL)
        assert compat.level == LEVEL_RECOMMENDED
        assert compat.compatible is True
        assert compat.status == "compatible"
        assert compat.errors == []
        assert compat.ram_ok and compat.disk_ok
        assert compat.model_id == SMALL

    def test_incompatible_when_total_ram_below_minimum(self, workspace):
        manager = _make_manager(workspace)
        compat = manager.check_compatibility(BIG)
        assert compat.level == LEVEL_INCOMPATIBLE
        assert compat.compatible is False
        assert compat.ram_ok is False
        assert any("RAM total insuficiente" in e for e in compat.errors)

    def test_risky_when_free_ram_below_minimum(self, workspace):
        """RAM total serve, mas não há RAM livre agora: arriscado, não fatal."""
        manager = _make_manager(
            workspace,
            hardware=_hardware(ram_total_gb=32.0, ram_available_gb=3.0),
        )
        compat = manager.check_compatibility(BIG)
        assert compat.level == LEVEL_RISKY
        assert compat.compatible is True
        assert compat.ram_ok is False
        assert any("Pouca RAM livre" in w for w in compat.warnings)

    def test_possible_when_free_ram_below_recommended(self, workspace):
        manager = _make_manager(
            workspace,
            hardware=_hardware(ram_total_gb=16.0, ram_available_gb=3.0),
            install=(SMALL,),
        )
        compat = manager.check_compatibility(SMALL)
        assert compat.level == LEVEL_POSSIBLE
        assert compat.compatible is True
        assert any("abaixo do recomendado" in w for w in compat.warnings)

    def test_incompatible_when_disk_too_small_for_download(self, workspace):
        """Modelo ausente + disco insuficiente: nem começa o download."""
        manager = _make_manager(workspace, hardware=_hardware(disk_free_gb=10.0))
        compat = manager.check_compatibility(HUGE)
        assert compat.level == LEVEL_INCOMPATIBLE
        assert compat.disk_ok is False
        assert any("Espaço insuficiente" in e for e in compat.errors)
        assert compat.required_disk_gb > 0

    def test_installed_model_is_not_blocked_by_disk(self, workspace):
        """Já instalado: não faz sentido exigir espaço de download."""
        manager = _make_manager(
            workspace,
            hardware=_hardware(disk_free_gb=0.4),
            install=(SMALL,),
        )
        compat = manager.check_compatibility(SMALL)
        assert compat.level != LEVEL_INCOMPATIBLE
        assert compat.disk_ok is True

    def test_tight_disk_is_only_a_warning(self, workspace):
        """Cabe, mas apertado: warning, não impedimento."""
        manager = _make_manager(
            workspace,
            hardware=_hardware(
                ram_total_gb=64.0, ram_available_gb=48.0, disk_free_gb=5.5
            ),
        )
        compat = manager.check_compatibility(BIG)
        assert compat.disk_ok is True
        assert compat.level != LEVEL_INCOMPATIBLE
        assert any("apertado" in w for w in compat.warnings)

    def test_vram_never_blocks_by_estimate(self, workspace):
        """VRAM estimada gera warning; nunca 'incompatible'."""
        manager = _make_manager(workspace, install=(SMALL,))
        model = manager.get_model(SMALL)
        model.recommended_vram_gb = 8.0
        compat = manager.check_compatibility(model)
        assert compat.level != LEVEL_INCOMPATIBLE
        assert compat.vram_ok is True
        assert any("VRAM" in w for w in compat.warnings)

    def test_unknown_when_metadata_is_missing(self, workspace):
        """Sem tamanho nem requisitos: não afirmamos nada."""
        manager = _make_manager(workspace)
        compat = manager.check_compatibility(MYSTERY)
        assert compat.level == LEVEL_UNKNOWN
        assert compat.compatible is None
        assert any("não declara tamanho" in w for w in compat.warnings)

    def test_unknown_for_unknown_model_name(self, workspace):
        manager = _make_manager(workspace)
        compat = manager.check_compatibility("modelo fantasma")
        assert compat.level == LEVEL_UNKNOWN
        assert compat.compatible is None
        assert any("não encontrado" in e for e in compat.errors)

    def test_disk_not_detected_is_a_warning(self, workspace):
        manager = _make_manager(workspace, hardware=_hardware(disk_free_gb=None))
        compat = manager.check_compatibility(BIG)
        assert compat.disk_ok is True
        assert any("não foi detectado" in w for w in compat.warnings)

    def test_unknown_backend_is_a_warning(self, workspace):
        manager = _make_manager(
            workspace, hardware=_hardware(backend="unknown"), install=(SMALL,)
        )
        compat = manager.check_compatibility(SMALL)
        assert compat.backend_ok is False
        assert compat.level != LEVEL_INCOMPATIBLE

    def test_accepts_alias_and_explicit_hardware(self, workspace):
        manager = _make_manager(workspace)
        compat = manager.check_compatibility(
            "forte", _hardware(ram_total_gb=64.0, ram_available_gb=48.0)
        )
        assert compat.model_id == BIG
        assert compat.ram_ok is True

    def test_result_is_serializable(self, workspace):
        manager = _make_manager(workspace)
        data = manager.check_compatibility(BIG).to_dict()
        assert data["level"] == LEVEL_INCOMPATIBLE
        assert data["reasons"]
        assert json.dumps(data)  # nada não-serializável
# --------------------------------------------------------------------- #
# Modelo ativo: escolha inicial e estados inconsistentes
# --------------------------------------------------------------------- #

def _write_config(workspace: Path, data: dict) -> None:
    """Reescreve o davios.json do workspace de teste."""
    (workspace / "config" / "davios.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


class TestActiveModel:
    def test_auto_choice_when_nothing_is_persisted(self, workspace):
        """Sem registro: escolha determinística entre os INSTALADOS."""
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.get_active_model().id == SMALL
        assert manager.active_state == ACTIVE_STATE_AUTO

    def test_auto_choice_does_not_write_config(self, workspace):
        """Subir o DaviOS não altera arquivos sem necessidade."""
        manager = _make_manager(workspace, install=(SMALL,))
        manager.get_active_model()
        assert "active_model_id" not in _read_config(workspace)

    def test_auto_prefers_best_compatibility_over_smaller_size(self, workspace):
        """MYSTERY é menor, mas é 'unknown'; SMALL é 'recommended' e ganha."""
        manager = _make_manager(workspace, install=(SMALL, MYSTERY))
        assert manager.get_active_model().id == SMALL

    def test_auto_picks_larger_when_equally_compatible(self, workspace):
        manager = _make_manager(workspace, install=(SMALL, HUGE))
        assert manager.get_active_model().id == HUGE

    def test_auto_never_picks_a_non_installed_model(self, workspace):
        """Escolher um modelo ausente deixaria o DaviOS sem cérebro."""
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.get_active_model().id != BIG

    def test_none_when_no_model_is_installed(self, workspace):
        manager = _make_manager(workspace)
        assert manager.get_active_model() is None
        assert manager.active_state == ACTIVE_STATE_NONE

    def test_persisted_active_model_is_used(self, workspace):
        _write_config(workspace, {"active_model_id": SMALL})
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.get_active_model().id == SMALL
        assert manager.active_state == ACTIVE_STATE_PERSISTED

    def test_persisted_id_missing_from_catalog(self, workspace):
        """Não fingimos que existe modelo ativo."""
        _write_config(workspace, {"active_model_id": "fantasma"})
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.get_active_model() is None
        assert manager.active_state == ACTIVE_STATE_MISSING
        assert "não existe no catálogo" in manager.last_error

    def test_persisted_id_not_installed(self, workspace):
        _write_config(workspace, {"active_model_id": BIG})
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.get_active_model() is None
        assert manager.active_state == ACTIVE_STATE_NOT_INSTALLED
        assert "não está instalado" in manager.last_error

    def test_select_initial_model_prefers_persisted(self, workspace):
        _write_config(workspace, {"active_model_id": SMALL})
        manager = _make_manager(workspace, install=(SMALL, HUGE))
        assert manager.select_initial_model().id == SMALL
        assert manager.active_state == ACTIVE_STATE_PERSISTED

    def test_select_initial_model_falls_back_to_auto(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.select_initial_model().id == SMALL
        assert manager.active_state == ACTIVE_STATE_AUTO


class TestActiveModelPersistence:
    def test_set_active_model_writes_config(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.set_active_model(SMALL).id == SMALL
        assert _read_config(workspace)["active_model_id"] == SMALL
        assert manager.active_state == ACTIVE_STATE_PERSISTED

    def test_set_active_model_preserves_other_keys(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        manager.set_active_model(SMALL)
        data = _read_config(workspace)
        assert data["offline_mode"] is True
        assert data["backend"] == "llama_cpp_standalone"

    def test_set_active_model_leaves_no_temp_file(self, workspace):
        """Escrita atômica: o temporário é renomeado, não sobra lixo."""
        manager = _make_manager(workspace, install=(SMALL,))
        manager.set_active_model(SMALL)
        assert list((workspace / "config").glob(".davios-*.tmp")) == []

    def test_set_active_model_by_alias(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.set_active_model("leve").id == SMALL

    def test_set_active_model_survives_new_manager(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        manager.set_active_model(SMALL)

        reopened = _make_manager(workspace, install=(SMALL,))
        assert reopened.active_model_id == SMALL
        assert reopened.get_active_model().id == SMALL

    def test_set_active_unknown_is_refused(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.set_active_model("modelo fantasma") is None
        assert "active_model_id" not in _read_config(workspace)
        assert "não reconhecido" in manager.last_error

    def test_set_active_not_installed_is_refused(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.set_active_model(BIG) is None
        assert "active_model_id" not in _read_config(workspace)
        assert "não está instalado" in manager.last_error

    def test_set_active_incompatible_is_refused(self, workspace):
        """Instalado, mas não roda nesta máquina: recusa antes de persistir."""
        manager = _make_manager(
            workspace,
            hardware=_hardware(ram_total_gb=8.0, ram_available_gb=6.0),
            install=(SMALL, BIG),
        )
        manager.set_active_model(SMALL)
        assert manager.set_active_model(BIG) is None
        assert "incompatível" in manager.last_error

    def test_previous_active_is_kept_after_refusal(self, workspace):
        """Uma troca recusada não pode derrubar o modelo ativo atual."""
        manager = _make_manager(
            workspace,
            hardware=_hardware(ram_total_gb=8.0, ram_available_gb=6.0),
            install=(SMALL, BIG),
        )
        manager.set_active_model(SMALL)
        manager.set_active_model(BIG)
        assert _read_config(workspace)["active_model_id"] == SMALL
        assert manager.get_active_model().id == SMALL

    def test_corrupt_config_does_not_crash(self, workspace):
        (workspace / "config" / "davios.json").write_text("{{{ nao e json", "utf-8")
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.get_active_model().id == SMALL  # cai no modo automático
        assert manager.set_active_model(SMALL).id == SMALL
        assert _read_config(workspace)["active_model_id"] == SMALL

    def test_config_without_active_model_id(self, workspace):
        _write_config(workspace, {"offline_mode": False})
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.active_model_id is None
        assert manager.get_active_model().id == SMALL


# --------------------------------------------------------------------- #
# Efeitos colaterais: nada de llama-server, nada de download
# --------------------------------------------------------------------- #

class TestNoSideEffects:
    def test_manager_never_touches_the_provider(self, workspace):
        """Consultar/definir o ativo é lógica pura: o provider não é chamado."""
        manager = _make_manager(workspace, install=(SMALL,))
        provider = RecordingProvider()
        manager.attach_provider(provider)

        manager.list_models()
        manager.get_active_model()
        manager.check_compatibility(SMALL)
        manager.set_active_model(SMALL)
        manager.build_selection()

        # is_available() é leitura de health check; initialize/unload NÃO podem
        # aparecer: iniciar ou parar o llama-server é de outra etapa.
        assert "initialize" not in provider.calls
        assert "unload" not in provider.calls

    def test_refused_switch_does_not_touch_the_provider(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        provider = RecordingProvider()
        manager.attach_provider(provider)
        assert manager.set_active_model(BIG) is None  # não instalado
        assert provider.calls == []

    def test_nothing_here_downloads(self, workspace):
        """Nenhum modelo do catálogo tem origem de download confirmada."""
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.catalog.get_downloadable_models() == []
        for model in manager.list_available_to_install():
            assert model.download_available is False
            assert model.download_url == ""

    def test_set_active_creates_no_extra_files(self, workspace):
        """Trocar o ativo não deixa arquivo novo atrás (nem temporário)."""
        manager = _make_manager(workspace, install=(SMALL,))
        before = {p for p in workspace.rglob("*") if p.is_file()}
        manager.set_active_model(SMALL)
        after = {p for p in workspace.rglob("*") if p.is_file()}
        assert after == before
        assert list(workspace.rglob("*.part")) == []
        assert list(workspace.rglob("*.tmp")) == []

    def test_status_reports_provider_state(self, workspace):
        manager = _make_manager(workspace, install=(SMALL,))
        manager.set_active_model(SMALL)

        status = manager.get_status()
        assert status["active_model_id"] == SMALL
        assert status["active_model_state"] == ACTIVE_STATE_PERSISTED
        assert status["provider_running"] is None  # sem provider ligado
        assert status["installed_count"] == 1
        assert status["catalog_total"] == 4
        assert json.dumps(status)  # serializável

        manager.attach_provider(RecordingProvider(available=False))
        assert manager.get_status()["provider_running"] is False


# --------------------------------------------------------------------- #
# Catálogo inválido: degradar sem derrubar o DaviOS
# --------------------------------------------------------------------- #

class TestInvalidCatalog:
    def test_missing_catalog_file(self, workspace):
        (workspace / "config" / "model_catalog.json").unlink()
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.list_models() == []
        assert manager.get_active_model() is None
        assert manager.active_state == ACTIVE_STATE_NONE
        assert manager.check_compatibility(SMALL).level == LEVEL_UNKNOWN

    def test_corrupt_catalog_file(self, workspace):
        (workspace / "config" / "model_catalog.json").write_text(
            "{{{ nao e json", encoding="utf-8"
        )
        manager = _make_manager(workspace, install=(SMALL,))
        assert manager.get_model(SMALL) is None
        assert manager.set_active_model(SMALL) is None
        assert "não reconhecido" in manager.last_error