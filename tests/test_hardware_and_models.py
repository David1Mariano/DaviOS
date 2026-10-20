"""Testes do HardwareDetector e do ModelManager."""

import os

import pytest

from config.davios_config import DaviosConfig
from core.hardware_detector import (
    UNKNOWN,
    HardwareDetector,
    HardwareProfile,
    detect_inference_backend,
)
from brain.model_manager import ModelInfo, ModelManager


@pytest.fixture
def detector_no_probe():
    return HardwareDetector(enable_gpu_probe=False)


class TestHardwareDetector:
    def test_detect_returns_profile(self, detector_no_probe):
        assert isinstance(detector_no_probe.detect(), HardwareProfile)

    def test_os_detected(self, detector_no_probe):
        assert detector_no_probe.detect_os() != UNKNOWN

    def test_cpu_cores_detected(self, detector_no_probe):
        assert detector_no_probe.detect_cpu_cores() >= 1

    def test_ram_returns_pair(self, detector_no_probe):
        total, available = detector_no_probe.detect_ram()
        if total is not None:
            assert total > 0
            if available is not None:
                assert available <= total

    def test_gpu_probe_disabled_returns_unknown(self):
        name, vram, vendor = HardwareDetector(enable_gpu_probe=False).detect_gpu()
        assert (name, vram, vendor) == (UNKNOWN, None, UNKNOWN)

    def test_gpu_probe_enabled_never_raises(self):
        name, vram, vendor = HardwareDetector(enable_gpu_probe=True).detect_gpu()
        assert isinstance(name, str)

    def test_disk_free_positive(self, detector_no_probe):
        free = detector_no_probe.detect_disk_free()
        assert free is None or free >= 0

    def test_profile_to_dict(self, detector_no_probe):
        data = detector_no_probe.detect().to_dict()
        assert "os" in data and "cpu_cores" in data

    def test_summary_lines(self):
        profile = HardwareProfile(
            os="Windows 11", cpu_name="AMD Ryzen 5 5600", cpu_cores=12,
            ram_total_gb=15.9, ram_available_gb=2.1,
            gpu_name="AMD Radeon RX 6600", gpu_vram_gb=None, gpu_vendor="AMD",
        )
        lines = profile.summary_lines()
        assert any("Ryzen" in line for line in lines)
        assert any("RX 6600" in line for line in lines)

    def test_inference_backend_is_string(self):
        assert isinstance(detect_inference_backend(), str)


def _write_gguf(directory, name, size_mb=10):
    """Cria um .gguf FALSO mas com cabecalho GGUF valido.

    A validacao de cabecalho (4 bytes "GGUF") e feita pelo ModelManager e
    pelo ModelCatalog: um arquivo sem a assinatura seria rejeitado como
    invalido, e o teste mediria outra coisa que nao a pretendida.
    """
    path = os.path.join(directory, name)
    with open(path, "wb") as f:
        f.write(b"GGUF")
        # Escreve em chunks de 1 MB para evitar MemoryError
        chunk = b"\0" * (1024 * 1024)
        for _ in range(size_mb):
            f.write(chunk)
    return path


@pytest.fixture
def models_tree(tmp_path):
    for tier in ("light", "balanced", "performance"):
        (tmp_path / tier).mkdir()
    return tmp_path


@pytest.fixture
def config_with_models(models_tree):
    config = DaviosConfig.load()
    config.models_dir = str(models_tree)
    # Isola os testes do diretório externo configurado em davios.json local.
    config.external_models_dir = None
    config.profile_override = None
    return config


class TestModelManager:
    def test_discover_empty(self, config_with_models):
        assert ModelManager(config_with_models).discover_models() == []

    def test_discover_models(self, config_with_models, models_tree):
        _write_gguf(models_tree / "balanced", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
        _write_gguf(models_tree / "light", "tiny-llama-Q8_0.gguf")
        models = ModelManager(config_with_models).discover_models()
        assert len(models) == 2
        by_tier = {m.tier: m for m in models}
        assert by_tier["balanced"].quantization == "Q4_K_M"
        assert by_tier["balanced"].params_hint == "1.5B"
        assert by_tier["light"].quantization == "Q8_0"

    def test_profile_light_for_weak_machine(self, config_with_models):
        weak = HardwareProfile(ram_total_gb=4.0, ram_available_gb=1.0, cpu_cores=2)
        assert ModelManager(config_with_models).select_profile(weak) == "LIGHT"

    def test_profile_balanced_for_mid_machine(self, config_with_models):
        mid = HardwareProfile(ram_total_gb=12.0, ram_available_gb=6.0, cpu_cores=8)
        assert ModelManager(config_with_models).select_profile(mid) == "BALANCED"

    def test_profile_performance_for_strong_machine(self, config_with_models):
        strong = HardwareProfile(
            ram_total_gb=32.0, ram_available_gb=20.0, cpu_cores=12,
            gpu_name="AMD Radeon RX 6600", gpu_vram_gb=8.0, gpu_vendor="AMD",
        )
        assert ModelManager(config_with_models).select_profile(strong) == "PERFORMANCE"

    def test_profile_override(self, config_with_models):
        config_with_models.profile_override = "LIGHT"
        strong = HardwareProfile(ram_total_gb=32.0, ram_available_gb=20.0, cpu_cores=16)
        assert ModelManager(config_with_models).select_profile(strong) == "LIGHT"

    def test_select_model_without_models(self, config_with_models):
        hardware = HardwareProfile(ram_total_gb=16.0, ram_available_gb=8.0, cpu_cores=8)
        selection = ModelManager(config_with_models).select_model(hardware)
        assert selection.model is None
        assert "Nenhum modelo" in selection.reason

    def test_corrupt_small_model_rejected(self, config_with_models, models_tree):
        """Um GGUF com parametros declarados (1.5B Q4_K_M) mas tamanho
        ridiculamente pequeno (~10 MB vs ~918 MB esperados) e excluido dos
        candidatos, com razao clara na selecao."""
        _write_gguf(
            models_tree / "balanced", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        )
        hardware = HardwareProfile(
            ram_total_gb=16.0, ram_available_gb=8.0, cpu_cores=8
        )
        selection = ModelManager(config_with_models).select_model(hardware)
        assert selection.model is None
        assert "corrompido/incompleto" in selection.reason

    def test_plausible_size_model_accepted(self, config_with_models, models_tree):
        """Um GGUF do tamanho esperado (>=60% de ~918 MB para 1.5B Q4_K_M)
        e selecionado normalmente."""
        _write_gguf(
            models_tree / "balanced", "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            size_mb=600,
        )
        hardware = HardwareProfile(
            ram_total_gb=16.0, ram_available_gb=8.0, cpu_cores=8
        )
        selection = ModelManager(config_with_models).select_model(hardware)
        assert selection.model is not None
        assert selection.model.quantization == "Q4_K_M"
        assert selection.compatible is True

    def test_discover_models_includes_external_dir(self, tmp_path):
        """external_models_dir é escaneado recursivamente; tier inferido do
        subdiretório pai. Não afeta a varredura local padrão."""
        from config.davios_config import DaviosConfig

        external = tmp_path / "external_models"
        (external / "balanced").mkdir(parents=True)
        (external / "light").mkdir(parents=True)
        _write_gguf(external / "balanced", "Qwen3-4B-Q4_K_M.gguf", size_mb=10)
        _write_gguf(external / "light", "tiny-Q8_0.gguf", size_mb=5)

        config = DaviosConfig()
        config.models_dir = str(tmp_path / "local_models")  # vazio
        config.external_models_dir = str(external)

        models = ModelManager(config).discover_models()
        names = {m.name for m in models}
        assert "Qwen3-4B-Q4_K_M" in names
        assert "tiny-Q8_0" in names
        by_name = {m.name: m for m in models}
        assert by_name["Qwen3-4B-Q4_K_M"].tier == "balanced"
        assert by_name["tiny-Q8_0"].tier == "light"


class TestModelSelection:
    def test_select_model_prefers_tier(self, config_with_models, models_tree):
        _write_gguf(models_tree / "light", "tiny-Q8_0.gguf", size_mb=5)
        _write_gguf(models_tree / "balanced", "mid-Q4_K_M.gguf", size_mb=30)
        hardware = HardwareProfile(ram_total_gb=16.0, ram_available_gb=8.0, cpu_cores=8)
        selection = ModelManager(config_with_models).select_model(hardware)
        assert selection.model is not None
        assert selection.model.tier == "balanced"
        assert selection.compatible is True

    def test_select_model_falls_back_to_other_tier(self, config_with_models, models_tree):
        _write_gguf(models_tree / "light", "tiny-Q8_0.gguf", size_mb=5)
        hardware = HardwareProfile(ram_total_gb=16.0, ram_available_gb=8.0, cpu_cores=8)
        selection = ModelManager(config_with_models).select_model(hardware)
        assert selection.model is not None
        assert selection.model.tier == "light"

    def test_incompatible_model_rejected(self, config_with_models, models_tree):
        # Modelo de 800 MB e incompativel com 1 GB RAM disponivel (precisa ~1.1 GB)
        _write_gguf(models_tree / "balanced", "huge-Q4_K_M.gguf", size_mb=800)
        hardware = HardwareProfile(ram_total_gb=8.0, ram_available_gb=1.0, cpu_cores=4)
        selection = ModelManager(config_with_models).select_model(hardware)
        assert selection.model is None

    def test_ram_blocked_reason_is_explicit(self, config_with_models, models_tree):
        """Mesmo cenario de RAM insuficiente, mas a razao agora nomeia o
        modelo e a conta de GB (gap anterior: so 'excedem os recursos')."""
        _write_gguf(models_tree / "balanced", "huge-Q4_K_M.gguf", size_mb=800)
        hardware = HardwareProfile(ram_total_gb=8.0, ram_available_gb=1.0, cpu_cores=4)
        selection = ModelManager(config_with_models).select_model(hardware)
        assert selection.model is None
        assert "huge-Q4_K_M" in selection.reason
        assert "livres de RAM" in selection.reason
        assert "disponivel: 1.00 GB" in selection.reason

    def test_incompatibility_reason_reports_size_cap(self, config_with_models, models_tree):
        """_incompatibility_reason explica o bloqueio por teto de tamanho do
        perfil e devolve vazio quando o modelo e elegivel."""
        _write_gguf(models_tree / "balanced", "mid-Q4_K_M.gguf", size_mb=30)
        manager = ModelManager(config_with_models)
        model = manager.discover_models()[0]
        hardware = HardwareProfile(ram_total_gb=16.0, ram_available_gb=8.0, cpu_cores=8)

        why = manager._incompatibility_reason(model, max_size_gb=0.01, hardware=hardware)
        assert "excede o limite" in why
        assert "mid-Q4_K_M" in why
        # Sem teto e com RAM de sobra: elegivel.
        assert manager._incompatibility_reason(model, None, hardware) == ""

    def test_status_summary_without_model(self, config_with_models):
        hardware = HardwareProfile(ram_total_gb=16.0, cpu_cores=8)
        manager = ModelManager(config_with_models)
        selection = manager.select_model(hardware)
        assert "nao encontrado" in manager.status_summary(selection)

    def test_status_summary_with_model(self, config_with_models, models_tree):
        _write_gguf(models_tree / "balanced", "mid-Q4_K_M.gguf", size_mb=30)
        hardware = HardwareProfile(ram_total_gb=16.0, ram_available_gb=8.0, cpu_cores=8)
        manager = ModelManager(config_with_models)
        summary = manager.status_summary(manager.select_model(hardware))
        assert "mid-Q4_K_M" in summary

