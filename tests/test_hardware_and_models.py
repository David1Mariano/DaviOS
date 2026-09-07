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
    path = os.path.join(directory, name)
    with open(path, "wb") as f:
        f.write(b"\0" * (size_mb * 1024 * 1024))
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
        _write_gguf(models_tree / "balanced", "huge-Q4_K_M.gguf", size_mb=4600)
        hardware = HardwareProfile(ram_total_gb=8.0, ram_available_gb=1.0, cpu_cores=4)
        selection = ModelManager(config_with_models).select_model(hardware)
        assert selection.model is None

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

