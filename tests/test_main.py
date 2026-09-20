"""Testes isolados do boot construído por :func:`main.build_engine`."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import main
from brain.model_manager import ModelManager as RealModelManager
from brain.model_manager import ModelSelection
from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareProfile


class _EmptyCatalog:
    """Catálogo mínimo para exercitar o ModelManager real sem o disco do projeto."""

    def __len__(self) -> int:
        return 0

    def get_model_by_id(self, _model_id: str) -> None:
        return None

    def resolve_alias(self, _request: str) -> None:
        return None

    def all_models(self) -> list[Any]:
        return []

    def get_installed_models(self) -> list[Any]:
        return []


def _install_boot_fakes(monkeypatch, tmp_path, *, standalone_succeeds: bool) -> dict:
    """Substitui somente as bordas externas de ``build_engine``.

    O ``TrackingModelManager`` continua sendo um ``ModelManager`` real para
    que ``get_status()`` valide o vínculo com o provider de verdade.
    """

    events: list[tuple[str, object]] = []
    managers: list[RealModelManager] = []
    standalone_instances: list[object] = []
    fallback_instances: list[object] = []
    config_path = tmp_path / "davios.json"
    config_path.write_text("{}", encoding="utf-8")
    hardware = HardwareProfile(
        os="Teste",
        cpu_name="CPU Fake",
        cpu_cores=8,
        ram_total_gb=16.0,
        ram_available_gb=12.0,
        gpu_name="GPU Fake",
        gpu_vram_gb=None,
        gpu_vendor="Fake",
        disk_free_gb=100.0,
        inference_backend="llama_cpp",
    )
    selection = ModelSelection(profile="TEST", reason="Seleção de teste")

    class TrackingModelManager(RealModelManager):
        def __init__(self, config: DaviosConfig) -> None:
            super().__init__(
                config=config,
                catalog=_EmptyCatalog(),
                config_path=str(config_path),
            )
            self._hardware = hardware
            managers.append(self)
            events.append(("manager_created", self))

        def select_model(self, detected_hardware: HardwareProfile) -> ModelSelection:
            self._hardware = detected_hardware
            events.append(("select_model", self))
            return selection

        def attach_provider(self, provider: object) -> None:
            events.append(("attach_provider", provider))
            super().attach_provider(provider)

    class FakeHardwareDetector:
        def detect(self) -> HardwareProfile:
            events.append(("detect_hardware", hardware))
            return hardware

    class FakeStandaloneProvider:
        def __init__(self, *, config, model_manager, selection) -> None:
            self.config = config
            self.model_manager = model_manager
            self.selection = selection
            self.initialize_calls = 0
            self.is_available_calls = 0
            self.attached_when_initialized = False
            self.diagnostics = {} if standalone_succeeds else {"backend": "indisponível"}
            standalone_instances.append(self)
            events.append(("standalone_created", self))

        def initialize(self) -> bool:
            self.initialize_calls += 1
            self.attached_when_initialized = self.model_manager.provider is self
            events.append(("standalone_initialize", self))
            return standalone_succeeds

        def is_available(self) -> bool:
            self.is_available_calls += 1
            return True

        def health_check(self) -> dict[str, str]:
            return {"backend": "fake-standalone"}

    class FakeFallbackProvider:
        # Deliberadamente não implementa switch_model().
        def __init__(self, *, config, model_manager, selection) -> None:
            self.config = config
            self.model_manager = model_manager
            self.selection = selection
            self.initialize_calls = 0
            self.is_available_calls = 0
            self.attached_when_initialized = False
            fallback_instances.append(self)
            events.append(("fallback_created", self))

        def initialize(self) -> bool:
            self.initialize_calls += 1
            self.attached_when_initialized = self.model_manager.provider is self
            events.append(("fallback_initialize", self))
            return True

        def is_available(self) -> bool:
            self.is_available_calls += 1
            return True

        def health_check(self) -> dict[str, str]:
            return {"backend": "fake-fallback"}

    class FakeActionManager:
        def __init__(self, *, enabled: bool) -> None:
            self.enabled = enabled

    class FakeWebAccess:
        def __init__(self, *, enabled: bool) -> None:
            self.enabled = enabled

    class FakeToolRegistry:
        pass

    class FakeToolRouter:
        def __init__(self, registry: FakeToolRegistry) -> None:
            self.registry = registry

    def fake_register_default_tools(*_args, **_kwargs) -> None:
        events.append(("register_tools", object()))

    class FakeCognitiveCore:
        instances: list["FakeCognitiveCore"] = []

        def __init__(self, *, llm_provider, config, tool_router) -> None:
            self.llm_provider = llm_provider
            self.config = config
            self.tool_router = tool_router
            self.__class__.instances.append(self)
            events.append(("cognitive_core_created", self))

    class FakeConversationEngine:
        instances: list["FakeConversationEngine"] = []

        def __init__(self, *, cognitive_core, model_manager=None) -> None:
            self.cognitive_core = cognitive_core
            # Etapa 6: o boot tambem injeta o ModelManager no engine, para que
            # pedidos de troca de modelo cheguem a switch_active_model().
            self.model_manager = model_manager
            self.__class__.instances.append(self)
            events.append(("engine_created", self))

    monkeypatch.setattr(main, "HardwareDetector", FakeHardwareDetector)
    monkeypatch.setattr(main, "ModelManager", TrackingModelManager)
    monkeypatch.setattr(main, "LocalLlamaCppProvider", FakeStandaloneProvider)
    monkeypatch.setattr(main, "LocalLLMProvider", FakeFallbackProvider)
    monkeypatch.setattr(main, "ConversationEngine", FakeConversationEngine)

    # build_engine importa estas dependências dentro da função. Módulos falsos
    # evitam importar o restante do runtime (incluindo personality) e deixam
    # o teste estritamente no contrato de composição do boot.
    fake_modules = {
        "brain.action_manager": ("ActionManager", FakeActionManager),
        "brain.cognitive_core": ("CognitiveCore", FakeCognitiveCore),
        "brain.tool_adapters": (
            "register_default_tools",
            fake_register_default_tools,
        ),
        "brain.tool_registry": ("ToolRegistry", FakeToolRegistry),
        "brain.web_access": ("WebAccess", FakeWebAccess),
    }
    tool_registry_module = ModuleType("brain.tool_registry")
    tool_registry_module.ToolRegistry = FakeToolRegistry
    tool_registry_module.ToolRouter = FakeToolRouter
    monkeypatch.setitem(sys.modules, "brain.tool_registry", tool_registry_module)
    for module_name, (attribute, value) in fake_modules.items():
        if module_name == "brain.tool_registry":
            continue
        module = ModuleType(module_name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, module_name, module)

    return {
        "events": events,
        "managers": managers,
        "selection": selection,
        "standalone_instances": standalone_instances,
        "fallback_instances": fallback_instances,
        "cognitive_instances": FakeCognitiveCore.instances,
        "engine_instances": FakeConversationEngine.instances,
    }


def _event_index(events: list[tuple[str, object]], name: str, value: object) -> int:
    return events.index((name, value))


def test_build_engine_attaches_standalone_to_the_runtime_manager(monkeypatch, tmp_path):
    fakes = _install_boot_fakes(monkeypatch, tmp_path, standalone_succeeds=True)

    engine, _hardware, manager, selection, provider = main.build_engine(DaviosConfig())

    assert len(fakes["managers"]) == 1
    assert manager is fakes["managers"][0]
    assert selection is fakes["selection"]
    assert provider is fakes["standalone_instances"][0]
    assert manager.provider is provider
    assert provider.model_manager is manager
    assert provider.selection is selection
    assert engine.cognitive_core.llm_provider is provider
    assert engine.model_manager is manager
    assert len(fakes["cognitive_instances"]) == 1
    assert len(fakes["engine_instances"]) == 1
    assert provider.initialize_calls == 1
    assert provider.attached_when_initialized is True
    assert _event_index(fakes["events"], "attach_provider", provider) < _event_index(
        fakes["events"], "standalone_initialize", provider
    )

    status = manager.get_status()
    assert status["provider_running"] is True
    assert provider.is_available_calls == 1


def test_build_engine_attaches_fallback_without_switch_model(monkeypatch, tmp_path):
    fakes = _install_boot_fakes(monkeypatch, tmp_path, standalone_succeeds=False)

    engine, _hardware, manager, selection, provider = main.build_engine(DaviosConfig())

    standalone = fakes["standalone_instances"][0]
    assert standalone.initialize_calls == 1
    assert len(fakes["managers"]) == 1
    assert manager is fakes["managers"][0]
    assert selection is fakes["selection"]
    assert provider is fakes["fallback_instances"][0]
    assert not hasattr(provider, "switch_model")
    assert manager.provider is provider
    assert provider.model_manager is manager
    assert provider.selection is selection
    assert provider.initialize_calls == 1
    assert provider.attached_when_initialized is True
    assert engine.cognitive_core.llm_provider is provider
    assert engine.model_manager is manager
    assert _event_index(fakes["events"], "standalone_initialize", standalone) < _event_index(
        fakes["events"], "attach_provider", provider
    ) < _event_index(fakes["events"], "fallback_initialize", provider)

    status = manager.get_status()
    assert status["provider_running"] is True
    assert provider.is_available_calls == 2
