"""ModelManager do DaviOS.

Descobre modelos locais (GGUF), lê metadados, verifica compatibilidade com
o hardware e seleciona o modelo adequado ao perfil. NÃO executa inferência.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareProfile

KNOWN_QUANTS = (
    "IQ1_S", "IQ2_XXS", "IQ3_XXS", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S",
    "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32",
)

TIERS = ("light", "balanced", "performance")


@dataclass
class ModelInfo:
    """Metadados de um modelo local descoberto."""

    path: Path
    name: str
    tier: str = "balanced"
    size_gb: float = 0.0
    quantization: Optional[str] = None
    params_hint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "size_gb": self.size_gb,
            "quantization": self.quantization,
            "params_hint": self.params_hint,
            "path": str(self.path),
        }


@dataclass
class ModelSelection:
    """Resultado da seleção: modelo escolhido + perfil de execução."""

    model: Optional[ModelInfo] = None
    profile: str = "BALANCED"
    reason: str = ""
    compatible: bool = False
    generation: dict[str, Any] = field(default_factory=dict)


class ModelManager:
    """Descobre e seleciona modelos locais; não gera texto."""

    def __init__(self, config: Optional[DaviosConfig] = None):
        self.config = config or DaviosConfig.load()
        self._profiles: dict[str, Any] = {}
        self._load_profiles()

    def _load_profiles(self) -> None:
        path = self.config.profiles_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._profiles = data.get("profiles", {})
        except (OSError, json.JSONDecodeError):
            self._profiles = {}

    def _raw_profiles(self) -> dict[str, Any]:
        try:
            path = self.config.profiles_path()
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def discover_models(self) -> list[ModelInfo]:
        """Varre models/{tier}/*.gguf e extrai metadados."""
        models: list[ModelInfo] = []
        root = self.config.models_path()
        if not root.exists():
            return models
        for tier in TIERS:
            tier_dir = root / tier
            if not tier_dir.is_dir():
                continue
            for file in sorted(tier_dir.glob("*.gguf")):
                models.append(self._info_from_file(file, tier))
        return models

    @staticmethod
    def _info_from_file(path: Path, tier: str) -> ModelInfo:
        name = path.stem
        size_gb = round(path.stat().st_size / (1024**3), 2)
        quant = None
        for q in KNOWN_QUANTS:
            if re.search(rf"{re.escape(q)}\b", name, flags=re.IGNORECASE):
                quant = q
                break
        params = None
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", name)
        if match:
            params = f"{match.group(1)}B"
        return ModelInfo(
            path=path, name=name, tier=tier, size_gb=size_gb,
            quantization=quant, params_hint=params,
        )

    def select_profile(self, hardware: HardwareProfile) -> str:
        """Escolhe LIGHT/BALANCED/PERFORMANCE conforme hardware + config."""
        override = self.config.profile_override
        if override and override in self._profiles:
            return override
        selection_order = (
            self._raw_profiles().get("selection_order")
            or ["PERFORMANCE", "BALANCED", "LIGHT"]
        )
        for profile_name in selection_order:
            if self._profile_matches(profile_name, hardware):
                return profile_name
        return "LIGHT"

    def _profile_matches(self, profile_name: str, hardware: HardwareProfile) -> bool:
        profile = self._profiles.get(profile_name) or {}
        criteria = profile.get("criteria", {})
        ram = hardware.ram_total_gb
        vram = hardware.gpu_vram_gb
        cores = hardware.cpu_cores

        min_ram = criteria.get("min_ram_total_gb")
        if min_ram is not None and (ram is None or ram < min_ram):
            return False
        max_ram = criteria.get("max_ram_total_gb")
        if max_ram is not None and (ram is not None and ram > max_ram):
            return False
        max_cores = criteria.get("max_cpu_cores")
        if max_cores is not None and (cores is not None and cores > max_cores):
            return False
        min_vram = criteria.get("min_vram_gb")
        if min_vram is not None and (vram is None or vram < min_vram):
            return False
        return True

    def select_model(self, hardware: HardwareProfile) -> ModelSelection:
        """Seleciona o melhor modelo instalado para o hardware atual."""
        profile_name = self.select_profile(hardware)
        profile = self._profiles.get(profile_name) or {}
        tier = profile.get("model_tier", "balanced")
        max_size = profile.get("max_model_size_gb")

        models = self.discover_models()
        if not models:
            return ModelSelection(
                profile=profile_name,
                reason="Nenhum modelo .gguf encontrado em models/.",
                generation=profile.get("generation", {}),
            )

        tier_models = [m for m in models if m.tier == tier]
        candidates = tier_models or models  # fallback para qualquer tier

        usable = [m for m in candidates if self._is_compatible(m, max_size, hardware)]
        if not usable:
            return ModelSelection(
                profile=profile_name,
                reason=(
                    "Modelos encontrados excedem os recursos do perfil "
                    f"{profile_name}."
                ),
                generation=profile.get("generation", {}),
            )

        best = max(usable, key=lambda m: m.size_gb)
        return ModelSelection(
            model=best,
            profile=profile_name,
            compatible=True,
            reason=(
                f"Modelo {best.name} ({best.size_gb} GB) selecionado para o "
                f"perfil {profile_name}."
            ),
            generation=profile.get("generation", {}),
        )

    def _is_compatible(
        self,
        model: ModelInfo,
        max_size_gb: Optional[float],
        hardware: HardwareProfile,
    ) -> bool:
        if max_size_gb is not None and model.size_gb > max_size_gb:
            return False
        available = hardware.ram_available_gb
        # GGUF e mapeado em memoria; exigimos ~1.4x o tamanho em RAM livre
        if available is not None and model.size_gb * 1.4 > available:
            return False
        return True

    def status_summary(self, selection: ModelSelection) -> str:
        """Linha de status para o boot."""
        if selection.model is None:
            return "Modelo local nao encontrado."
        model = selection.model
        quant = model.quantization or "quantizacao desconhecida"
        return f"{model.name} ({quant}, {model.size_gb} GB)"

