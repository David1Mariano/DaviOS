"""ModelManager do DaviOS.

Descobre modelos locais (GGUF), lê metadados, verifica compatibilidade com
o hardware e seleciona o modelo adequado ao perfil. NÃO executa inferência.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareProfile

logger = logging.getLogger("davios.models")

KNOWN_QUANTS = (
    "IQ1_S", "IQ2_XXS", "IQ3_XXS", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S",
    "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32",
)

# Bytes aproximados por parametro segun cuantizacion (llama.cpp). Se usam
# para estimar el tamano esperado de un GGUF real y detectar archivos
# truncados/corrompidos (ej: um "1.5B Q4_K_M" real ocupa ~0.92 GB, no 11 MB).
BYTES_PER_PARAM: dict[str, float] = {
    "IQ1_S": 0.19, "IQ2_XXS": 0.21, "IQ3_XXS": 0.32, "IQ3_XS": 0.35,
    "Q2_K": 0.332,
    "Q3_K_S": 0.429, "Q3_K_M": 0.457, "Q3_K_L": 0.479,
    "Q4_0": 0.568, "Q4_1": 0.728,
    "Q4_K_S": 0.561, "Q4_K_M": 0.612,
    "Q5_0": 0.675, "Q5_1": 0.836,
    "Q5_K_S": 0.626, "Q5_K_M": 0.663,
    "Q6_K": 0.714,
    "Q8_0": 1.102,
    "BF16": 2.0, "F16": 2.0, "F32": 4.0,
}
DEFAULT_BYTES_PER_PARAM = 0.61
# Margen para aceptar um GGUF real: al menos el 60% del tamano estimado.
# Separa archivos truncados/incompletos de modelos legitimos. Nunca se
# borra el archivo — solo se deja de seleccionarlo.
PLAUSIBLE_SIZE_RATIO = 0.6

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
    # Razón de exclusión durante la selección (archivo corrompido/incompleto),
    # rellenada por _file_is_plausible(). "" = sin problema detectado.
    invalid_reason: str = ""

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

    @staticmethod
    def _expected_size_mb(model: ModelInfo) -> Optional[float]:
        """Tamaño aproximado esperado de un GGUF real (MB).

        Se estima desde parámetros declarados en el nombre (ej "1.5B") y la
        cuantización (bytes/parámetro conocido de llama.cpp). None si no hay
        parámetros en el nombre para poder estimar.
        """
        if not model.params_hint:
            return None
        try:
            params_b = float(model.params_hint[:-1])
        except (ValueError, AttributeError):
            return None
        bytes_per = BYTES_PER_PARAM.get(
            model.quantization or "", DEFAULT_BYTES_PER_PARAM
        )
        # params_b * bytes/param * 1e9 bytes per GB, expressado em MB (1e6)
        return params_b * bytes_per * 1000.0

    def _file_is_plausible(self, model: ModelInfo) -> bool:
        """Descarta modelos cuyo archivo está truncado/corrompido.

        Criterio: si el nombre declara parámetros y cuantización, el archivo
        debe tener al menos 60% del tamaño esperado para ser un GGUF real.
        Sin parámetros declarados no hay cómo estimar → se acepta. El archivo
        NUNCA se borra: solo se excluye de la selección y se loguea el aviso.
        """
        expected_mb = self._expected_size_mb(model)
        if expected_mb is None:
            return True
        try:
            actual_mb = model.path.stat().st_size / (1024**2)
        except OSError:
            model.invalid_reason = "archivo inaccesible o inexistente"
            return False
        min_mb = expected_mb * PLAUSIBLE_SIZE_RATIO
        if actual_mb < min_mb:
            model.invalid_reason = (
                "archivo parece corrompido/incompleto "
                f"({actual_mb:.0f} MB, esperado ~{expected_mb:.0f} MB)"
            )
            logger.warning("Modelo %s ignorado: %s", model.name, model.invalid_reason)
            return False
        return True

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
            invalid = [
                m.invalid_reason for m in candidates if m.invalid_reason
            ]
            if invalid:
                reason = (
                    "Modelos encontrados parecen invalidos: "
                    + "; ".join(sorted(set(invalid)))
                )
            else:
                reason = (
                    "Modelos encontrados excedem os recursos do perfil "
                    f"{profile_name}."
                )
            return ModelSelection(
                profile=profile_name,
                reason=reason,
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
        if not self._file_is_plausible(model):
            return False
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

