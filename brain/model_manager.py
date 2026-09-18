"""ModelManager do DaviOS.

Camada de DECISÃO e COORDENAÇÃO sobre modelos:

- consulta o ModelCatalog (dados dos modelos conhecidos);
- descobre modelos .gguf presentes no disco (models/<tier>/*.gguf);
- verifica compatibilidade com o hardware;
- resolve aliases em linguagem natural ("leve", "forte", "qwen3 8b");
- escolhe um modelo inicial determinístico quando nada está configurado;
- mantém e persiste qual é o modelo ativo (active_model_id no davios.json).

NÃO executa inferência, NÃO inicia processos e NÃO fala HTTP. Quem faz isso é
o LocalLlamaCppProvider. O ConversationEngine nunca chama o provider direto:
ele pede ao ModelManager, que por sua vez delega ao provider.

Estado desta etapa: NÃO existe download nem troca real de llama-server. O
"modelo ativo" é uma INTENÇÃO registrada (active_model_id), não uma garantia de
que o processo de inferência já está rodando com ele. A troca transacional
(parar A -> iniciar B -> health check -> rollback) será coordenada aqui e
executada pelo provider numa etapa posterior; `build_selection()` já entrega a
estrutura que o provider consome, para que essa etapa não precise tocar no
ConversationEngine.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.davios_config import CONFIG_DIR, DaviosConfig
from core.hardware_detector import HardwareDetector, HardwareProfile
from brain.model_catalog import (
    LEGACY_TIER_MAP,
    TIERS as CATALOG_TIERS,
    TIER_LABELS,
    ModelCatalog,
    ModelInfo,
    get_tier_label,
    normalize_tier,
    tier_from_legacy,
)

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

# Diretórios varridos na descoberta local. "performance" é o diretório legado
# (ver LEGACY_TIER_MAP em model_catalog): PERFORMANCE -> strong. Mantemos os
# dois nomes para não quebrar instalações antigas.
TIERS = ("light", "balanced", "performance", "strong", "very_strong")


# =============================================================================
# NÍVEIS DE COMPATIBILIDADE
# =============================================================================
# Nunca bloqueamos um modelo só por estimativa de VRAM: quantização, backend,
# contexto, -ngl e KV cache mudam muito o consumo real. Por isso o resultado
# distingue "recomendado" de "pode funcionar com limitações".
LEVEL_RECOMMENDED = "recommended"
LEVEL_POSSIBLE = "possible"
LEVEL_RISKY = "risky"
LEVEL_INCOMPATIBLE = "incompatible"

# Estado usado quando os metadados do modelo não permitem estimar nada
# (nem tamanho, nem requisitos declarados). É diferente de "risky": aqui não
# sabemos, então NÃO afirmamos compatibilidade nem incompatibilidade.
LEVEL_UNKNOWN = "unknown"

# Ordem de preferência na escolha determinística do modelo inicial.
LEVEL_RANK = {
    LEVEL_RECOMMENDED: 0,
    LEVEL_POSSIBLE: 1,
    LEVEL_RISKY: 2,
    LEVEL_UNKNOWN: 3,
}

# Estados possíveis do modelo ativo (ver ModelManager.active_state).
ACTIVE_STATE_UNSET = "unset"
ACTIVE_STATE_PERSISTED = "persisted"
ACTIVE_STATE_AUTO = "auto"
ACTIVE_STATE_MISSING = "missing"
ACTIVE_STATE_NOT_INSTALLED = "not_installed"
ACTIVE_STATE_NONE = "none"

# Margem de segurança de disco: além do arquivo, o downloader escreve o
# temporário (.part) no mesmo volume. 15% cobre isso e picos de fragmentação.
DISK_SAFETY_MARGIN = 1.15


@dataclass
class LocalModelInfo:
    """Metadados de um arquivo .gguf descoberto varrendo models/.

    Este é o resultado da DESCOBERTA em disco, não a representação canônica de
    um modelo catalogado — essa é `ModelInfo`, importada de brain.model_catalog.
    A distinção existe porque a descoberta enxerga apenas o que o nome do
    arquivo revela, enquanto o catálogo tem metadados declarados.
    """

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
    """Resultado da seleção: modelo escolhido + perfil de execução.

    `model` é um LocalModelInfo (descoberto em disco) para preservar a API
    existente usada pelo provider e pelos testes.
    """

    model: Optional[LocalModelInfo] = None
    profile: str = "BALANCED"
    reason: str = ""
    compatible: bool = False
    generation: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompatibilityResult:
    """Resultado estruturado de check_compatibility().

    `level` resume a situação; `warnings`/`errors` explicam em linguagem
    humana. Nem toda limitação vira `compatible=False`: só erros de fato
    (RAM total insuficiente, disco insuficiente) bloqueiam.
    """

    compatible: Optional[bool] = True
    status: str = "compatible"
    installed: bool = False
    level: str = LEVEL_RECOMMENDED
    ram_ok: bool = True
    vram_ok: bool = True
    disk_ok: bool = True
    backend_ok: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # Estimativas usadas, para poder explicar a decisão ao usuário.
    required_disk_gb: float = 0.0
    free_disk_gb: Optional[float] = None
    ram_available_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    # Identidade do modelo avaliado (útil quando a avaliação vem de um id/alias).
    model_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "compatible": self.compatible,
            "status": self.status,
            "installed": self.installed,
            "reasons": list(self.errors),
            "level": self.level,
            "ram_ok": self.ram_ok,
            "vram_ok": self.vram_ok,
            "disk_ok": self.disk_ok,
            "backend_ok": self.backend_ok,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "required_disk_gb": self.required_disk_gb,
            "free_disk_gb": self.free_disk_gb,
            "ram_available_gb": self.ram_available_gb,
            "ram_total_gb": self.ram_total_gb,
        }

    @property
    def label(self) -> str:
        return {
            LEVEL_RECOMMENDED: "RECOMENDADO",
            LEVEL_POSSIBLE: "POSSÍVEL",
            LEVEL_RISKY: "ARRISCADO",
            LEVEL_INCOMPATIBLE: "INCOMPATÍVEL",
            LEVEL_UNKNOWN: "DESCONHECIDO",
        }.get(self.level, self.level.upper())


class ModelManager:
    """Decide e coordena o uso de modelos; não gera texto e não inicia processos."""

    def __init__(
        self,
        config: Optional[DaviosConfig] = None,
        catalog: Optional[ModelCatalog] = None,
        config_path: Optional[str] = None,
    ):
        self.config = config or DaviosConfig.load()
        # `is not None` em vez de `or`: ModelCatalog define __len__, então um
        # catálogo VAZIO é falsy — com `or` ele seria descartado em silêncio e
        # substituído pelo catálogo do projeto.
        self.catalog = catalog if catalog is not None else ModelCatalog(config=self.config)
        self._profiles: dict[str, Any] = {}
        self._load_profiles()
        # Provider de inferência (LocalLlamaCppProvider), ligado depois do boot
        # por `attach_provider()`. Fica None em contexto de linha de comando,
        # nos testes e quando só se quer consultar/instalar modelos.
        self._provider: Optional[Any] = None
        # Arquivo onde active_model_id é persistido. É o MESMO davios.json que o
        # DaviosConfig lê: reutilizamos a persistência existente em vez de criar
        # outro arquivo/banco. Injetável para permitir isolamento em testes.
        self._config_path = (
            Path(config_path) if config_path else (CONFIG_DIR / "davios.json")
        )
        # Cache do hardware e do último erro (exposto em get_status()).
        self._hardware: Optional[HardwareProfile] = None
        self._last_error: str = ""
        self._active_state: str = ACTIVE_STATE_UNSET

    # ------------------------------------------------------------------ #
    # Ligação com o provider de inferência
    # ------------------------------------------------------------------ #

    def attach_provider(self, provider: Any) -> None:
        """Registra o provider que executa o llama-server.

        O ModelManager é o único que conversa com o provider: o
        ConversationEngine nunca vê initialize()/unload()/subprocess.
        """
        self._provider = provider

    @property
    def provider(self) -> Optional[Any]:
        return self._provider

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

    def discover_models(self) -> list[LocalModelInfo]:
        """Varre models/{tier}/*.gguf e extrai metadados."""
        models: list[LocalModelInfo] = []
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
    def _info_from_file(path: Path, tier: str) -> LocalModelInfo:
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
        return LocalModelInfo(
            path=path, name=name, tier=tier, size_gb=size_gb,
            quantization=quant, params_hint=params,
        )

    @staticmethod
    def _expected_size_mb(model: LocalModelInfo) -> Optional[float]:
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

    def _file_is_plausible(self, model: LocalModelInfo) -> bool:
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
        model: LocalModelInfo,
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

    # ------------------------------------------------------------------ #
    # Consultas ao catálogo
    # ------------------------------------------------------------------ #

    def get_catalog(self) -> ModelCatalog:
        """Acesso ao catálogo (camada de dados)."""
        return self.catalog

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Modelo catalogado por ID (não implica estar instalado)."""
        return self.catalog.get_model_by_id(model_id)

    def list_known_models(self) -> list[ModelInfo]:
        """Todos os modelos que o DaviOS conhece, instalados ou não."""
        return sorted(self.catalog.all_models(), key=lambda m: m.size_bytes)

    def list_installed_models(self) -> list[ModelInfo]:
        """Somente os modelos cujo arquivo já está no disco."""
        return sorted(self.catalog.get_installed_models(), key=lambda m: m.size_bytes)

    def list_available_to_install(self) -> list[ModelInfo]:
        """Modelos catalogados que ainda não estão instalados."""
        return sorted(
            self.catalog.get_known_not_installed_models(), key=lambda m: m.size_bytes
        )

    def list_models_by_tier(self, tier: str) -> list[ModelInfo]:
        """Modelos de um tier, aceitando alias ('forte' -> strong)."""
        canonical = normalize_tier(tier) or tier
        return sorted(
            self.catalog.get_models_by_tier(canonical), key=lambda m: m.size_bytes
        )

    def resolve_model_request(self, text: str) -> Optional[ModelInfo]:
        """Resolve linguagem natural para um modelo catalogado.

        Aceita aliases de tier ("forte"), IDs, nomes ("qwen3 8b") e
        quantização. Delega ao catálogo, que nunca inventa: sem casamento,
        retorna None e quem chamou decide pedir esclarecimento.
        """
        if not text:
            return None
        return self.catalog.resolve_alias(text)

    def tier_label(self, tier: str) -> str:
        """Rótulo humano do tier ('strong' -> 'Forte')."""
        return get_tier_label(tier)
    # ------------------------------------------------------------------ #
    # Compatibilidade com o hardware
    # ------------------------------------------------------------------ #

    def check_compatibility(
        self,
        model: Any,
        hardware: Optional[HardwareProfile] = None,
        require_disk: Optional[bool] = None,
    ) -> CompatibilityResult:
        """Avalia se um modelo pode rodar nesta máquina.

        Aceita ModelInfo, ID do catálogo ou alias em linguagem natural
        ("forte", "qwen3 8b"); o hardware é detectado automaticamente quando
        não é informado.

        Estrutura: level = recommended | possible | risky | incompatible |
        unknown.

        Política (bloqueia só o que é fatal):
        - RAM total < mínimo do modelo  -> incompatible (erro);
        - RAM livre < mínimo            -> risky (dá para fechar programas);
        - RAM livre >= recomendado      -> recommended; senão -> possible;
        - Disco insuficiente (só quando `require_disk`) -> incompatible (erro);
        - VRAM nunca bloqueia por estimativa: no máximo gera warning;
        - sem tamanho nem requisitos declarados -> unknown (não afirmamos nada).

        `require_disk` fica None por padrão e é derivado: se o modelo já está
        instalado não faz sentido exigir espaço para baixá-lo; se ainda não
        está, o disco passa a ser crítico.
        """
        resolved = self.resolve_model(model)
        if resolved is None:
            # Nome desconhecido não é "incompatível": é ausência de dado.
            return CompatibilityResult(
                compatible=None,
                status=LEVEL_UNKNOWN,
                level=LEVEL_UNKNOWN,
                errors=[f"Modelo não encontrado no catálogo: {model!r}"],
            )
        model = resolved
        hardware = hardware or self.detect_hardware()

        result = CompatibilityResult(
            model_id=model.id,
            compatible=None,
            status=LEVEL_UNKNOWN,
            installed=self.catalog.is_model_installed(model),
            ram_available_gb=hardware.ram_available_gb,
            ram_total_gb=hardware.ram_total_gb,
            free_disk_gb=hardware.disk_free_gb,
        )

        installed = self.catalog.is_model_installed(model)
        if require_disk is None:
            require_disk = not installed

        size_gb = model.size_gb
        min_ram = model.min_ram_gb
        rec_ram = model.recommended_ram_gb

        # --- RAM -------------------------------------------------------- #
        ram_available = hardware.ram_available_gb
        ram_total = hardware.ram_total_gb
        if min_ram > 0:
            if ram_total is not None and ram_total < min_ram:
                result.ram_ok = False
                result.errors.append(
                    f"RAM total insuficiente: o modelo pede {min_ram:.1f} GB e a "
                    f"máquina tem {ram_total:.1f} GB."
                )
            elif ram_available is not None:
                if ram_available < min_ram:
                    result.ram_ok = False
                    result.warnings.append(
                        f"Pouca RAM livre agora ({ram_available:.1f} GB); o mínimo "
                        f"estimado é {min_ram:.1f} GB. Feche outros programas."
                    )
                elif rec_ram > 0 and ram_available < rec_ram:
                    result.warnings.append(
                        f"RAM livre abaixo do recomendado ({rec_ram:.1f} GB). "
                        "Pode ficar lento em contextos grandes."
                    )

        # --- Disco ------------------------------------------------------ #
        result.required_disk_gb = round(size_gb * DISK_SAFETY_MARGIN, 2)
        free_disk = hardware.disk_free_gb
        if require_disk and size_gb > 0:
            if free_disk is None:
                result.warnings.append(
                    "Espaço livre em disco não foi detectado; não é possível "
                    "confirmar se o download cabe."
                )
            elif free_disk < size_gb:
                result.disk_ok = False
                result.errors.append(
                    f"Espaço insuficiente: o modelo ocupa ~{size_gb:.1f} GB e há "
                    f"{free_disk:.1f} GB livres."
                )
            elif free_disk < result.required_disk_gb:
                result.warnings.append(
                    f"Armazenamento apertado: ~{size_gb:.1f} GB necessários e "
                    f"{free_disk:.1f} GB livres."
                )

        # --- VRAM (informativo, nunca bloqueia) ------------------------- #
        rec_vram = model.recommended_vram_gb
        vram = hardware.gpu_vram_gb
        if rec_vram > 0:
            if vram is None:
                result.warnings.append(
                    f"VRAM não detectada; o recomendado é {rec_vram:.1f} GB. "
                    "Rodando em CPU o modelo continua funcionando, mais devagar."
                )
            elif vram < rec_vram:
                result.warnings.append(
                    f"VRAM abaixo do recomendado ({vram:.1f} GB de "
                    f"{rec_vram:.1f} GB). Reduza as camadas offloaded (-ngl)."
                )

        # --- Backend ---------------------------------------------------- #
        backend = (hardware.inference_backend or "none").lower()
        if backend in ("", "none", "unknown"):
            result.backend_ok = False
            result.warnings.append(
                "Backend de inferência não detectado; a inicialização pode falhar."
            )

        # --- Dados suficientes? ----------------------------------------- #
        # Sem tamanho nem requisitos declarados não há o que estimar. Dizer
        # "unknown" é mais honesto que afirmar compatibilidade: o consumo real
        # depende de quantização, backend, contexto (-c), camadas offloaded
        # (-ngl) e KV cache, que o catálogo não conhece.
        if size_gb <= 0 and min_ram <= 0 and rec_ram <= 0 and not result.errors:
            result.level = LEVEL_UNKNOWN
            result.status = LEVEL_UNKNOWN
            result.compatible = None
            result.warnings.append(
                f"{model.name} não declara tamanho nem requisitos no catálogo; "
                "não é possível estimar se roda nesta máquina."
            )
            return result

        # --- Nível final ------------------------------------------------ #
        if result.errors:
            result.level = LEVEL_INCOMPATIBLE
            result.compatible = False
            result.status = "incompatible"
        elif result.ram_ok and result.disk_ok:
            result.level = LEVEL_RECOMMENDED
            result.compatible = True
            result.status = "compatible"
            if result.warnings:
                result.level = LEVEL_POSSIBLE
        else:
            result.level = LEVEL_RISKY
            result.compatible = True
            result.status = "risky"

        return result

    # ------------------------------------------------------------------ #
    # Modelo ativo: estado lógico + persistência
    # ------------------------------------------------------------------ #
    # Esta camada NÃO inicia nem reinicia o llama-server. Ela apenas decide
    # qual modelo o DaviOS PRETENDE usar e registra isso em config/davios.json.
    # A troca real (parar A -> iniciar B -> health check -> rollback) pertence
    # ao LocalLlamaCppProvider, coordenada aqui numa etapa posterior. Portanto
    # `set_active_model()` significa "intenção registrada", NÃO "modelo
    # carregado" — quem sabe se há servidor no ar é o provider.

    @property
    def active_model_id(self) -> Optional[str]:
        """ID do modelo ativo registrado, ou None se nada foi registrado."""
        return self._read_persisted_active_id()

    @property
    def active_state(self) -> str:
        """Como o modelo ativo foi resolvido na última consulta.

        persisted     -> veio do active_model_id gravado e o arquivo existe;
        auto          -> nada gravado; escolha determinística, SEM gravar;
        missing       -> gravado, mas o ID não existe no catálogo;
        not_installed -> gravado e catalogado, mas o .gguf não está no disco;
        none          -> nada gravado e nenhum modelo instalado/utilizável;
        unset         -> ainda não houve consulta.
        """
        return self._active_state

    def detect_hardware(self, force: bool = False) -> HardwareProfile:
        """Retrato do hardware atual, delegado ao HardwareDetector.

        Fica em cache porque a detecção dispara processos auxiliares; use
        force=True para re-medir (ex.: depois de liberar memória).
        """
        if self._hardware is None or force:
            self._hardware = HardwareDetector().detect()
        return self._hardware

    def resolve_model(self, request: Any) -> Optional[ModelInfo]:
        """Aceita ModelInfo, ID exato ou alias em linguagem natural.

        Não inventa: sem casamento no catálogo, retorna None e quem chamou
        decide pedir esclarecimento (ou oferecer download).
        """
        if request is None:
            return None
        if isinstance(request, ModelInfo):
            return request
        text = str(request).strip()
        if not text:
            return None
        model = self.catalog.get_model_by_id(text)
        if model is not None:
            return model
        return self.catalog.resolve_alias(text)

    def list_models(self, only_installed: bool = False) -> list[ModelInfo]:
        """Modelos conhecidos pelo catálogo (ou só os instalados).

        Ordenados por tamanho, do menor para o maior.
        """
        models = (
            self.catalog.get_installed_models()
            if only_installed
            else self.catalog.all_models()
        )
        return sorted(models, key=lambda m: (m.size_bytes, m.id))

    # ------------------------------------------------------------------ #
    # Persistência de active_model_id
    # ------------------------------------------------------------------ #
    # Reutiliza o MESMO config/davios.json que o DaviosConfig já lê: uma chave
    # a mais no arquivo existente, sem banco novo. A leitura prefere o DISCO ao
    # objeto em memória, porque o arquivo é a fonte da verdade entre execuções.

    def _read_config_file(self) -> dict[str, Any]:
        """Lê o davios.json como dict. Nunca levanta exceção.

        Arquivo ausente, ilegível ou com JSON inválido devolve {} e o
        ModelManager segue com o que tem em memória — gerenciar modelos não
        pode impedir o DaviOS de subir.
        """
        try:
            if not self._config_path.exists():
                return {}
            with open(self._config_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as exc:
            logger.warning("[MODEL] falha ao ler %s: %s", self._config_path, exc)
            return {}

    def _read_persisted_active_id(self) -> Optional[str]:
        """active_model_id registrado em disco (ou None se não houver)."""
        raw = self._read_config_file().get("active_model_id")
        if raw is None:
            # Config injetado por teste/env pode ter o campo sem arquivo.
            raw = getattr(self.config, "active_model_id", None)
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    def _write_persisted_active_id(self, model_id: Optional[str]) -> bool:
        """Grava active_model_id preservando as outras chaves do davios.json.

        Escrita atômica: serializa num temporário no MESMO diretório e troca com
        os.replace(). Uma falha no meio da escrita não deixa o config truncado,
        e o arquivo antigo continua válido até a troca ser atômica.
        """
        data = self._read_config_file()
        data["active_model_id"] = model_id
        tmp_path: Optional[Path] = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".davios-", suffix=".tmp", dir=str(self._config_path.parent)
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            os.replace(tmp_path, self._config_path)
        except (OSError, ValueError, TypeError) as exc:
            logger.error("[MODEL][ERROR] falha ao persistir active_model_id: %s", exc)
            self._last_error = f"falha ao persistir active_model_id: {exc}"
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            return False

        # Espelha no objeto em memória para quem lê config.active_model_id.
        self.config.active_model_id = model_id
        logger.info("[MODEL] active_model_id persistido: %s", model_id)
        return True

    # ------------------------------------------------------------------ #
    # Escolha determinística do modelo inicial
    # ------------------------------------------------------------------ #

    def _choose_initial_model(
        self,
        hardware: Optional[HardwareProfile] = None,
        installed_only: bool = True,
    ) -> Optional[ModelInfo]:
        """Política determinística do modelo inicial, quando nada está gravado.

        Ordem de preferência:
          1. só modelos INSTALADOS — escolher um modelo ausente deixaria o
             DaviOS sem cérebro, contando com um download que ainda não existe;
          2. descarta os INCOMPATÍVEIS (falta de RAM/disco bloqueia de verdade);
          3. melhor nível de compatibilidade primeiro
             (recommended > possible > risky > unknown);
          4. empate: maior arquivo (mais capacidade), depois o id em ordem
             alfabética.

        O desempate final por id é o que torna a escolha REPRODUTÍVEL: a mesma
        máquina, com os mesmos arquivos, escolhe sempre o mesmo modelo.

        Não grava nada: persistir é decisão explícita de set_active_model().
        """
        hardware = hardware or self.detect_hardware()
        candidates = self.list_models(only_installed=installed_only)
        if not candidates:
            return None

        scored: list[tuple[int, int, str, ModelInfo]] = []
        for model in candidates:
            compat = self.check_compatibility(model, hardware, require_disk=False)
            if compat.level == LEVEL_INCOMPATIBLE:
                continue
            scored.append(
                (LEVEL_RANK.get(compat.level, 99), -model.size_bytes, model.id, model)
            )
        if not scored:
            return None
        scored.sort(key=lambda item: item[:3])
        return scored[0][3]

    def select_initial_model(
        self, hardware: Optional[HardwareProfile] = None
    ) -> Optional[ModelInfo]:
        """Modelo inicial do boot: o persistido, ou a escolha determinística.

        Prefere o active_model_id gravado (quando ele existe e está instalado).
        Sem registro, aplica `_choose_initial_model()` SEM gravar em disco: o
        DaviOS sobe sem alterar arquivos de configuração às escondidas.
        """
        persisted = self._read_persisted_active_id()
        if persisted:
            model = self.catalog.get_model_by_id(persisted)
            if model is not None and self.catalog.is_model_installed(model):
                self._active_state = ACTIVE_STATE_PERSISTED
                return model
        model = self._choose_initial_model(hardware)
        self._active_state = ACTIVE_STATE_AUTO if model else ACTIVE_STATE_NONE
        return model

    # ------------------------------------------------------------------ #
    # Modelo ativo
    # ------------------------------------------------------------------ #

    def get_active_model(
        self, hardware: Optional[HardwareProfile] = None
    ) -> Optional[ModelInfo]:
        """Modelo ativo segundo a configuração persistida.

        Casos, registrados em `active_state`:
        - nada gravado -> escolha determinística ('auto') ou None ('none'),
          SEM gravar em disco;
        - gravado e o arquivo existe -> 'persisted';
        - gravado e o id não existe no catálogo -> 'missing'; retorna None,
          porque NÃO fingimos que existe modelo ativo;
        - gravado, catalogado, mas o .gguf não está no disco -> 'not_installed';
          retorna None pelo mesmo motivo.

        Nunca levanta exceção: config apontando para modelo ausente é um estado
        que o chamador precisa poder mostrar ao usuário.
        """
        persisted = self._read_persisted_active_id()
        if persisted:
            model = self.catalog.get_model_by_id(persisted)
            if model is None:
                self._active_state = ACTIVE_STATE_MISSING
                self._last_error = (
                    f"active_model_id '{persisted}' não existe no catálogo"
                )
                logger.warning("[MODEL] %s", self._last_error)
                return None
            if not self.catalog.is_model_installed(model):
                self._active_state = ACTIVE_STATE_NOT_INSTALLED
                self._last_error = (
                    f"modelo ativo '{model.id}' não está instalado em "
                    f"{self.catalog.model_file_path(model)}"
                )
                logger.warning("[MODEL] %s", self._last_error)
                return None
            self._active_state = ACTIVE_STATE_PERSISTED
            return model

        model = self._choose_initial_model(hardware)
        self._active_state = ACTIVE_STATE_AUTO if model else ACTIVE_STATE_NONE
        return model

    def set_active_model(
        self, request: Any, require_installed: bool = True
    ) -> Optional[ModelInfo]:
        """Registra o modelo ativo (estado lógico + active_model_id em disco).

        Esta etapa NÃO reinicia o llama-server. A troca real (parar A ->
        iniciar B -> health check -> rollback) pertence ao provider e será
        coordenada aqui numa etapa separada; enquanto isso, esta função apenas
        decide e grava.

        Só grava depois de validar: id resolvido no catálogo, arquivo presente
        (por padrão) e compatibilidade não-incompatível. Se qualquer validação
        falhar, o modelo ativo ANTERIOR permanece intacto e o motivo fica em
        `last_error` — nunca deixamos o DaviOS sem referência de modelo por
        causa de um pedido inválido.
        """
        model = self.resolve_model(request)
        if model is None:
            self._last_error = f"modelo não reconhecido: {request!r}"
            logger.warning("[MODEL] %s", self._last_error)
            return None

        if require_installed and not self.catalog.is_model_installed(model):
            self._last_error = (
                f"{model.name} ({model.id}) não está instalado em "
                f"{self.catalog.model_file_path(model)}; "
                "download ainda não faz parte desta etapa"
            )
            logger.warning("[MODEL] %s", self._last_error)
            return None

        compat = self.check_compatibility(model, self.detect_hardware())
        if compat.level == LEVEL_INCOMPATIBLE:
            self._last_error = (
                f"{model.name} é incompatível com este hardware: "
                + "; ".join(compat.errors)
            )
            logger.warning("[MODEL][ERROR] %s", self._last_error)
            return None

        if not self._write_persisted_active_id(model.id):
            return None

        self._active_state = ACTIVE_STATE_PERSISTED
        self._last_error = ""
        logger.info("[MODEL] modelo ativo: %s", model.id)
        return model

    @property
    def last_error(self) -> str:
        """Último motivo de recusa/erro do gerenciamento (vazio se não houve)."""
        return self._last_error

    # ------------------------------------------------------------------ #
    # Status e ponte para o provider
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict[str, Any]:
        """Retrato do gerenciamento de modelos, pronto para exibição.

        `active_model_state` e `persisted_active_model_id` podem divergir: o
        registro é uma INTENÇÃO. Quem sabe se há servidor no ar é o provider,
        consultado em `provider_running` (None quando não há provider ligado).
        """
        active = self.get_active_model()
        hardware = self.detect_hardware()

        provider_running: Optional[bool] = None
        if self._provider is not None:
            checker = getattr(self._provider, "is_available", None)
            if callable(checker):
                try:
                    provider_running = bool(checker())
                except Exception as exc:  # defensivo: status nunca pode quebrar
                    logger.warning("[MODEL] provider falhou no health check: %s", exc)

        return {
            "active_model_id": active.id if active else None,
            "active_model": active.to_dict() if active else None,
            "active_model_state": self._active_state,
            "persisted_active_model_id": self._read_persisted_active_id(),
            "provider_running": provider_running,
            "catalog_total": len(self.catalog),
            "installed_count": len(self.catalog.get_installed_models()),
            "hardware": {
                "os": hardware.os,
                "cpu_name": hardware.cpu_name,
                "cpu_cores": hardware.cpu_cores,
                "ram_total_gb": hardware.ram_total_gb,
                "ram_available_gb": hardware.ram_available_gb,
                "gpu_name": hardware.gpu_name,
                "gpu_vram_gb": hardware.gpu_vram_gb,
                "disk_free_gb": hardware.disk_free_gb,
                "inference_backend": hardware.inference_backend,
            },
            "last_error": self._last_error or None,
        }

    def _as_local_model(self, model: ModelInfo) -> LocalModelInfo:
        """Converte um modelo do catálogo para a estrutura que o provider usa.

        O tamanho é medido no arquivo real quando ele existe; as estimativas do
        catálogo só entram quando não há arquivo para medir.
        """
        path = self.catalog.model_file_path(model)
        size_gb = model.size_gb
        try:
            if path.is_file():
                size_gb = round(path.stat().st_size / (1024 ** 3), 2)
        except OSError:
            pass
        return LocalModelInfo(
            path=path,
            name=model.name or model.filename or model.id,
            tier=normalize_tier(model.tier) or model.tier or "balanced",
            size_gb=size_gb,
            quantization=model.quantization or None,
            params_hint=model.parameters or None,
        )

    def build_selection(
        self,
        model: Any = None,
        hardware: Optional[HardwareProfile] = None,
    ) -> ModelSelection:
        """Monta a ModelSelection que o provider consome.

        Existe para que a troca futura não duplique conversão de dados e para
        que o ConversationEngine nunca precise conhecer GGUF: o engine pede ao
        ModelManager, que entrega esta estrutura ao provider. Hoje é leitura
        pura — não inicia nem reinicia nada.
        """
        hardware = hardware or self.detect_hardware()
        profile_name = self.select_profile(hardware)
        generation = (self._profiles.get(profile_name) or {}).get("generation", {})

        resolved = (
            self.resolve_model(model)
            if model is not None
            else self.get_active_model(hardware)
        )
        if resolved is None:
            return ModelSelection(
                profile=profile_name,
                reason=self._last_error or "Nenhum modelo ativo disponível.",
                generation=generation,
            )

        compat = self.check_compatibility(resolved, hardware)
        return ModelSelection(
            model=self._as_local_model(resolved),
            profile=profile_name,
            compatible=compat.level != LEVEL_INCOMPATIBLE,
            reason=(
                compat.errors[0] if compat.errors else f"Modelo ativo: {resolved.name}."
            ),
            generation=generation,
        )
