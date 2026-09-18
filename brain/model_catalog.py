"""
model_catalog.py — Camada de dados pura para o catálogo de modelos do DaviOS.

Responsabilidades deste módulo:
- definir a representação canônica de um modelo (ModelInfo);
- carregar e consultar o catálogo de modelos conhecidos (ModelCatalog);
- constantes de tier (light/balanced/strong/very_strong) e aliases.

Este módulo NÃO faz:
- detecção de hardware;
- download de arquivos;
- manipulação de processos / llama-server;
- seleção dinâmica de modelo;
- persistência de qual modelo está ativo.

Essas responsabilidades pertencem ao ModelManager.

LEIA ANTES DE EDITAR OS NÚMEROS DO CATÁLOGO
------------------------------------------
Todos os valores de tamanho e de memória deste módulo são ESTIMATIVAS, não
medições. O consumo real de RAM/VRAM depende de vários fatores simultâneos:
quantização, backend, tamanho do contexto (-c), número de camadas offloaded
para GPU (-ngl), batch size, tamanho do KV cache, uso de mmap e a versão do
runtime. Por isso cada modelo carrega as flags `size_is_estimate` e
`memory_is_estimate`, que devem permanecer True até que alguém meça o consumo
real na máquina de referência.

Nenhum `download_url` vem preenchido de fábrica: `download_available=False`
significa "o DaviOS ainda não sabe baixar este modelo com segurança".
Preencha uma URL apenas depois de confirmar a origem oficial do arquivo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.davios_config import CONFIG_DIR, DaviosConfig

logger = logging.getLogger("davios.models.catalog")

# Raiz do projeto. config/ fica sempre um nível abaixo dela; usamos isso para
# resolver os caminhos do catálogo sem depender do diretório de trabalho atual.
PROJECT_ROOT = Path(CONFIG_DIR).resolve().parent

# =============================================================================
# CONSTANTES DE TIER
# =============================================================================

# Ordem canônica dos tiers, do menor/mais leve para o maior/mais forte.
TIERS: List[str] = ["light", "balanced", "strong", "very_strong"]

# Aliases aceitos em linguagem natural (português e inglês).
# A resolução de alias vive no ModelCatalog.resolve_alias().
TIER_SYNONYMS: Dict[str, str] = {
    "leve": "light",
    "light": "light",
    "mais leve": "light",
    "mais_leve": "light",
    "pequeno": "light",
    "balanceado": "balanced",
    "equilibrado": "balanced",
    "balanced": "balanced",
    "medio": "balanced",
    "médio": "balanced",
    "forte": "strong",
    "strong": "strong",
    "mais forte": "strong",
    "mais_forte": "strong",
    "poderoso": "strong",
    "potente": "strong",
    "muito forte": "very_strong",
    "muito_forte": "very_strong",
    "very_strong": "very_strong",
    "very-strong": "very_strong",
    "máximo": "very_strong",
    "maximo": "very_strong",
}

# Nome amigável de cada tier, usado em respostas ao usuário.
TIER_LABELS: Dict[str, str] = {
    "light": "Leve",
    "balanced": "Balanceado",
    "strong": "Forte",
    "very_strong": "Muito Forte",
}

# Ponte com o sistema legado: config/model_profiles.json usa os nomes
# LIGHT / BALANCED / PERFORMANCE. Mantemos o mapa para não quebrar o
# ModelManager nem os perfis existentes.
LEGACY_TIER_MAP: Dict[str, str] = {
    "LIGHT": "light",
    "BALANCED": "balanced",
    "PERFORMANCE": "strong",
}

# Ordem de preferência quando o ModelManager precisa escolher um tier.
TIER_ORDER: List[str] = ["light", "balanced", "strong", "very_strong"]


# =============================================================================
# REPRESENTAÇÃO CANÔNICA DE UM MODELO
# =============================================================================

@dataclass
class ModelInfo:
    """Representação canônica de um modelo conhecido pelo DaviOS.

    Convenções de estimativa (importantes para manutenção):
    - `size_bytes`: tamanho aproximado do arquivo GGUF. Quando `size_is_estimate`
      é True, o valor foi derivado de fórmula (parâmetros x bits-por-peso da
      quantização / 8), não de um arquivo baixado e medido.
    - `min_ram_gb` / `recommended_ram_gb`: faixa estimada de RAM.
      Mínimo = suficiente para carregar os pesos e rodar contexto pequeno.
      Recomendado = margem para contexto maior, KV cache, buffers do backend
      e o resto do sistema operacional.
    - `min_vram_gb` / `recommended_vram_gb`: mesma lógica para GPU. Só são
      relevantes quando o backend usa offload de camadas; em execução 100% CPU
      ficam em 0.0, o que significa "não aplicável", não "não precisa".
    """

    # --- Identidade -----------------------------------------------------
    id: str                          # único: "qwen3-4b-q4km"
    name: str                        # legível: "Qwen3 4B"
    family: str = ""                 # "Qwen3"
    tier: str = "balanced"           # light|balanced|strong|very_strong
    format: str = "GGUF"
    quantization: str = ""           # "Q4_K_M", "Q8_0", ...
    parameters: str = ""             # "4B"
    architecture: str = ""           # "Qwen3ForCausalLM"

    # --- Arquivo --------------------------------------------------------
    filename: str = ""
    path: str = ""                   # caminho relativo/absoluto do .gguf
    size_bytes: int = 0
    size_is_estimate: bool = True

    # --- Requisitos de memória (estimativas, ver docstring) -------------
    min_ram_gb: float = 0.0
    recommended_ram_gb: float = 0.0
    min_vram_gb: float = 0.0
    recommended_vram_gb: float = 0.0
    memory_is_estimate: bool = True

    # --- Inferência -----------------------------------------------------
    context_window: int = 0          # janela nativa em tokens
    max_tokens_default: int = 0      # sugestão de max_tokens por resposta

    # --- Download -------------------------------------------------------
    download_url: str = ""
    download_available: bool = False
    download_note: str = ""          # explica por que não há download

    # --- Metadados ------------------------------------------------------
    description: str = ""
    source: str = ""                 # origem declarada dos números
    enabled: bool = True

    @property
    def size_gb(self) -> float:
        """Tamanho em GB (GiB) derivado de size_bytes."""
        return round(self.size_bytes / (1024 ** 3), 2) if self.size_bytes > 0 else 0.0

    def is_installed(self) -> bool:
        """True apenas se o arquivo do modelo existe no disco.

        Um modelo catalogado mas não baixado retorna False. Nunca considere
        um modelo instalado apenas porque ele existe no catálogo.
        """
        if not self.path:
            return False
        try:
            return Path(self.path).is_file()
        except OSError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serializa para dicionário (JSON-friendly)."""
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "tier": self.tier,
            "format": self.format,
            "quantization": self.quantization,
            "parameters": self.parameters,
            "architecture": self.architecture,
            "filename": self.filename,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size_gb": self.size_gb,
            "size_is_estimate": self.size_is_estimate,
            "min_ram_gb": self.min_ram_gb,
            "recommended_ram_gb": self.recommended_ram_gb,
            "min_vram_gb": self.min_vram_gb,
            "recommended_vram_gb": self.recommended_vram_gb,
            "memory_is_estimate": self.memory_is_estimate,
            "context_window": self.context_window,
            "max_tokens_default": self.max_tokens_default,
            "download_url": self.download_url,
            "download_available": self.download_available,
            "download_note": self.download_note,
            "description": self.description,
            "source": self.source,
            "enabled": self.enabled,
            "is_installed": self.is_installed(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Cria ModelInfo a partir de um dict do catálogo JSON.

        Campos desconhecidos são ignorados para manter o catálogo tolerante a
        metadados extras (ex.: comentários/justificativas em chaves "_*").
        """
        valid = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid}
        filtered.pop("size_gb", None)
        filtered.pop("is_installed", None)
        return cls(**filtered)


# =============================================================================
# CATÁLOGO DE MODELOS (camada de dados)
# =============================================================================

# Caminho padrão do catálogo. Absoluto de propósito: o DaviOS pode ser iniciado
# de qualquer diretório de trabalho.
DEFAULT_CATALOG_RELPATH = "config/model_catalog.json"
DEFAULT_CATALOG_PATH = Path(CONFIG_DIR) / "model_catalog.json"


def resolve_model_path(path: str, project_root: Optional[Path] = None) -> Path:
    """Resolve o caminho declarado no catálogo para um caminho absoluto.

    Justificativa: o catálogo guarda caminhos relativos à raiz do projeto
    (ex.: "models/balanced/Qwen3-4B-Q4_K_M.gguf"), porque caminhos absolutos
    quebrariam ao mover o projeto de pasta. A conversão para absoluto acontece
    aqui, em um único lugar, para que `ModelInfo.is_installed()` seja confiável
    independentemente do diretório de trabalho.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(project_root).resolve() if project_root else PROJECT_ROOT
    return (root / candidate).resolve()


class ModelCatalog:
    """Acesso aos DADOS do catálogo de modelos conhecidos pelo DaviOS.

    Esta classe responde perguntas sobre dados:
    - quais modelos o DaviOS conhece;
    - quais metadados cada modelo tem;
    - qual modelo está realmente instalado no disco;
    - qual modelo corresponde a um alias ("forte", "qwen3 8b", ...).

    Ela NÃO decide qual modelo usar, NÃO verifica hardware, NÃO baixa arquivos
    e NÃO inicia processos. Isso é responsabilidade do ModelManager.

    Um modelo no catálogo é apenas "conhecido": estar no catálogo NÃO significa
    estar instalado. Use `is_installed()` / `get_installed_models()` para isso.
    """

    def __init__(
        self,
        config: Optional[DaviosConfig] = None,
        catalog_path: Optional[str] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self.config = config or DaviosConfig.load()
        self.project_root = Path(project_root).resolve() if project_root else PROJECT_ROOT
        if catalog_path:
            self._catalog_path = resolve_model_path(catalog_path, self.project_root)
        else:
            self._catalog_path = DEFAULT_CATALOG_PATH
        self._models: Dict[str, ModelInfo] = {}
        self._metadata: Dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------ #
    # Carregamento
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Lê o catálogo do disco. Falha de leitura não derruba o DaviOS."""
        self._models.clear()
        self._metadata.clear()

        if not self._catalog_path.exists():
            logger.warning("[CATALOG] arquivo não encontrado: %s", self._catalog_path)
            return

        try:
            raw = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("[CATALOG][ERROR] falha ao ler %s: %s", self._catalog_path, exc)
            return

        if not isinstance(raw, dict):
            logger.error("[CATALOG][ERROR] formato inesperado (esperado objeto JSON)")
            return

        # Metadados globais (chaves que começam com "_" são documentação).
        self._metadata = {
            k: v for k, v in raw.items() if k != "models"
        }

        entries = raw.get("models", [])
        if not isinstance(entries, list):
            logger.error("[CATALOG][ERROR] 'models' deve ser uma lista")
            return

        for entry in entries:
            if not isinstance(entry, dict):
                logger.warning("[CATALOG] entrada inválida ignorada: %r", entry)
                continue
            if not entry.get("enabled", True):
                continue
            try:
                model = ModelInfo.from_dict(entry)
            except TypeError as exc:
                logger.warning("[CATALOG] modelo inválido ignorado (%s): %s", entry.get("id"), exc)
                continue
            if not model.id:
                logger.warning("[CATALOG] modelo sem 'id' ignorado")
                continue
            if model.path:
                # Converte "models/..." (relativo ao projeto) em caminho absoluto.
                model.path = str(resolve_model_path(model.path, self.project_root))
            if model.tier not in TIERS:
                logger.warning(
                    "[CATALOG] tier desconhecido '%s' em %s (mantido, mas não resolve alias)",
                    model.tier, model.id,
                )
            self._models[model.id] = model

        logger.info("[CATALOG] %d modelos conhecidos carregados", len(self._models))

    def reload(self) -> None:
        """Recarrega o catálogo do disco (útil depois de um download)."""
        self.load()
        logger.info("[CATALOG] catálogo recarregado: %d modelos", len(self._models))

    # ------------------------------------------------------------------ #
    # Propriedades
    # ------------------------------------------------------------------ #

    @property
    def catalog_path(self) -> Path:
        """Caminho do arquivo de catálogo em uso."""
        return self._catalog_path

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadados globais do catálogo (versão, notas, etc.)."""
        return dict(self._metadata)

    # ------------------------------------------------------------------ #
    # Consultas básicas
    # ------------------------------------------------------------------ #

    def all_models(self) -> List[ModelInfo]:
        """Todos os modelos conhecidos (instalados ou não)."""
        return list(self._models.values())

    def get_model_by_id(self, model_id: str) -> Optional[ModelInfo]:
        """Busca por ID exato. Retorna None se não existir."""
        if not model_id:
            return None
        return self._models.get(model_id)

    def get_models_by_tier(self, tier: str) -> List[ModelInfo]:
        """Modelos de um tier. Aceita alias em português ('forte')."""
        resolved = TIER_SYNONYMS.get(str(tier).strip().lower(), tier)
        return [m for m in self._models.values() if m.tier == resolved]

    def get_models_by_family(self, family: str) -> List[ModelInfo]:
        """Modelos de uma família (ex.: 'Qwen3'), case-insensitive."""
        target = str(family).strip().lower()
        return [m for m in self._models.values() if m.family.lower() == target]

    def get_models_by_parameter_size(self, parameters: str) -> List[ModelInfo]:
        """Modelos por tamanho de parâmetros (ex.: '8B')."""
        target = str(parameters).strip().lower()
        return [m for m in self._models.values() if m.parameters.lower() == target]

    # ------------------------------------------------------------------ #
    # Estado de instalação
    # ------------------------------------------------------------------ #

    def model_file_path(self, model: ModelInfo) -> Path:
        """Caminho absoluto esperado do arquivo .gguf de um modelo.

        Não verifica existência — para isso use `is_model_installed()`.
        """
        if not model.path:
            return Path()
        return resolve_model_path(model.path, self.project_root)

    def is_model_installed(self, model: ModelInfo) -> bool:
        """True se o arquivo .gguf do modelo existe no disco.

        Esta é a verificação canônica de "instalado". Um download parcial NÃO
        conta como instalado: o arquivo temporário (.part) fica fora do caminho
        final, e o downloader só renomeia depois de validar o arquivo.
        """
        path = self.model_file_path(model)
        if not path or not str(path):
            return False
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    def get_installed_models(self) -> List[ModelInfo]:
        """Modelos cujo arquivo .gguf existe no disco."""
        return [m for m in self._models.values() if self.is_model_installed(m)]

    def get_known_not_installed_models(self) -> List[ModelInfo]:
        """Modelos catalogados que ainda não foram baixados."""
        return [m for m in self._models.values() if not self.is_model_installed(m)]

    def get_downloadable_models(self) -> List[ModelInfo]:
        """Modelos com origem de download confirmada.

        Modelos sem `download_url` verificado ficam de fora, mesmo que sejam
        conhecidos pelo catálogo.
        """
        return [
            m for m in self._models.values()
            if m.download_available and m.download_url
        ]

    def is_installed(self, model_id: str) -> bool:
        """True se o model_id existe no catálogo E está no disco."""
        model = self.get_model_by_id(model_id)
        return bool(model and self.is_model_installed(model))

    # ------------------------------------------------------------------ #
    # Busca e resolução de alias
    # ------------------------------------------------------------------ #

    def search_models(self, query: str) -> List[ModelInfo]:
        """Busca textual simples por nome, família, tier, quantização ou params.

        A busca é por substring, case-insensitive. Retorna a lista ordenada por
        tamanho crescente, deixando a decisão de "melhor" para o chamador.
        """
        q = str(query or "").strip().lower()
        if not q:
            return []
        found: List[ModelInfo] = []
        for m in self._models.values():
            haystack = " ".join([
                m.id, m.name, m.family, m.tier, m.quantization, m.parameters, m.description,
            ]).lower()
            if q in haystack:
                found.append(m)
        return sorted(found, key=lambda m: m.size_bytes)

    def resolve_alias(self, alias: str) -> Optional[ModelInfo]:
        """Resolve linguagem natural para um modelo do catálogo.

        Ordem de tentativa:
          1. ID exato;
          2. alias de tier ('forte' -> strong). Se HOUVER modelo do tier instalado,
             devolve o maior deles (melhor qualidade que o usuário já pode usar).
             Se NENHUM estiver instalado, devolve o MENOR do tier — o ponto de
             entrada mais barato em disco/RAM, deixando a decisão de baixar para
             o chamador;
          3. nome/parametrização exatos ('qwen3 8b' -> Qwen3 8B);
          4. substring única dentro do catálogo.

        Nunca inventa antecedente: se nada casar, retorna None. Cabe ao
        ModelManager decidir se oferece download ou pede esclarecimento.
        """
        if not alias:
            return None
        key = str(alias).strip().lower()
        if not key:
            return None

        # 1. ID exato
        if key in self._models:
            return self._models[key]

        # 2. Alias de tier (ou nome do tier)
        tier = TIER_SYNONYMS.get(key)
        if tier is None and key in TIERS:
            tier = key
        if tier:
            candidates = [m for m in self._models.values() if m.tier == tier]
            if candidates:
                installed = [m for m in candidates if self.is_model_installed(m)]
                if installed:
                    # Já existe algo do tier no disco: entregue o de melhor
                    # qualidade que o usuário pode usar agora (maior arquivo).
                    return max(installed, key=lambda m: m.size_bytes)
                # Nada instalado nesse tier: escolha o MENOR candidato, que é o
                # ponto de entrada mais barato do tier (menos disco, mais chance
                # de caber na RAM) e, em empate de parâmetros, a quantização
                # canônica (Q4_K_M). Quem chama é que decide se oferece download.
                return min(candidates, key=lambda m: m.size_bytes)

        # 3a. Casamento exato por nome ou parâmetros
        for m in self._models.values():
            if key in (m.name.lower(), m.parameters.lower()):
                return m

        # 3b. "qwen3 8b" / "qwen3-8b" -> parâmetros + família
        tokens = key.replace("-", " ").replace("/", " ").split()
        for token in tokens:
            if len(token) > 1 and token.endswith("b") and token[:-1].replace(".", "").isdigit():
                same = self.get_models_by_parameter_size(token)
                if len(same) == 1:
                    return same[0]
                if same:
                    # desempate: quantização citada explicitamente na string
                    for m in same:
                        if m.quantization and m.quantization.lower() in key:
                            return m
                    return sorted(same, key=lambda m: m.size_bytes)[0]

        # 4. Substring única
        matches = self.search_models(key)
        if len(matches) == 1:
            return matches[0]

        return None

    def tiers_present(self) -> List[str]:
        """Tiers que realmente têm modelos no catálogo, em ordem canônica."""
        present = {m.tier for m in self._models.values()}
        return [t for t in TIER_ORDER if t in present]

    # ------------------------------------------------------------------ #
    # Conveniências
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._models

    def __iter__(self):
        return iter(self._models.values())


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def get_tier_label(tier: str) -> str:
    """Nome amigável de um tier ('strong' -> 'Forte').

    Aceita alias em português. Se o tier for desconhecido, devolve a própria
    string recebida, sem inventar rótulo.
    """
    if not tier:
        return ""
    key = str(tier).strip().lower()
    canonical = TIER_SYNONYMS.get(key, key)
    return TIER_LABELS.get(canonical, str(tier))


def normalize_tier(tier: str) -> Optional[str]:
    """Normaliza um tier/alias para o nome canônico. None se desconhecido."""
    if not tier:
        return None
    key = str(tier).strip().lower()
    if key in TIERS:
        return key
    return TIER_SYNONYMS.get(key)


def tier_from_legacy(legacy_tier: str) -> Optional[str]:
    """Converte LIGHT/BALANCED/PERFORMANCE para o tier canônico.

    Usado para ler config/model_profiles.json sem duplicar a lógica.
    """
    if not legacy_tier:
        return None
    return LEGACY_TIER_MAP.get(str(legacy_tier).strip().upper())


def build_catalog(
    config: Optional[DaviosConfig] = None,
    catalog_path: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> ModelCatalog:
    """Factory explícita, útil em testes (permite catálogo alternativo)."""
    return ModelCatalog(config=config, catalog_path=catalog_path, project_root=project_root)


__all__ = [
    "ModelInfo",
    "ModelCatalog",
    "TIERS",
    "TIER_ORDER",
    "TIER_SYNONYMS",
    "TIER_LABELS",
    "LEGACY_TIER_MAP",
    "DEFAULT_CATALOG_RELPATH",
    "DEFAULT_CATALOG_PATH",
    "PROJECT_ROOT",
    "resolve_model_path",
    "get_tier_label",
    "normalize_tier",
    "tier_from_legacy",
    "build_catalog",
]