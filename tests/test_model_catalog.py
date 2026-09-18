"""Testes do catálogo de modelos (brain/model_catalog.py).

Escopo: camada de DADOS apenas. Estes testes não baixam nada, não iniciam
llama-server e não tocam rede. Os arquivos .gguf usados são placeholders
minúsculos criados em diretórios temporários.
"""

import json
from pathlib import Path

import pytest

from brain.model_catalog import (
    DEFAULT_CATALOG_PATH,
    LEGACY_TIER_MAP,
    TIERS,
    TIER_SYNONYMS,
    ModelCatalog,
    ModelInfo,
    get_tier_label,
    normalize_tier,
    resolve_model_path,
    tier_from_legacy,
)


@pytest.fixture
def catalog():
    """Catálogo real do projeto (config/model_catalog.json)."""
    return ModelCatalog()


def _model_entry(
    model_id,
    tier,
    quant="Q4_K_M",
    size_bytes=1_000_000,
    path=None,
    download_available=False,
    download_url="",
):
    """Entrada mínima e válida de catálogo, para testes isolados."""
    return {
        "id": model_id,
        "name": model_id.upper(),
        "family": "TestFamily",
        "tier": tier,
        "quantization": quant,
        "parameters": "1B",
        "filename": f"{model_id}.gguf",
        "path": path or f"models/{tier}/{model_id}.gguf",
        "size_bytes": size_bytes,
        "download_available": download_available,
        "download_url": download_url,
    }


def _write_catalog(tmp_path, models):
    """Escreve um catálogo temporário e devolve o caminho do arquivo."""
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "model_catalog.json"
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return path


def _make_catalog(tmp_path, models):
    """Catálogo isolado, ancorado em tmp_path como raiz do projeto."""
    path = _write_catalog(tmp_path, models)
    return ModelCatalog(catalog_path=str(path), project_root=tmp_path)


def _touch_model(tmp_path, entry, size=64):
    """Cria o arquivo .gguf correspondente à entrada, dentro de tmp_path."""
    target = tmp_path / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\0" * size)
    return target


class TestCatalogFile:
    """O arquivo de catálogo em si é válido e segue a política declarada."""

    def test_catalog_file_exists(self):
        assert DEFAULT_CATALOG_PATH.is_file()

    def test_catalog_file_is_valid_json(self):
        data = json.loads(DEFAULT_CATALOG_PATH.read_text(encoding="utf-8"))
        assert isinstance(data.get("models"), list)
        assert data["models"], "o catálogo não pode estar vazio"

    def test_loads_expected_number_of_models(self, catalog):
        assert len(catalog) == 8

    def test_declares_estimation_policy(self, catalog):
        """Os números são estimativas; a política precisa estar documentada."""
        policy = catalog.metadata.get("_estimation_policy")
        assert isinstance(policy, dict)
        assert "size_bytes" in policy
        assert "min_ram_gb" in policy

    def test_all_tiers_are_canonical(self, catalog):
        for model in catalog:
            assert model.tier in TIERS, f"{model.id} usa tier inválido {model.tier}"

    def test_required_metadata_present(self, catalog):
        for model in catalog:
            assert model.id and model.name and model.family
            assert model.quantization and model.parameters
            assert model.filename and model.path
            assert model.size_bytes > 0
            assert model.min_ram_gb > 0
            assert model.recommended_ram_gb >= model.min_ram_gb
            assert model.context_window > 0

    def test_estimates_are_flagged_and_sourced(self, catalog):
        """Todo número precisa de origem; medido e estimado não se misturam.

        Regra: `memory_is_estimate` permanece True (ninguém mediu RAM/VRAM
        nesta máquina de referência). `size_is_estimate=False` só é aceitável
        quando a própria origem declara que o tamanho foi medido.
        """
        for model in catalog:
            assert model.source, f"{model.id} não declara a origem dos números"
            assert model.memory_is_estimate is True
            if model.size_is_estimate is False:
                assert "MEDIDO" in model.source.upper(), (
                    f"{model.id} afirma ter tamanho medido, mas não diz de onde veio"
                )

    def test_no_model_claims_download_without_url(self, catalog):
        """download_available=True exige URL real; sem URL, o flag é False."""
        for model in catalog:
            if model.download_available:
                assert model.download_url.startswith("http")
            else:
                assert model.download_url == ""
                assert model.download_note, (
                    f"{model.id} não explica por que o download está indisponível"
                )

    def test_vram_not_invented_for_cpu_only_entries(self, catalog):
        """VRAM em 0.0 significa 'não aplicável', não um requisito real."""
        for model in catalog:
            assert model.min_vram_gb >= 0.0
            assert model.recommended_vram_gb >= model.min_vram_gb


class TestCatalogQueries:
    """Consultas básicas sobre o catálogo."""

    def test_get_model_by_id_found(self, catalog):
        model = catalog.get_model_by_id("qwen3-4b-q4km")
        assert model is not None
        assert model.name == "Qwen3 4B"
        assert model.quantization == "Q4_K_M"

    def test_get_model_by_id_not_found_is_none(self, catalog):
        assert catalog.get_model_by_id("modelo-inexistente") is None
        assert catalog.get_model_by_id("") is None

    def test_get_models_by_family(self, catalog):
        models = catalog.get_models_by_family("Qwen3")
        assert models
        assert all(m.family == "Qwen3" for m in models)

    def test_get_models_by_family_is_case_insensitive(self, catalog):
        assert len(catalog.get_models_by_family("qwen3")) == len(
            catalog.get_models_by_family("QWEN3")
        )

    def test_get_models_by_tier(self, catalog):
        strong = catalog.get_models_by_tier("strong")
        assert strong
        assert all(m.tier == "strong" for m in strong)

    def test_get_models_by_tier_accepts_portuguese_alias(self, catalog):
        assert len(catalog.get_models_by_tier("forte")) == len(
            catalog.get_models_by_tier("strong")
        )

    def test_get_models_by_parameter_size(self, catalog):
        """Mesmo tamanho de parâmetros com quantizações diferentes = 2 modelos."""
        models = catalog.get_models_by_parameter_size("8B")
        assert len(models) == 2
        assert all(m.parameters == "8B" for m in models)
        assert {m.quantization for m in models} == {"Q4_K_M", "Q5_K_M"}

    def test_search_matches_name_family_and_quantization(self, catalog):
        assert catalog.search_models("qwen3")
        assert catalog.search_models("Q8_0")
        assert catalog.search_models("zzz-inexistente") == []

    def test_search_results_sorted_by_size(self, catalog):
        sizes = [m.size_bytes for m in catalog.search_models("qwen3")]
        assert sizes == sorted(sizes)

    def test_tiers_present_matches_canonical_order(self, catalog):
        present = catalog.tiers_present()
        assert present == [t for t in TIERS if t in present]
        assert "light" in present and "strong" in present

    def test_len_iter_and_contains(self, catalog):
        assert len(catalog) == 8
        assert "qwen3-4b-q4km" in catalog
        assert "nada-a-ver" not in catalog
        assert len(list(catalog)) == 8

class TestCatalogIsolation:
    """O catálogo tolera arquivos ausentes, corrompidos e entradas inválidas."""

    def test_missing_catalog_file_does_not_crash(self, tmp_path):
        cat = ModelCatalog(catalog_path=str(tmp_path / "nao_existe.json"))
        assert len(cat) == 0
        assert cat.get_model_by_id("qualquer") is None

    def test_malformed_json_does_not_crash(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{ isso nao e json", encoding="utf-8")
        cat = ModelCatalog(catalog_path=str(bad))
        assert len(cat) == 0

    def test_non_object_json_does_not_crash(self, tmp_path):
        bad = tmp_path / "list.json"
        bad.write_text("[1, 2, 3]", encoding="utf-8")
        cat = ModelCatalog(catalog_path=str(bad))
        assert len(cat) == 0

    def test_disabled_model_is_skipped(self, tmp_path):
        entry = _model_entry("desligado", "light")
        entry["enabled"] = False
        cat = _make_catalog(tmp_path, [entry])
        assert len(cat) == 0

    def test_entry_without_id_is_skipped(self, tmp_path):
        entry = _model_entry("x", "light")
        del entry["id"]
        cat = _make_catalog(tmp_path, [entry])
        assert len(cat) == 0

    def test_extra_metadata_keys_are_tolerated(self, tmp_path):
        """Chaves de documentação (ex.: _justification) não quebram o parse."""
        entry = _model_entry("com-extra", "light")
        entry["_justification"] = {"size": "porque sim"}
        entry["size_is_estimate"] = True
        entry["campo_que_nao_existe"] = 123
        cat = _make_catalog(tmp_path, [entry])
        assert cat.get_model_by_id("com-extra") is not None

    def test_reload_rereads_disk(self, tmp_path):
        entry = _model_entry("recarregavel", "light")
        path = _write_catalog(tmp_path, [entry])
        cat = ModelCatalog(catalog_path=str(path), project_root=tmp_path)
        assert len(cat) == 1
        _write_catalog(tmp_path, [entry, _model_entry("novo", "light")])
        assert len(cat) == 1, "sem reload, o catálogo mantém o estado antigo"
        cat.reload()
        assert len(cat) == 2
class TestInstalledDetection:
    """'Catalogado' e 'instalado' são estados diferentes."""

    def test_known_model_is_not_installed_by_default(self, tmp_path):
        entry = _model_entry("nao-baixado", "light")
        cat = _make_catalog(tmp_path, [entry])
        model = cat.get_model_by_id("nao-baixado")
        assert model is not None
        assert cat.is_model_installed(model) is False
        assert cat.get_installed_models() == []
        assert len(cat.get_known_not_installed_models()) == 1

    def test_model_with_file_on_disk_is_installed(self, tmp_path):
        entry = _model_entry("baixado", "light")
        cat = _make_catalog(tmp_path, [entry])
        model = cat.get_model_by_id("baixado")
        _touch_model(tmp_path, entry)
        assert cat.is_model_installed(model) is True
        assert cat.get_installed_models() == [model]
        assert cat.is_installed("baixado") is True

    def test_empty_file_is_not_installed(self, tmp_path):
        """Arquivo de 0 byte (download abortado) NÃO conta como instalado."""
        entry = _model_entry("vazio", "light")
        cat = _make_catalog(tmp_path, [entry])
        _touch_model(tmp_path, entry, size=0)
        assert cat.is_installed("vazio") is False

    def test_partial_download_file_is_not_installed(self, tmp_path):
        """O arquivo temporário .part não é o caminho final do modelo."""
        entry = _model_entry("parcial", "light")
        cat = _make_catalog(tmp_path, [entry])
        part = tmp_path / (entry["path"] + ".part")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(b"\0" * 128)
        assert cat.is_installed("parcial") is False
        assert cat.get_installed_models() == []

    def test_directory_in_place_of_file_is_not_installed(self, tmp_path):
        entry = _model_entry("pasta", "light")
        cat = _make_catalog(tmp_path, [entry])
        (tmp_path / entry["path"]).mkdir(parents=True, exist_ok=True)
        assert cat.is_installed("pasta") is False

    def test_is_installed_unknown_id_is_false(self, catalog):
        assert catalog.is_installed("modelo-que-nao-existe") is False

    def test_get_downloadable_models_requires_confirmed_url(self, tmp_path):
        sem_url = _model_entry("sem-url", "light", download_available=False)
        com_url = _model_entry(
            "com-url", "light",
            download_available=True,
            download_url="https://example.invalid/model.gguf",
        )
        cat = _make_catalog(tmp_path, [sem_url, com_url])
        assert [m.id for m in cat.get_downloadable_models()] == ["com-url"]

    def test_model_file_path_is_absolute(self, catalog):
        model = catalog.get_model_by_id("qwen3-4b-q4km")
        assert catalog.model_file_path(model).is_absolute()

    def test_real_catalog_detects_locally_present_model(self, catalog):
        """Se o arquivo do 4B Q4_K_M existe no disco, ele conta como instalado.

        Este teste não baixa nada: apenas confirma que a detecção de instalado
        funciona sobre o catálogo real do projeto.
        """
        model = catalog.get_model_by_id("qwen3-4b-q4km")
        assert model is not None
        if catalog.model_file_path(model).is_file():
            assert catalog.is_installed("qwen3-4b-q4km") is True
            assert "qwen3-4b-q4km" in [m.id for m in catalog.get_installed_models()]


class TestAliasResolution:
    """Resolução de linguagem natural para modelos do catálogo.

    Regra central: resolve quando dá, devolve None quando não dá.
    Nunca inventa um antecedente.
    """

    def test_resolve_by_exact_id(self, catalog):
        model = catalog.resolve_alias("qwen3-4b-q4km")
        assert model is not None and model.id == "qwen3-4b-q4km"

    def test_resolve_light_alias_without_installed_models(self, catalog):
        """Sem nada instalado, o alias aponta para o ponto de entrada mais barato.

        O DaviOS não deve sugerir o maior arquivo do tier para quem ainda não
        tem nada baixado — isso empurraria o usuário para o maior download.
        """
        model = catalog.resolve_alias("leve")
        assert model is not None
        assert model.tier == "light"
        assert model.parameters == "0.6B"

    def test_resolve_balanced_alias(self, catalog):
        model = catalog.resolve_alias("balanceado")
        assert model is not None and model.tier == "balanced"

    def test_resolve_strong_alias(self, catalog):
        model = catalog.resolve_alias("forte")
        assert model is not None and model.tier == "strong"

    def test_resolve_prefers_installed_model_of_tier(self, tmp_path):
        """Instalado tem prioridade; sem instalado, vence o menor do tier."""
        pequeno = _model_entry("leve-a", "light", size_bytes=1_000_000)
        grande = _model_entry("leve-b", "light", size_bytes=9_000_000)
        cat = _make_catalog(tmp_path, [pequeno, grande])
        # Nada instalado: menor candidato (não sugere o maior download).
        assert cat.resolve_alias("leve").id == "leve-a"
        # Com o maior instalado, é ele que o usuário consegue usar agora.
        _touch_model(tmp_path, grande)
        assert cat.resolve_alias("leve").id == "leve-b"

    def test_resolve_by_parameter_size(self, catalog):
        model = catalog.resolve_alias("8b")
        assert model is not None and model.parameters == "8B"

    def test_resolve_compound_name(self, catalog):
        model = catalog.resolve_alias("qwen3 8b")
        assert model is not None and model.id == "qwen3-8b-q4km"

    def test_resolve_disambiguates_by_quantization(self, catalog):
        """'qwen3 4b q8_0' precisa escolher o 4B Q8_0, não o Q4_K_M."""
        model = catalog.resolve_alias("qwen3 4b q8_0")
        assert model is not None
        assert model.id == "qwen3-4b-q80"
        assert model.quantization == "Q8_0"

    def test_quantizations_are_distinct_models(self, catalog):
        """Q4_K_M e Q8_0 do mesmo 4B são entradas separadas, não sobrescritas."""
        q4 = catalog.get_model_by_id("qwen3-4b-q4km")
        q8 = catalog.get_model_by_id("qwen3-4b-q80")
        assert q4 is not None and q8 is not None
        assert q4.quantization != q8.quantization
        assert q4.path != q8.path
        assert q4.size_bytes != q8.size_bytes

    def test_resolve_unknown_alias_returns_none(self, catalog):
        """Não existe antecedente: devolve None em vez de adivinhar."""
        assert catalog.resolve_alias("modelo mistico 999b") is None
        assert catalog.resolve_alias("") is None
        assert catalog.resolve_alias(None) is None

    def test_resolve_case_insensitive(self, catalog):
        assert catalog.resolve_alias("FORTE").id == catalog.resolve_alias("forte").id


class TestTierHelpers:
    """Normalização de tier e ponte com o model_profiles.json legado."""

    def test_tier_synonyms_cover_portuguese_aliases(self):
        for alias in ("leve", "balanceado", "forte", "muito forte"):
            assert alias in TIER_SYNONYMS
            assert TIER_SYNONYMS[alias] in TIERS

    def test_normalize_tier(self):
        assert normalize_tier("FORTE") == "strong"
        assert normalize_tier("strong") == "strong"
        assert normalize_tier("muito forte") == "very_strong"

    def test_normalize_unknown_tier_is_none(self):
        assert normalize_tier("turbo") is None
        assert normalize_tier("") is None

    def test_get_tier_label_human_readable(self):
        assert get_tier_label("light") == "Leve"
        assert get_tier_label("balanced") == "Balanceado"
        assert get_tier_label("strong") == "Forte"
        assert get_tier_label("very_strong") == "Muito Forte"

    def test_get_tier_label_accepts_portuguese_alias(self):
        assert get_tier_label("forte") == "Forte"

    def test_get_tier_label_unknown_returns_input(self):
        """Sem rótulo inventado: devolve o que recebeu."""
        assert get_tier_label("turbo") == "turbo"

    def test_legacy_profile_mapping_preserves_profiles_json(self):
        """LIGHT/BALANCED/PERFORMANCE continuam mapeáveis sem quebrar."""
        assert tier_from_legacy("LIGHT") == "light"
        assert tier_from_legacy("BALANCED") == "balanced"
        assert tier_from_legacy("PERFORMANCE") == "strong"
        assert tier_from_legacy("desconhecido") is None
        assert LEGACY_TIER_MAP["PERFORMANCE"] == "strong"


class TestModelInfo:
    """Comportamento do ModelInfo isoladamente."""

    def test_size_gb_derived_from_bytes(self):
        info = ModelInfo(id="x", name="X", size_bytes=1024 ** 3)
        assert info.size_gb == 1.0

    def test_size_gb_zero_when_unknown(self):
        assert ModelInfo(id="x", name="X").size_gb == 0.0

    def test_is_installed_false_without_path(self):
        assert ModelInfo(id="x", name="X").is_installed() is False

    def test_roundtrip_to_from_dict(self):
        info = ModelInfo(
            id="round", name="Round", tier="balanced", quantization="Q5_K_M",
            parameters="8B", path="models/balanced/round.gguf",
            size_bytes=5_000_000_000, min_ram_gb=6.0, recommended_ram_gb=12.0,
        )
        restored = ModelInfo.from_dict(info.to_dict())
        assert restored.id == info.id
        assert restored.quantization == info.quantization
        assert restored.size_bytes == info.size_bytes

    def test_from_dict_ignores_computed_and_unknown_keys(self):
        info = ModelInfo.from_dict({
            "id": "x", "name": "X",
            "size_gb": 99.0,           # derivado, não aceito na entrada
            "is_installed": True,      # derivado, não aceito na entrada
            "campo_aleatorio": "abc",  # ignorado
        })
        assert info.id == "x"
        assert info.path == ""

    def test_from_dict_rejects_missing_required_field(self):
        with pytest.raises(TypeError):
            ModelInfo.from_dict({"name": "sem id"})

    def test_resolve_model_path_absolute_input_unchanged(self, tmp_path):
        absolute = tmp_path / "a" / "b.gguf"
        assert resolve_model_path(str(absolute)) == absolute.resolve()

    def test_resolve_model_path_relative_to_project_root(self, tmp_path):
        resolved = resolve_model_path("models/light/a.gguf", tmp_path)
        assert resolved == (tmp_path / "models" / "light" / "a.gguf").resolve()

