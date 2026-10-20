"""Testes da validacao de arquivos GGUF, da descoberta do llama-server.exe
em instalacoes irmas e do relato honesto de erros de inicializacao.

Nenhum teste toca a rede nem a maquina real: tudo roda em tmp_path com fakes.
"""

from pathlib import Path

from brain.model_catalog import (
    GGUF_MAGIC,
    gguf_header_error,
    is_valid_gguf,
)
from utils.llama_binary_downloader import (
    find_llama_server_exe,
    get_llama_bin_candidates,
    get_llama_bin_dir,
)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class TestGgufHeaderValidation:
    """A assinatura 'GGUF' separa 'modelo ausente' de 'modelo invalido'."""

    def test_arquivo_valido_passa(self, tmp_path):
        path = _write(tmp_path / "m.gguf", GGUF_MAGIC + b"\0" * 100)
        assert is_valid_gguf(path) is True
        assert gguf_header_error(path) == ""

    def test_arquivo_sem_magic_e_invalido(self, tmp_path):
        """new.gguf do usuario: existe, mas NAO e um GGUF."""
        path = _write(tmp_path / "new.gguf", b"<!DOCTYPE html><html>erro</html>")
        assert is_valid_gguf(path) is False
        error = gguf_header_error(path)
        assert "GGUF" in error
        assert "cabe" in error and "alho inv" in error

    def test_arquivo_vazio_e_invalido(self, tmp_path):
        path = _write(tmp_path / "vazio.gguf", b"")
        assert is_valid_gguf(path) is False
        assert "vazio" in gguf_header_error(path)

    def test_arquivo_inexistente_e_invalido(self, tmp_path):
        path = tmp_path / "nao_existe.gguf"
        assert is_valid_gguf(path) is False
        assert "n" in gguf_header_error(path) and "encontrado" in gguf_header_error(path)

    def test_gguf_truncado_no_cabecalho(self, tmp_path):
        """Arquivo com menos de 4 bytes nao pode ter a assinatura."""
        path = _write(tmp_path / "curto.gguf", b"GG")
        assert is_valid_gguf(path) is False
        assert "cabe" in gguf_header_error(path)



class TestLlamaServerDiscovery:
    """O llama-server.exe pode estar numa instalacao IRMA do DaviOS."""

    def test_candidatos_incluem_o_proprio_projeto(self, tmp_path):
        candidates = get_llama_bin_candidates(tmp_path)
        assert candidates[0] == tmp_path / "bin" / "llama.cpp"

    def test_candidatos_incluem_instalacao_irma(self, tmp_path):
        """../DaviOS/bin/llama.cpp entra na lista de candidatos."""
        sibling = tmp_path / "DaviOS"
        sibling.mkdir(parents=True, exist_ok=True)
        root = tmp_path / "DaviOS-telegram"
        root.mkdir(parents=True, exist_ok=True)

        candidates = get_llama_bin_candidates(root)
        assert sibling / "bin" / "llama.cpp" in candidates

    def test_encontra_executavel_na_instalacao_irma(self, tmp_path):
        sibling = tmp_path / "DaviOS"
        root = tmp_path / "DaviOS-telegram"
        exe = sibling / "bin" / "llama.cpp" / "llama-server.exe"
        _write(exe, b"fake-exe")
        root.mkdir(parents=True, exist_ok=True)

        found = find_llama_server_exe(root)
        assert found == exe

    def test_encontra_executavel_no_proprio_projeto(self, tmp_path):
        exe = tmp_path / "bin" / "llama.cpp" / "llama-server.exe"
        _write(exe, b"fake-exe")

        assert find_llama_server_exe(tmp_path) == exe

    def test_projeto_tem_prioridade_sobre_irma(self, tmp_path):
        """Se os dois existirem, vale o do proprio projeto."""
        sibling = tmp_path / "DaviOS"
        root = tmp_path / "DaviOS-telegram"
        own = root / "bin" / "llama.cpp" / "llama-server.exe"
        _write(own, b"fake-exe")
        _write(sibling / "bin" / "llama.cpp" / "llama-server.exe", b"fake-exe")
        root.mkdir(parents=True, exist_ok=True)

        assert find_llama_server_exe(root) == own

    def test_sem_executavel_devolve_none(self, tmp_path):
        root = tmp_path / "DaviOS-telegram"
        root.mkdir(parents=True, exist_ok=True)
        assert find_llama_server_exe(root) is None

    def test_get_llama_bin_dir_inalterado(self, tmp_path):
        assert get_llama_bin_dir(tmp_path) == tmp_path / "bin" / "llama.cpp"


class TestCatalogInstallationError:
    """O catalogo distingue 'nao instalado' de 'instalado mas invalido'."""

    def test_arquivo_invalido_nao_conta_como_instalado(self, tmp_path):
        from tests.test_model_catalog import _make_catalog, _model_entry

        entry = _model_entry("corrompido", "light")
        cat = _make_catalog(tmp_path, [entry])
        target = tmp_path / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"NAO E GGUF" * 10)

        assert cat.is_installed("corrompido") is False

    def test_motivo_acionavel_para_arquivo_invalido(self, tmp_path):
        from tests.test_model_catalog import _make_catalog, _model_entry

        entry = _model_entry("corrompido", "light")
        cat = _make_catalog(tmp_path, [entry])
        target = tmp_path / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"NAO E GGUF" * 10)

        model = cat.get_model_by_id("corrompido")
        error = cat.installation_error(model)
        assert "GGUF" in error

    def test_motivo_vazio_quando_modelo_valido(self, tmp_path):
        from tests.test_model_catalog import _make_catalog, _model_entry, _touch_model

        entry = _model_entry("ok", "light")
        cat = _make_catalog(tmp_path, [entry])
        _touch_model(tmp_path, entry)

        model = cat.get_model_by_id("ok")
        assert cat.installation_error(model) == ""


class TestManagerInvalidReason:
    """O ModelManager exclui .gguf invalidos sem destruir o arquivo."""

    def _manager(self, tmp_path):
        from brain.model_manager import ModelManager
        from config.davios_config import DaviosConfig

        config = DaviosConfig.load()
        config.models_dir = str(tmp_path / "models")
        config.external_models_dir = None
        config.profile_override = None
        return ModelManager(config)

    def _hardware(self):
        from core.hardware_detector import HardwareProfile

        return HardwareProfile(
            os="Teste",
            cpu_name="CPU Fake",
            cpu_cores=8,
            ram_total_gb=32.0,
            ram_available_gb=24.0,
            gpu_name="GPU Fake",
            gpu_vram_gb=None,
            gpu_vendor="Fake",
            inference_backend="llama_cpp",
            disk_free_gb=200.0,
        )

    def test_arquivo_sem_magic_nao_e_selecionado(self, tmp_path):
        """O .gguf invalido NAO pode virar o modelo escolhido."""
        models_dir = tmp_path / "models" / "light"
        _write(models_dir / "modelo-ruim.gguf", b"NAO E GGUF" * 100)

        manager = self._manager(tmp_path)
        selection = manager.select_model(self._hardware())
        assert selection.model is None

    def test_motivo_da_seleciona_menciona_o_arquivo_invalido(self, tmp_path):
        """A razao da falha diz que o arquivo e invalido (nao 'nao encontrado')."""
        models_dir = tmp_path / "models" / "light"
        _write(models_dir / "modelo-ruim.gguf", b"NAO E GGUF" * 100)

        manager = self._manager(tmp_path)
        selection = manager.select_model(self._hardware())
        assert "GGUF" in selection.reason

    def test_arquivo_invalido_nao_e_destruido(self, tmp_path):
        models_dir = tmp_path / "models" / "light"
        _write(models_dir / "modelo-ruim.gguf", b"NAO E GGUF" * 100)

        manager = self._manager(tmp_path)
        manager.select_model(self._hardware())
        assert (models_dir / "modelo-ruim.gguf").exists()

    def test_arquivo_valido_e_selecionado(self, tmp_path):
        models_dir = tmp_path / "models" / "light"
        _write(models_dir / "modelo-ok.gguf", b"GGUF" + b"\0" * 4096)

        manager = self._manager(tmp_path)
        selection = manager.select_model(self._hardware())
        assert selection.model is not None
        assert selection.model.name == "modelo-ok"


class TestProviderDiagnostics:
    """O provider relata o motivo REAL da falha de inicializacao."""

    def _provider(self, tmp_path, model_path):
        from brain.providers.local_llama_cpp_provider import LocalLlamaCppProvider
        from config.davios_config import DaviosConfig
        from tests.test_provider import make_selection

        config = DaviosConfig()
        config.llama_server_exe = str(tmp_path / "llama-server.exe")
        provider = LocalLlamaCppProvider(config=config)
        provider.selection = make_selection(model_path)
        return provider

    def test_gguf_invalido_e_relatado_como_invalido(self, tmp_path):
        """O arquivo existe mas nao e GGUF: o erro nao pode dizer 'nao encontrado'."""
        model = tmp_path / "modelo-ruim.gguf"
        model.write_bytes(b"NAO E GGUF" * 100)

        provider = self._provider(tmp_path, model)
        assert provider.initialize() is False
        assert "invalido" in provider.diagnostics["[BACKEND]"]
        assert "GGUF" in provider.diagnostics["[MODEL]"]

    def test_arquivo_inexistente_e_relatado_como_ausente(self, tmp_path):
        model = tmp_path / "nao_existe.gguf"

        provider = self._provider(tmp_path, model)
        assert provider.initialize() is False
        assert "nao encontrado" in provider.diagnostics["[MODEL]"]
        assert "modelo ausente" in provider.diagnostics["[BACKEND]"]
