"""Testes para o LocalLlamaCppProvider."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.llm_provider import LLMRequest
from brain.providers.local_llama_cpp_provider import LocalLlamaCppProvider


@pytest.fixture
def mock_config():
    """Configuracao mock para testes."""
    config = MagicMock()
    config.debug = False
    config.use_gpu = False
    config.threads = 4
    config.context_window = 2048
    config.gpu_layers = 0
    config.temperature = 0.7
    config.repeat_penalty = 1.3
    config.repeat_last_n = 256
    config.models_dir = str(ROOT / "models")
    return config


@pytest.fixture
def mock_selection():
    """Selecao de modelo mock."""
    model = MagicMock()
    model.name = "test-model"
    model.path = ROOT / "models" / "balanced" / "test.gguf"
    model.size_gb = 1.0

    selection = MagicMock()
    selection.model = model
    selection.generation = {"threads": 4, "context_window": 2048}
    selection.profile = "BALANCED"
    return selection


class TestHealthCheck:
    """Testes de health check."""

    def test_health_ok(self, mock_config, mock_selection):
        """Testa health check quando servidor responde OK."""
        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "external"

        with patch.object(provider, "_check_existing_server", return_value=True):
            assert provider.is_available() is True

    def test_health_fail(self, mock_config, mock_selection):
        """Testa health check quando servidor falha."""
        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "external"

        with patch.object(provider, "_check_existing_server", return_value=False):
            assert provider.is_available() is False


class TestGenerate:
    """Testes de geracao de resposta."""

    def test_generate_success(self, mock_config, mock_selection):
        """Testa geracao bem sucedida."""
        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "cpu"

        mock_response = {
            "choices": [{
                "message": {"content": "Ola! Sou o DaviOS."},
                "finish_reason": "stop"
            }],
            "usage": {"completion_tokens": 10}
        }

        with patch("brain.providers.local_llama_cpp_provider.url_request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            request = LLMRequest(prompt="Ola!", max_tokens=64)
            response = provider.generate(request)

            assert response.text == "Ola! Sou o DaviOS."
            assert response.provider == "local_llama_cpp"
            assert response.tokens_generated == 10

    def test_generate_payload_includes_repeat_defaults(
        self, mock_config, mock_selection
    ):
        """Payload inclui os sampling defaults do config quando o request nao
        os especifica: repeat_penalty=1.3 e repeat_last_n=256."""
        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "cpu"

        mock_response = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }

        with patch("brain.providers.local_llama_cpp_provider.url_request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            request = LLMRequest(prompt="Ola!", max_tokens=64)
            provider.generate(request)

        sent_request = mock_urlopen.call_args[0][0]
        payload = json.loads(sent_request.data.decode("utf-8"))
        assert payload["repeat_penalty"] == 1.3
        assert payload["repeat_last_n"] == 256

    def test_generate_repeat_params_explicit_overrides_config(
        self, mock_config, mock_selection
    ):
        """Valores explicitos em LLMRequest sobrepoem os defaults do config."""
        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "cpu"

        mock_response = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1},
        }

        with patch("brain.providers.local_llama_cpp_provider.url_request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            request = LLMRequest(
                prompt="Ola!", max_tokens=64,
                repeat_penalty=1.5, repeat_last_n=512,
            )
            provider.generate(request)

        sent_request = mock_urlopen.call_args[0][0]
        payload = json.loads(sent_request.data.decode("utf-8"))
        assert payload["repeat_penalty"] == 1.5
        assert payload["repeat_last_n"] == 512

    def test_external_server_reports_real_loaded_model(
        self, mock_config, mock_selection
    ):
        """Com servidor ja ativo, o nome do modelo exibido vem da consulta
        real ao servidor, nao da selecao teorica (que aponta p/ qwen2.5)."""
        mock_selection.model.name = "qwen2.5-1.5b-instruct-q4_k_m"
        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        real_path = r"C:\Users\Davi\DaviOS\models\balanced\Qwen3-4B-Q4_K_M.gguf"
        with patch.object(
                provider, "_find_server_exe",
                return_value=Path("bin/llama.cpp/llama-server.exe")), \
                patch.object(provider, "_check_existing_server",
                             return_value=True), \
                patch.object(provider, "_query_loaded_model",
                             return_value=real_path):
            assert provider.initialize() is True
        assert provider.diagnostics["[MODEL]"] == (
            "Qwen3-4B-Q4_K_M.gguf (carregado no servidor externo)."
        )
        assert provider.health_check()["model"] == "Qwen3-4B-Q4_K_M.gguf"

    def test_external_server_unknown_model_does_not_guess(
        self, mock_config, mock_selection
    ):
        """Se nao for possivel obter o nome real, loga aviso explicito e nao
        exibe um nome teorico como se fosse certeza."""
        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        with patch.object(
                provider, "_find_server_exe",
                return_value=Path("bin/llama.cpp/llama-server.exe")), \
                patch.object(provider, "_check_existing_server",
                             return_value=True), \
                patch.object(provider, "_query_loaded_model",
                             return_value=None):
            assert provider.initialize() is True
        assert "desconocido" in provider.diagnostics["[MODEL]"]
        assert "nao verificado" in provider.diagnostics["[MODEL]"]
        assert provider.health_check()["model"] == "test-model"

    def test_generate_timeout(self, mock_config, mock_selection):
        """Testa timeout na geracao."""
        from urllib.error import URLError

        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "cpu"

        with patch("brain.providers.local_llama_cpp_provider.url_request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("timeout")

            request = LLMRequest(prompt="Ola!", max_tokens=64)

            with pytest.raises(Exception) as exc_info:
                provider.generate(request)

            assert "Falha na geracao" in str(exc_info.value)

    def test_generate_connection_refused(self, mock_config, mock_selection):
        """Testa conexao recusada."""
        from urllib.error import URLError

        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "cpu"

        with patch("brain.providers.local_llama_cpp_provider.url_request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection refused")

            request = LLMRequest(prompt="Ola!", max_tokens=64)

            with pytest.raises(Exception) as exc_info:
                provider.generate(request)

            assert "Falha na geracao" in str(exc_info.value)

    def test_generate_http_error(self, mock_config, mock_selection):
        """Testa erro HTTP."""
        from urllib.error import HTTPError

        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "cpu"

        with patch("brain.providers.local_llama_cpp_provider.url_request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(
                "http://127.0.0.1:8080", 500, "Internal Server Error", {}, None
            )

            request = LLMRequest(prompt="Ola!", max_tokens=64)

            with pytest.raises(Exception) as exc_info:
                provider.generate(request)

            assert "Falha na geracao" in str(exc_info.value)

    def test_generate_invalid_json(self, mock_config, mock_selection):
        """Testa JSON invalido na resposta."""
        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "cpu"

        with patch("brain.providers.local_llama_cpp_provider.url_request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b"invalid json"
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            request = LLMRequest(prompt="Ola!", max_tokens=64)

            with pytest.raises(Exception):
                provider.generate(request)

    def test_generate_empty_content(self, mock_config, mock_selection):
        """Testa resposta com conteudo vazio (reasoning mode)."""
        with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
            provider = LocalLlamaCppProvider(
                config=mock_config,
                selection=mock_selection,
            )
        provider._initialized = True
        provider._available = True
        provider._backend_used = "cpu"

        mock_response = {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "Vou responder..."
                },
                "finish_reason": "length"
            }],
            "usage": {"completion_tokens": 64}
        }

        with patch("brain.providers.local_llama_cpp_provider.url_request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            request = LLMRequest(prompt="Ola!", max_tokens=64)
            response = provider.generate(request)

            # Deve usar reasoning_content como fallback
            assert response.text == "Vou responder..."


class FakeProcess:
    """Processo fake que simula o subprocess.Popen do llama-server."""

    def __init__(self):
        self._alive = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return 0


def make_gguf(tmp_path, name):
    """Cria um arquivo .gguf falso em disco."""
    path = tmp_path / name
    path.write_bytes(b"fake gguf")
    return path


def make_selection(model_path, name=None, generation=None):
    """Cria uma ModelSelection falsa com caminho real em disco."""
    model = MagicMock()
    model.name = name or Path(model_path).name
    model.path = Path(model_path)
    model.size_gb = 1.5

    selection = MagicMock()
    selection.model = model
    selection.generation = generation or {"threads": 4, "context_window": 2048}
    selection.profile = "BALANCED"
    return selection


def make_running_provider(mock_config, selection):
    """Provider no estado 'modelo A rodando' (processo proprio + healthy).

    is_available() vira um stub que representa o servidor respondendo a
    /health e _find_server_exe() vira um stub, de modo que a troca nao
    dependa de um llama-server.exe real.
    """
    with patch("brain.providers.local_llama_cpp_provider.ModelManager"):
        provider = LocalLlamaCppProvider(config=mock_config, selection=selection)

    provider._initialized = True
    provider._available = True
    provider._backend_used = "cpu"
    provider._process = FakeProcess()
    provider._loaded_model = str(selection.model.path)
    provider.is_available = MagicMock(return_value=True)
    provider._find_server_exe = MagicMock(
        return_value=Path("bin") / "llama.cpp" / "llama-server.exe"
    )
    return provider


def install_start_server(provider, results):
    """Instala um _start_server fake.

    `results` eh consumido em ordem; quando True simula o sucesso real
    (cria processo e preenche _loaded_model/_backend_used). Devolve
    (calls, processes) para inspecao do comportamento observavel.
    """
    queue = list(results)
    calls: list[dict] = []
    processes: list[FakeProcess] = []

    def fake_start(server_exe, model_path, n_threads, ctx_size, backend, gpu_layers):
        calls.append(
            {
                "path": str(model_path),
                "backend": backend,
                "selection": provider.selection,
            }
        )
        ok = queue.pop(0) if queue else False
        if ok:
            process = FakeProcess()
            processes.append(process)
            provider._process = process
            provider._initialized = True
            provider._available = True
            provider._backend_used = backend
            provider._loaded_model = Path(model_path).name
        return ok

    provider._start_server = fake_start
    return calls, processes


class TestSwitchModel:
    """Testes para troca transacional de modelo (switch_model)."""

    def test_switch_same_model_does_not_restart(self, mock_config, tmp_path):
        """Mesmo modelo nao reinicia o servidor nem troca a selecao."""
        a_path = make_gguf(tmp_path, "a.gguf")
        selection_a = make_selection(a_path, name="a")
        provider = make_running_provider(mock_config, selection_a)
        original_process = provider._process
        calls, _ = install_start_server(provider, [])

        same_model = make_selection(a_path, name="a")
        result = provider.switch_model(same_model)

        assert result is True
        assert calls == []  # nao reiniciou
        assert provider.selection is selection_a
        assert provider._process is original_process
        assert original_process.terminated is False

    def test_switch_invalid_selection_returns_false(self, mock_config, tmp_path):
        """Selecao invalida retorna False sem modificar o estado atual."""
        a_path = make_gguf(tmp_path, "a.gguf")
        selection_a = make_selection(a_path, name="a")
        provider = make_running_provider(mock_config, selection_a)
        original_process = provider._process

        assert provider.switch_model(None) is False

        no_model = MagicMock()
        no_model.model = None
        assert provider.switch_model(no_model) is False

        no_path = MagicMock()
        no_path.model = MagicMock()
        no_path.model.path = None
        assert provider.switch_model(no_path) is False

        # Estado atual intacto: A continua selecionado e o processo vive.
        assert provider.selection is selection_a
        assert provider._process is original_process
        assert original_process.terminated is False

    def test_switch_missing_gguf_does_not_touch_current_server(
        self, mock_config, tmp_path
    ):
        """GGUF inexistente retorna False sem parar o servidor atual."""
        a_path = make_gguf(tmp_path, "a.gguf")
        selection_a = make_selection(a_path, name="a")
        provider = make_running_provider(mock_config, selection_a)
        original_process = provider._process
        calls, _ = install_start_server(provider, [])

        missing = make_selection(tmp_path / "missing.gguf", name="missing")
        result = provider.switch_model(missing)

        assert result is False
        assert calls == []
        assert provider.selection is selection_a
        assert provider._process is original_process
        assert original_process.terminated is False

    def test_switch_external_server_is_not_killed(self, mock_config, tmp_path):
        """Servidor externo nao eh controlado nem encerrado pelo provider."""
        a_path = make_gguf(tmp_path, "a.gguf")
        selection_a = make_selection(a_path, name="a")
        provider = make_running_provider(mock_config, selection_a)
        provider._backend_used = "external"
        provider._process = None

        cleanup = MagicMock()
        provider._cleanup_process = cleanup
        calls, _ = install_start_server(provider, [True])

        b_path = make_gguf(tmp_path, "b.gguf")
        selection_b = make_selection(b_path, name="b")
        result = provider.switch_model(selection_b)

        assert result is False
        cleanup.assert_not_called()  # nao mata processo externo
        assert calls == []  # nem tenta iniciar outro servidor
        assert provider._process is None
        assert provider.selection is selection_a
        assert provider._backend_used == "external"

    def test_switch_a_to_b_success(self, mock_config, tmp_path):
        """Troca A -> B com sucesso: commit somente apos identidade real."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)
        old_process = provider._process
        calls, processes = install_start_server(provider, [True])
        provider._query_loaded_model = MagicMock(return_value=str(b_path))

        result = provider.switch_model(selection_b)

        assert result is True
        assert provider.selection is selection_b
        assert provider._loaded_model == str(b_path)  # identidade REAL do servidor
        assert provider._backend_used == "cpu"
        assert provider._available is True
        assert provider._initialized is True
        assert calls == [
            {"path": str(b_path), "backend": "cpu", "selection": selection_a}
        ]
        # A antigo foi encerrado; B esta ativo.
        assert old_process.terminated is True
        assert provider._process is processes[0]
        assert processes[0].terminated is False

    def test_switch_uses_real_identity_from_server(self, mock_config, tmp_path):
        """_loaded_model usa o valor real devolvido pelo servidor e nao o
        basename do arquivo pedido (ex.: servidor responde caminho absoluto)."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)
        install_start_server(provider, [True])
        server_answer = str(b_path.resolve())
        provider._query_loaded_model = MagicMock(return_value=server_answer)

        assert provider.switch_model(selection_b) is True
        assert provider._loaded_model == server_answer

    def test_switch_selection_stays_a_until_b_confirmed(self, mock_config, tmp_path):
        """self.selection continua apontando para A durante a tentativa de B
        e so muda depois da confirmacao."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)

        observed = {}

        def fake_start(server_exe, model_path, n_threads, ctx_size, backend, gpu_layers):
            observed["selection_during_b"] = provider.selection
            provider._process = FakeProcess()
            provider._initialized = True
            provider._available = True
            provider._backend_used = backend
            provider._loaded_model = Path(model_path).name
            return True

        provider._start_server = fake_start
        provider._query_loaded_model = MagicMock(return_value=str(b_path))

        assert provider.switch_model(selection_b) is True
        assert observed["selection_during_b"] is selection_a  # A durante a troca
        assert provider.selection is selection_b  # commit apos confirmacao

    def test_switch_start_b_fails_triggers_rollback(self, mock_config, tmp_path):
        """Falha ao iniciar B: rollback recria A e retorna False."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)
        calls, processes = install_start_server(provider, [False, True])
        provider._query_loaded_model = MagicMock(return_value=str(a_path))

        result = provider.switch_model(selection_b)

        assert result is False  # a troca falhou...
        assert provider.selection is selection_a  # ...mesmo com A restaurado
        assert provider._available is True
        assert provider._initialized is True
        assert [c["path"] for c in calls] == [str(b_path), str(a_path)]
        # A foi recriado no rollback.
        assert provider._process is processes[0]
        assert provider._loaded_model == str(a_path)

    def test_switch_b_health_fails_triggers_rollback(self, mock_config, tmp_path):
        """B inicia mas nunca fica saudavel: processo de B eh limpo e A volta."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)
        old_process = provider._process

        created = []

        def fake_popen(*args, **kwargs):
            process = FakeProcess()
            created.append(process)
            return process

        # health de B falha; health de A (rollback) passa
        with patch(
            "brain.providers.local_llama_cpp_provider.subprocess.Popen",
            side_effect=fake_popen,
        ), patch.object(provider, "_wait_for_server", side_effect=[False, True]), \
                patch.object(provider, "_query_loaded_model", return_value=str(a_path)):
            result = provider.switch_model(selection_b)

        assert result is False
        assert provider.selection is selection_a
        assert old_process.terminated is True
        # processo de B foi criado e limpo; A foi recriado
        assert created[0].terminated is True
        assert provider._process is created[1]
        assert provider._process.terminated is False

    def test_switch_b_identity_mismatch_triggers_rollback(
        self, mock_config, tmp_path
    ):
        """B sobe mas o modelo real nao corresponde: rollback para A."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        wrong_path = make_gguf(tmp_path, "wrong.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)
        _, processes = install_start_server(provider, [True, True])
        # identidade de B errada; identidade de A correta no rollback
        provider._query_loaded_model = MagicMock(
            side_effect=[str(wrong_path), str(a_path)]
        )

        result = provider.switch_model(selection_b)

        assert result is False
        assert provider.selection is selection_a
        assert processes[0].terminated is True  # B nao ficou orfao
        assert provider._process is processes[1]  # A recriado
        assert provider._available is True

    def test_switch_b_unknown_identity_triggers_rollback(self, mock_config, tmp_path):
        """Identidade desconhecida (None) nao eh tratada como sucesso."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)
        _, processes = install_start_server(provider, [True, True])
        provider._query_loaded_model = MagicMock(side_effect=[None, str(a_path)])

        result = provider.switch_model(selection_b)

        assert result is False
        assert provider.selection is selection_a
        assert processes[0].terminated is True
        assert provider._process is processes[1]

    def test_switch_no_orphan_process_after_failure(self, mock_config, tmp_path):
        """Nenhum processo controlado pelo provider sobra ativo apos falha."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        wrong_path = make_gguf(tmp_path, "wrong.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)
        _, processes = install_start_server(provider, [True, True])
        provider._query_loaded_model = MagicMock(
            side_effect=[str(wrong_path), str(a_path)]
        )

        provider.switch_model(selection_b)

        # somente A (recriado) permanece vivo
        alive = [p for p in processes if p.terminated is False]
        assert alive == [provider._process]
        assert provider._process is processes[1]

    def test_switch_rollback_validates_a_identity(self, mock_config, tmp_path):
        """Se a identidade de A nao puder ser confirmada, o rollback falha."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        wrong_path = make_gguf(tmp_path, "wrong.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)
        _, processes = install_start_server(provider, [True, True])
        # B errado e A tambem errado -> rollback nao confirma A
        provider._query_loaded_model = MagicMock(
            side_effect=[str(wrong_path), str(wrong_path)]
        )

        result = provider.switch_model(selection_b)

        assert result is False
        assert provider.selection is selection_a  # estado logico de A
        assert provider._available is False
        assert provider._initialized is False
        assert processes[0].terminated is True  # B limpo

    def test_switch_rollback_failure_leaves_provider_unavailable(
        self, mock_config, tmp_path
    ):
        """Se B e o rollback de A falharem, o provider fica indisponivel."""
        a_path = make_gguf(tmp_path, "a.gguf")
        b_path = make_gguf(tmp_path, "b.gguf")
        selection_a = make_selection(a_path, name="a")
        selection_b = make_selection(b_path, name="b")
        provider = make_running_provider(mock_config, selection_a)
        install_start_server(provider, [False, False])
        provider._query_loaded_model = MagicMock(return_value=str(a_path))

        result = provider.switch_model(selection_b)

        assert result is False
        assert provider._available is False
        assert provider._initialized is False
        assert provider._process is None

    def test_switch_concurrent_calls_are_serialized(self, mock_config, tmp_path):
        """Duas trocas simultaneas nao executam ao mesmo tempo (lock)."""
        a_path = make_gguf(tmp_path, "a.gguf")
        selection_a = make_selection(a_path, name="a")
        provider = make_running_provider(mock_config, selection_a)

        state = {"active": 0, "max_active": 0}
        results = []

        def fake_inner(new_selection):
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.05)
            state["active"] -= 1
            return True

        provider._switch_model_inner = fake_inner

        selection_b = make_selection(make_gguf(tmp_path, "b.gguf"), name="b")
        selection_c = make_selection(make_gguf(tmp_path, "c.gguf"), name="c")

        threads = [
            threading.Thread(
                target=lambda sel: results.append(provider.switch_model(sel)),
                args=(sel,),
            )
            for sel in (selection_b, selection_c)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert results == [True, True]
        assert state["max_active"] == 1  # nunca houve execucao simultanea
        assert all(not thread.is_alive() for thread in threads)
