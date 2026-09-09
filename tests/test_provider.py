"""Testes para o LocalLlamaCppProvider."""

from __future__ import annotations

import json
import sys
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


class TestExtractText:
    """Testes para _extract_text."""

    def test_extract_content(self):
        """Testa extracao de content."""
        output = {
            "choices": [{
                "message": {"content": "Resposta normal"}
            }]
        }
        assert LocalLlamaCppProvider._extract_text(output) == "Resposta normal"

    def test_extract_reasoning_content(self):
        """Testa extracao de reasoning_content quando content vazio."""
        output = {
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "Raciocinio..."
                }
            }]
        }
        assert LocalLlamaCppProvider._extract_text(output) == "Raciocinio..."

    def test_extract_text_fallback(self):
        """Testa fallback para text."""
        output = {
            "choices": [{
                "text": "Resposta em text"
            }]
        }
        assert LocalLlamaCppProvider._extract_text(output) == "Resposta em text"

    def test_extract_empty(self):
        """Testa extracao com resposta vazia."""
        output = {"choices": [{}]}
        assert LocalLlamaCppProvider._extract_text(output) == ""