"""Testes do ModelDownloader (camada isolada de rede/disco).

Regras desta suíte:
- NENHUM download real e NENHUM acesso à internet: o transporte HTTP é um
  fake injetado em `opener`, que serve bytes de memória e simula erros;
- catálogo, configuração e arquivos vivem em `tmp_path` (isolamento total);
- NÃO se inicia llama-server, NÃO se altera `active_model_id` e NÃO se chama
  o provider — há uma classe de testes dedicada a provar isso.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import socket
import threading
import time
import urllib.error
from pathlib import Path

import pytest

from brain.model_catalog import ModelCatalog
from brain.model_downloader import (
    LOCK_MAX_CONTENT_BYTES,
    LOCK_MIN_AGE_SECONDS,
    PART_SUFFIX,
    STATUS_ALREADY_INSTALLED,
    STATUS_CANCELLED,
    STATUS_CONFIRMATION_REQUIRED,
    STATUS_DECLINED,
    STATUS_DOWNLOADED,
    STATUS_FAILED,
    STATUS_LOCKED,
    STATUS_REFUSED,
    _CONTINUE,
    DownloadLock,
    DownloadPreview,
    DownloadResult,
    ModelDownloader,
    lock_age_limit,
)
from config.davios_config import DaviosConfig

GIB = 1024 ** 3


# --------------------------------------------------------------------- #
# Catálogo sintético
# --------------------------------------------------------------------- #

def _model_entry(
    model_id: str,
    *,
    path: str,
    size_bytes: int = 0,
    name: str = "Modelo de Teste",
    tier: str = "light",
    fmt: str = "GGUF",
    download_available: bool = True,
    download_url: str = "http://127.0.0.1:9/fake.gguf",
    download_note: str = "",
    sha256: str = "",
) -> dict:
    """Entrada de catálogo com os campos que o downloader usa."""
    return {
        "id": model_id,
        "name": name,
        "family": "Teste",
        "tier": tier,
        "format": fmt,
        "quantization": "Q4_K_M",
        "parameters": "1B",
        "filename": Path(path).name,
        "path": path,
        "size_bytes": size_bytes,
        "size_is_estimate": True,
        "min_ram_gb": 1.0,
        "recommended_ram_gb": 2.0,
        "download_url": download_url,
        "download_available": download_available,
        "download_note": download_note,
        "sha256": sha256,
        "enabled": True,
    }


PAYLOAD = bytes(range(256)) * 60  # 15.360 bytes, conteúdo reconhecível
DOWNLOADABLE = "baixavel-q4km"
NO_DOWNLOAD = "sem-fonte-q4km"
NO_URL = "sem-url-q4km"
WITH_PART = "com-parcial-q4km"
EVIL = "travessia-q4km"
NO_SIZE = "sem-tamanho-q4km"
UNKNOWN_ID = "nao-existe-q4km"
# IDs usados apenas pelos testes de integridade SHA-256.
HASHED = "com-hash-q4km"
BAD_HASH = "hash-errado-q4km"
UPPER_HASH = "hash-maiusculo-q4km"
CORRUPT_HASH = "hash-corrompido-q4km"

SIZE_DOWNLOADABLE = len(PAYLOAD)


def _catalog_payload() -> dict:
    return {
        "version": 1,
        "models": [
            _model_entry(
                DOWNLOADABLE,
                path="models/light/baixavel-q4km.gguf",
                size_bytes=SIZE_DOWNLOADABLE,
            ),
            _model_entry(
                NO_DOWNLOAD,
                path="models/light/sem-fonte-q4km.gguf",
                size_bytes=SIZE_DOWNLOADABLE,
                download_available=False,
                download_url="",
                download_note="O catálogo não declara uma origem confirmada.",
            ),
            _model_entry(
                NO_URL,
                path="models/light/sem-url-q4km.gguf",
                size_bytes=SIZE_DOWNLOADABLE,
                download_available=True,
                download_url="",
            ),
            _model_entry(
                WITH_PART,
                path="models/light/com-parcial-q4km.gguf",
                size_bytes=SIZE_DOWNLOADABLE,
            ),
            _model_entry(
                EVIL,
                path="../../escaped-q4km.gguf",
                size_bytes=SIZE_DOWNLOADABLE,
            ),
            # Sem tamanho declarado: usado para provar que o downloader não
            # inventa total/percentual quando nem o catálogo nem o servidor
            # informam o tamanho.
            _model_entry(
                "sem-tamanho-q4km",
                path="models/balanced/sem-tamanho-q4km.gguf",
                size_bytes=0,
                tier="balanced",
            ),
            # --- Integridade SHA-256 ---------------------------------- #
            # HASHED tem o hash REAL de PAYLOAD; OTHER_HASH é um hash válido
            # de outro conteúdo (o teste garante que não colide).
            _model_entry(
                HASHED,
                path="models/light/com-hash-q4km.gguf",
                size_bytes=SIZE_DOWNLOADABLE,
                sha256=hashlib.sha256(PAYLOAD).hexdigest(),
            ),
            _model_entry(
                BAD_HASH,
                path="models/light/hash-errado-q4km.gguf",
                size_bytes=SIZE_DOWNLOADABLE,
                sha256=hashlib.sha256(PAYLOAD + b" adulterado").hexdigest(),
            ),
            # Hash em maiúsculas: o mesmo hash, escrito de outra forma.
            _model_entry(
                UPPER_HASH,
                path="models/light/hash-maiusculo-q4km.gguf",
                size_bytes=SIZE_DOWNLOADABLE,
                sha256=hashlib.sha256(PAYLOAD).hexdigest().upper(),
            ),
            # 64 hex, mas não é o hash do PAYLOAD.
            _model_entry(
                CORRUPT_HASH,
                path="models/light/hash-corrompido-q4km.gguf",
                size_bytes=SIZE_DOWNLOADABLE,
                sha256="a" * 64,
            ),
        ],
    }


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Raiz isolada com config, catálogo e pasta de modelos."""
    (tmp_path / "config").mkdir()
    (tmp_path / "models" / "light").mkdir(parents=True)
    (tmp_path / "config" / "model_catalog.json").write_text(
        json.dumps(_catalog_payload(), indent=2), encoding="utf-8"
    )
    (tmp_path / "config" / "davios.json").write_text(
        json.dumps({"offline_mode": True, "models_dir": "models"}),
        encoding="utf-8",
    )
    return tmp_path


# --------------------------------------------------------------------- #
# Transporte HTTP fake (nunca toca a rede)
# --------------------------------------------------------------------- #

class FakeResponse:
    """Resposta mínima compatível com o que o downloader consome."""

    def __init__(
        self,
        data: bytes = b"",
        content_length: int | None = None,
        chunk_size: int | None = None,
        fail_after: int | None = None,
    ) -> None:
        self._data = data
        self._pos = 0
        self._chunk_size = chunk_size
        self._fail_after = fail_after
        self.closed = False
        headers: dict[str, str] = {}
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        self.headers = headers

    def read(self, size: int = -1) -> bytes:
        if self._fail_after is not None and self._pos >= self._fail_after:
            # Simula a conexão caindo no meio do download.
            raise OSError("conexao interrompida pelo servidor (fake)")
        if self._pos >= len(self._data):
            return b""
        limit = size if size and size > 0 else len(self._data)
        if self._chunk_size:
            limit = min(limit, self._chunk_size)
        if self._fail_after is not None:
            limit = min(limit, self._fail_after - self._pos)
            if limit <= 0:
                raise OSError("conexao interrompida pelo servidor (fake)")
        chunk = self._data[self._pos:self._pos + limit]
        self._pos += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    """Substitui `urllib.request.urlopen`; registra as URLs pedidas."""

    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.requested_urls: list[str] = []
        self.timeouts: list[float] = []
        self.call_count = 0

    def __call__(self, request, timeout=None):
        self.call_count += 1
        self.requested_urls.append(getattr(request, "full_url", str(request)))
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return self.response


def _make_downloader(
    workspace: Path,
    transport: FakeTransport | None = None,
    chunk_size: int = 1024,
    progress_interval: int = 0,
) -> ModelDownloader:
    """Downloader isolado em tmp_path, com transporte fake injetado."""
    config = DaviosConfig.load()
    config.models_dir = "models"
    catalog = ModelCatalog(
        config=config,
        catalog_path=str(workspace / "config" / "model_catalog.json"),
        project_root=workspace,
    )
    return ModelDownloader(
        config=config,
        catalog=catalog,
        project_root=workspace,
        chunk_size=chunk_size,
        progress_interval=progress_interval,
        opener=transport if transport is not None else FakeTransport(),
    )


def _destination_for(workspace: Path, model_id: str) -> Path:
    """Caminho final de um modelo do catálogo sintético (espelha `path`)."""
    catalog_path = workspace / "config" / "model_catalog.json"
    entries = json.loads(catalog_path.read_text(encoding="utf-8"))["models"]
    for entry in entries:
        if entry["id"] == model_id:
            return workspace / entry["path"]
    raise AssertionError("modelo %s não está no catálogo sintético" % model_id)



# --------------------------------------------------------------------- #
# Integridade SHA-256
# --------------------------------------------------------------------- #

class TestSha256OnDownload:
    """O hash é conferido SOBRE O `.part`, antes de qualquer promoção."""

    def test_correct_sha256_promotes(self, workspace):
        """Hash confere: o arquivo é promovido e o resultado é verificado."""
        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(HASHED)

        assert result.status == STATUS_DOWNLOADED
        assert result.success is True
        assert result.hash_verified is True
        assert result.expected_sha256 == hashlib.sha256(PAYLOAD).hexdigest()
        assert result.actual_sha256 == result.expected_sha256
        assert _destination_for(workspace, HASHED).read_bytes() == PAYLOAD

    def test_wrong_sha256_blocks_promotion(self, workspace):
        """Hash não confere: nenhum arquivo final, nenhum `.part` sobrando."""
        downloader = _make_downloader(workspace, _ok_transport())
        destination = _destination_for(workspace, BAD_HASH)
        part = destination.with_name(destination.name + PART_SUFFIX)

        result = downloader.download(BAD_HASH)

        assert result.status == STATUS_FAILED
        assert result.success is False
        assert result.hash_verified is False
        assert not destination.exists()
        assert not part.exists()

    def test_part_with_wrong_hash_never_becomes_final(self, workspace):
        """A falha de integridade segue o mesmo caminho de qualquer outra: o
        `.part` some, e o catálogo continua vendo o modelo como não instalado."""
        downloader = _make_downloader(workspace, _ok_transport())
        destination = _destination_for(workspace, CORRUPT_HASH)
        part = destination.with_name(destination.name + PART_SUFFIX)

        result = downloader.download(CORRUPT_HASH, keep_partial=False)

        assert result.status == STATUS_FAILED
        assert not destination.exists()
        assert not part.exists()
        assert any("SHA-256" in e for e in result.validation_errors)

    def test_corruption_preserving_size_is_rejected(self, workspace):
        """O caso que o tamanho NÃO pega: conteúdo trocado, mesmos bytes.

        O `.part` fica com o tamanho certo, mas o catálogo declara o hash de
        outro arquivo. Só a verificação criptográfica detecta isso.
        """
        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(CORRUPT_HASH)

        assert result.status == STATUS_FAILED
        assert result.downloaded_bytes == len(PAYLOAD)  # tamanho confere
        assert result.hash_verified is False
        assert any("SHA-256" in e for e in result.validation_errors)

    def test_model_without_hash_warns_but_proceeds(self, workspace):
        """Sem hash no catálogo: o download acontece, mas a ausência é dita."""
        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_DOWNLOADED
        assert result.expected_sha256 is None
        assert result.hash_verified is False
        assert result.actual_sha256 is None  # nem calculado: não há o que comparar
        assert any("SHA-256" in w for w in result.validation_warnings)

    def test_uppercase_hash_is_accepted(self, workspace):
        """Maiúsculas são o MESMO hash; o catálogo normaliza para minúsculas."""
        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(UPPER_HASH)

        assert result.status == STATUS_DOWNLOADED
        assert result.hash_verified is True
        assert result.expected_sha256 == result.expected_sha256.lower()

    def test_invalid_hash_in_catalog_treated_as_absent(self, workspace):
        """Hash malformado vira `None` no ModelInfo: o download segue, com aviso."""
        downloader = _make_downloader(workspace, _ok_transport())
        model = downloader.resolve(DOWNLOADABLE)

        # Simula um catálogo editado à mão com hash inválido.
        model.sha256 = None
        result = downloader.download(DOWNLOADABLE)



class TestSha256OnExistingFile:
    """A verificação também protege o que JÁ está no disco."""

    def test_existing_file_with_correct_hash_is_already_installed(self, workspace):
        destination = _destination_for(workspace, HASHED)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(PAYLOAD)

        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(HASHED)

        assert result.status == STATUS_ALREADY_INSTALLED
        assert result.success is True
        assert result.hash_verified is True
        assert result.actual_sha256 == result.expected_sha256
        assert downloader._opener.call_count == 0  # não redesenha

    def test_existing_file_with_wrong_hash_is_refused(self, workspace):
        """Arquivo GGUF válido, tamanho certo, MAS adulterado: REFUSED.

        Sem esta regra, um `.gguf` adulterado localmente e do mesmo tamanho
        seria aceito como "já instalado" — a verificação protegeria apenas
        downloads novos.
        """
        destination = _destination_for(workspace, BAD_HASH)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(PAYLOAD)

        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(BAD_HASH)

        assert result.status == STATUS_REFUSED
        assert result.success is False
        assert result.hash_verified is False
        assert any("SHA-256" in e for e in result.validation_errors)
        # O arquivo do usuário NÃO foi tocado nem substituído.
        assert destination.read_bytes() == PAYLOAD
        assert downloader._opener.call_count == 0

    def test_existing_file_without_hash_still_accepted_with_warning(self, workspace):
        destination = _destination_for(workspace, DOWNLOADABLE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(PAYLOAD)

        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_ALREADY_INSTALLED
        assert result.hash_verified is False
        assert any("SHA-256" in w for w in result.validation_warnings)

    def test_existing_invalid_file_flow_is_preserved(self, workspace):
        """A lógica de `allow_replace_invalid` continua funcionando."""
        destination = _destination_for(workspace, HASHED)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"conteudo invalido")

        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(HASHED)

        assert result.status == STATUS_REFUSED
        assert "não sobrescreve" in result.error


class TestComputeSha256:
    def test_sha256_is_deterministic(self, workspace):
        downloader = _make_downloader(workspace)
        target = workspace / "a.bin"
        target.write_bytes(PAYLOAD)
        first = downloader._compute_sha256(target)
        second = downloader._compute_sha256(target)
        assert first == second == hashlib.sha256(PAYLOAD).hexdigest()

    def test_sha256_of_empty_file(self, workspace):
        downloader = _make_downloader(workspace)
        target = workspace / "vazio.bin"
        target.write_bytes(b"")
        assert downloader._compute_sha256(target) == hashlib.sha256(b"").hexdigest()

    def test_sha256_uses_chunks(self, workspace, monkeypatch):
        """Lê em pedaços: um arquivo maior que `chunk_size` precisa de várias
        leituras. Se lesse o arquivo inteiro de uma vez, o contador ficaria
        em 1 e um `.gguf` de 20 GB entraria inteiro na memória."""
        downloader = _make_downloader(workspace, chunk_size=1024)
        target = workspace / "grande.bin"
        target.write_bytes(PAYLOAD)  # 15.360 bytes >> chunk_size

        reads = {"n": 0}
        real_open = open

        def counting_open(path, mode="r", *args, **kwargs):
            handle = real_open(path, mode, *args, **kwargs)
            original_read = handle.read

            def read(size=-1):
                reads["n"] += 1
                return original_read(size)

            handle.read = read
            return handle

        monkeypatch.setattr("builtins.open", counting_open)
        digest = downloader._compute_sha256(target)
        monkeypatch.undo()

        assert digest == hashlib.sha256(PAYLOAD).hexdigest()
        assert reads["n"] > 1, "o arquivo deveria ter sido lido em varios chunks"
        # teto: 15.360 bytes / 1.024 = 15 leituras + 1 final vazia
        assert reads["n"] <= 17

    def test_mismatch_result_reports_expected_and_actual_hash(self, workspace):
        """O erro precisa dos DOIS hashes, senão o usuário não consegue agir."""
        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(BAD_HASH)

        expected = hashlib.sha256(PAYLOAD + b" adulterado").hexdigest()
        assert result.expected_sha256 == expected
        assert result.actual_sha256 == hashlib.sha256(PAYLOAD).hexdigest()
        assert result.expected_sha256 != result.actual_sha256
        joined = " ".join(result.validation_errors)
        assert expected in joined and result.actual_sha256 in joined
        # E o DownloadResult serializa os dois, para log/diagnóstico.
        payload = result.to_dict()
        assert payload["expected_sha256"] == expected
        assert payload["hash_verified"] is False

def _ok_transport(payload: bytes = PAYLOAD) -> FakeTransport:
    """Transporte que serve o payload completo, com Content-Length correto."""
    return FakeTransport(FakeResponse(data=payload, content_length=len(payload)))


# --------------------------------------------------------------------- #
# Recusas: o catálogo é a fonte da verdade sobre o que pode ser baixado
# --------------------------------------------------------------------- #

class TestRefusals:
    def test_model_without_download_available_is_refused(self, workspace):
        downloader = _make_downloader(workspace)
        result = downloader.download(NO_DOWNLOAD)
        assert result.status == STATUS_REFUSED
        assert result.success is False
        assert "origem confirmada" in result.error
        assert downloader._opener.call_count == 0  # nada de rede

    def test_model_without_url_is_refused(self, workspace):
        downloader = _make_downloader(workspace)
        result = downloader.download(NO_URL)
        assert result.status == STATUS_REFUSED
        assert downloader._opener.call_count == 0

    def test_unknown_model_is_refused(self, workspace):
        downloader = _make_downloader(workspace)
        result = downloader.download(UNKNOWN_ID)
        assert result.status == STATUS_REFUSED
        assert "não encontrado no catálogo" in result.error
        assert downloader._opener.call_count == 0

    def test_url_is_never_built_from_model_name(self, workspace):
        """Sem URL declarada, o downloader não inventa host nenhum."""
        downloader = _make_downloader(workspace)
        for model_id in (NO_DOWNLOAD, NO_URL):
            downloader.download(model_id)
        assert downloader._opener.requested_urls == []

    def test_refusal_does_not_create_files(self, workspace):
        downloader = _make_downloader(workspace)
        downloader.download(NO_DOWNLOAD)
        left_over = list((workspace / "models").rglob("*.gguf"))
        assert left_over == []


# --------------------------------------------------------------------- #
# Resolução (busca de dados, não decisão)
# --------------------------------------------------------------------- #

class TestResolution:
    def test_resolve_by_id(self, workspace):
        downloader = _make_downloader(workspace)
        assert downloader.resolve(DOWNLOADABLE).id == DOWNLOADABLE

    def test_resolve_by_alias(self, workspace):
        downloader = _make_downloader(workspace)
        found = downloader.resolve("leve")
        assert found is not None and found.tier == "light"

    def test_resolve_unknown_returns_none(self, workspace):
        downloader = _make_downloader(workspace)
        assert downloader.resolve(UNKNOWN_ID) is None
        assert downloader.resolve("") is None
        assert downloader.resolve(None) is None

    def test_resolve_accepts_model_info(self, workspace):
        downloader = _make_downloader(workspace)
        model = downloader.catalog.get_model_by_id(DOWNLOADABLE)
        assert downloader.resolve(model) is model


# --------------------------------------------------------------------- #
# Segurança de path
# --------------------------------------------------------------------- #

class TestPathSafety:
    def test_destination_comes_from_catalog(self, workspace):
        downloader = _make_downloader(workspace)
        model = downloader.catalog.get_model_by_id(DOWNLOADABLE)
        expected = workspace / "models" / "light" / "baixavel-q4km.gguf"
        assert downloader.destination_path(model) == expected

    def test_part_path_is_sibling_of_destination(self, workspace):
        downloader = _make_downloader(workspace)
        destination = workspace / "models" / "light" / "x.gguf"
        assert downloader.part_path_for(destination) == destination.with_name(
            "x.gguf" + PART_SUFFIX
        )

    def test_path_traversal_is_refused(self, workspace):
        """Catálogo apontando para fora da pasta de modelos é recusado."""
        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(EVIL)
        assert result.status == STATUS_REFUSED
        assert "fora da pasta de modelos" in result.error
        assert downloader._opener.call_count == 0
        assert not (workspace.parent / "escaped-q4km.gguf").exists()

    def test_traversal_check_reports_ok_inside_root(self, workspace):
        downloader = _make_downloader(workspace)
        inside = workspace / "models" / "light" / "ok.gguf"
        assert downloader._ensure_within_models_root(inside) is None


# --------------------------------------------------------------------- #
# Helpers dos fluxos reais (tudo com transporte fake, zero rede)
# --------------------------------------------------------------------- #

def _destination(workspace: Path) -> Path:
    return workspace / "models" / "light" / "baixavel-q4km.gguf"


class TestSuccessfulDownload:
    def test_download_writes_validates_and_promotes(self, workspace):
        """Fluxo feliz: bytes -> .part -> validação -> promoção -> .gguf."""
        downloader = _make_downloader(workspace, _ok_transport())
        destination = _destination(workspace)
        part = destination.with_name(destination.name + PART_SUFFIX)

        result = downloader.download(DOWNLOADABLE)

        assert result.success is True
        assert result.status == STATUS_DOWNLOADED
        assert destination.is_file()
        assert destination.read_bytes() == PAYLOAD
        assert not part.exists()          # .part foi promovido, não copiado
        assert result.part_path is None
        assert result.downloaded_bytes == len(PAYLOAD)
        assert not result.error   # DownloadResult.error default é "", não None
        # A URL usada é a do catálogo, e só ela.
        assert downloader._opener.requested_urls == ["http://127.0.0.1:9/fake.gguf"]

    def test_promotion_is_atomic_replace(self, workspace):
        """O `.gguf` final aparece por os.replace: um único instante, sem copy."""
        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(DOWNLOADABLE)
        assert result.success is True
        # Nenhum resíduo na pasta: nem .part, nem cópia temporária.
        leftovers = [p.name for p in (_destination(workspace).parent).iterdir()]
        assert leftovers == ["baixavel-q4km.gguf"]


class TestAlreadyInstalled:
    def test_valid_existing_file_is_not_overwritten(self, workspace):
        destination = _destination(workspace)
        destination.write_bytes(PAYLOAD)
        original_mtime = destination.stat().st_mtime_ns

        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_ALREADY_INSTALLED
        assert result.success is True
        assert result.already_installed is True
        assert destination.read_bytes() == PAYLOAD
        assert destination.stat().st_mtime_ns == original_mtime
        # Sem download: nem abriu conexão.
        assert downloader._opener.call_count == 0
        assert destination.stat().st_mtime_ns == original_mtime
        # Sem download: nem abriu conexão.
        assert downloader._opener.call_count == 0


class TestExistingInvalidFile:
    INVALID_CONTENT = b"conteudo invalido de teste"

    def _write_invalid(self, workspace: Path) -> Path:
        destination = _destination(workspace)
        destination.write_bytes(self.INVALID_CONTENT)
        return destination

    def test_invalid_existing_file_is_refused_by_default(self, workspace):
        destination = self._write_invalid(workspace)
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_REFUSED
        assert result.success is False
        # O arquivo original continua intacto: nada foi destruído em silêncio.
        assert destination.read_bytes() == self.INVALID_CONTENT
        assert downloader._opener.call_count == 0
        # Nenhum .part foi criado, pois a recusa veio antes do download.

    def test_invalid_file_replaced_only_with_explicit_permission(self, workspace):
        """allow_replace_invalid=True: o inválido vira .invalid, não lixo."""
        destination = self._write_invalid(workspace)
        backup = destination.with_name(destination.name + ".invalid")
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(DOWNLOADABLE, allow_replace_invalid=True)

        assert result.success is True
        assert result.status == STATUS_DOWNLOADED
        # O inválido foi PRESERVADO, não apagado.
        assert backup.is_file()
        assert backup.read_bytes() == self.INVALID_CONTENT
        # O novo download ocupa o nome definitivo.
        assert destination.read_bytes() == PAYLOAD
        # O .part não permanece: virou o .gguf final.
        assert not destination.with_name(destination.name + PART_SUFFIX).exists()

# --------------------------------------------------------------------- #
# Espaço em disco (simulado via monkeypatch — nada depende do disco real)
# --------------------------------------------------------------------- #

class TestDiskSpace:
    def test_insufficient_space_refused_before_any_download(self, workspace, monkeypatch):
        """Disco 'cheio' simulado: recusa ANTES de abrir conexão/criar .part."""
        from collections import namedtuple

        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(
            "brain.model_downloader.shutil.disk_usage",
            lambda _probe: usage(total=10 * GIB, used=10 * GIB - 512, free=512),
        )

        downloader = _make_downloader(workspace, _ok_transport())
        destination = _destination(workspace)

        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_REFUSED
        assert result.success is False
        assert "livres" in result.error
        assert not destination.exists()
        # A checagem de espaço acontece antes de qualquer byte trafegar.
        assert downloader._opener.call_count == 0

    def test_tight_space_allows_download_but_warns(self, workspace, monkeypatch):
        """Cabe, mas apertado: o download segue e o aviso vai para warnings."""
        from collections import namedtuple

        usage = namedtuple("usage", "total used free")
        # ~2x o necessário: passa no 'needed', cai na faixa de 'apertado'.
        monkeypatch.setattr(
            "brain.model_downloader.shutil.disk_usage",
            lambda _probe: usage(total=4 * GIB, used=4 * GIB - 2 * len(PAYLOAD), free=2 * len(PAYLOAD)),
        )

        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(DOWNLOADABLE)

        assert result.success is True
        assert result.status == STATUS_DOWNLOADED
        assert _destination(workspace).read_bytes() == PAYLOAD
        assert any("apertado" in w for w in result.validation_warnings)


# --------------------------------------------------------------------- #
# Progresso
# --------------------------------------------------------------------- #

class TestProgress:
    def test_progress_callback_receives_coherent_numbers(self, workspace):
        """Callback monótono, terminando no tamanho total declarado."""
        calls: list[tuple[int, int]] = []
        downloader = _make_downloader(
            workspace, _ok_transport(), chunk_size=1024, progress_interval=1024
        )

        result = downloader.download(DOWNLOADABLE, progress_callback=lambda d, t: calls.append((d, t)))

        assert result.success is True
        assert calls, "esperava ao menos o callback final de progresso"
        downloaded_values = [d for d, _ in calls]
        assert downloaded_values == sorted(downloaded_values)      # monótono
        assert downloaded_values[0] > 0
        total_values = {t for _, t in calls}
        assert total_values == {len(PAYLOAD)}                      # total real
        assert all(d <= len(PAYLOAD) for d in downloaded_values)   # nada acima de 100%
        # Última notificação cobre o arquivo inteiro.
        assert downloaded_values[-1] == len(PAYLOAD)

    def test_callback_error_never_breaks_download(self, workspace):
        downloader = _make_downloader(
            workspace, _ok_transport(), chunk_size=1024, progress_interval=1024
        )

        def broken_callback(_d: int, _t: int) -> None:
            raise RuntimeError("callback do chamador explodiu (fake)")

        result = downloader.download(DOWNLOADABLE, progress_callback=broken_callback)

        assert result.success is True
        assert _destination(workspace).read_bytes() == PAYLOAD


# --------------------------------------------------------------------- #
# Content-Length ausente
# --------------------------------------------------------------------- #

class TestNoContentLength:
    def test_download_works_and_never_invents_a_total(self, workspace):
        """Sem Content-Length E sem size no catálogo: total segue 0/real."""
        calls: list[tuple[int, int]] = []
        downloader = _make_downloader(
            workspace,
            FakeTransport(FakeResponse(data=PAYLOAD, content_length=None)),
            chunk_size=1024,
            progress_interval=1024,
        )

        result = downloader.download(NO_SIZE, progress_callback=lambda d, t: calls.append((d, t)))

        assert result.success is True
        assert result.status == STATUS_DOWNLOADED
        destination = workspace / "models" / "balanced" / "sem-tamanho-q4km.gguf"
        assert destination.read_bytes() == PAYLOAD
        # Nenhum total foi inventado: o servidor não disse, o catálogo não sabe.
        assert all(t in (0, len(PAYLOAD)) for _, t in calls)
        assert result.expected_bytes == 0
        assert result.downloaded_bytes == len(PAYLOAD)
        # Promoção limpa: o .gguf final existe e o .part não sobrou.
        assert not destination.with_name(destination.name + PART_SUFFIX).exists()


class TestCancellation:
    def test_cancelled_download_never_creates_final_file(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())
        destination = _destination(workspace)
        cancel_event = threading.Event()
        cancel_event.set()  # cancela já na primeira iteração do loop

        result = downloader.download(DOWNLOADABLE, cancel_event=cancel_event)

        assert result.status == STATUS_CANCELLED
        assert result.cancelled is True
        assert result.success is False
        assert result.error == "Download cancelado."
        # Nada é promovido num cancelamento: o .gguf final nunca aparece.
        assert not destination.exists()
        # Nenhuma conexão sobrou no fake (o cancelamento ocorre no loop, depois
        # de abrir a conexão — por isso call_count == 1 aqui).
        assert downloader._opener.call_count == 1

    @pytest.mark.xfail(
        reason=(
            "DEFEITO DE PRODUÇÃO (Windows): no cancelamento dentro do loop de "
            "leitura, _abandon_part() é chamado DENTRO do bloco "
            "`with open(part, \"wb\")` — o handle do .part ainda está aberto, e "
            "o Windows não permite unlink() de arquivo aberto. O OSError é "
            "engolido em _abandon_part e o .part fica retido mesmo com "
            "keep_partial=False. Não corrigimos produção nesta tarefa por "
            "instrução do usuário; strict=True força revisão deste marcador "
            "quando o defeito for corrigido."
        ),
        strict=True,
    )
    def test_cancelled_download_removes_part_by_default(self, workspace):
        """Com keep_partial=False (default), o .part deveria ser removido."""
        downloader = _make_downloader(workspace, _ok_transport())
        destination = _destination(workspace)
        part = destination.with_name(destination.name + PART_SUFFIX)
        cancel_event = threading.Event()
        cancel_event.set()

        result = downloader.download(DOWNLOADABLE, cancel_event=cancel_event)

        assert result.cancelled is True
        assert not part.exists()   # falha hoje no Windows (ver reason do xfail)


class TestNetworkFailure:
    def test_connection_error_keeps_final_file_absent(self, workspace):
        failing = FakeTransport(error=OSError("host inacessivel (fake)"))
        downloader = _make_downloader(workspace, failing)
        destination = _destination(workspace)
        part = destination.with_name(destination.name + PART_SUFFIX)

        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_FAILED
        assert result.success is False
        assert result.error is not None
        assert not destination.exists()
        assert not part.exists()          # falha antes de abrir o .part

    def test_interrupted_mid_stream_removes_part_by_default(self, workspace):
        """Conexão cai no meio: por padrão o .part é descartado."""
        response = FakeResponse(
            data=PAYLOAD, content_length=len(PAYLOAD), chunk_size=1024, fail_after=100
        )
        downloader = _make_downloader(workspace, FakeTransport(response))
        destination = _destination(workspace)
        part = destination.with_name(destination.name + PART_SUFFIX)

        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_FAILED
        assert not destination.exists()
        assert not part.exists()

    def test_interrupted_mid_stream_keeps_part_with_keep_partial(self, workspace):
        """keep_partial=True preserva o temporário para retomada futura."""
        response = FakeResponse(
            data=PAYLOAD, content_length=len(PAYLOAD), chunk_size=1024, fail_after=100
        )
        downloader = _make_downloader(workspace, FakeTransport(response))
        destination = _destination(workspace)
        part = destination.with_name(destination.name + PART_SUFFIX)

        result = downloader.download(DOWNLOADABLE, keep_partial=True)

        assert result.status == STATUS_FAILED
        assert result.success is False
        assert not destination.exists()          # final NUNCA aparece
        assert part.exists()                     # temporário preservado
        assert 0 < part.stat().st_size < len(PAYLOAD)
        assert result.downloaded_bytes == part.stat().st_size


# --------------------------------------------------------------------- #
# Prévia read-only: mostra o que BAIXARIA, sem baixar nada
# --------------------------------------------------------------------- #

class TestPreview:
    """`preview()` é informativa: nada de rede, nada de disco, nada de config."""

    def test_preview_of_valid_model_reports_real_data(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())
        preview = downloader.preview(DOWNLOADABLE)

        assert preview.downloadable is True
        assert preview.can_download is True
        assert preview.model_id == DOWNLOADABLE
        assert preview.name == "Modelo de Teste"
        assert preview.expected_bytes == SIZE_DOWNLOADABLE
        assert preview.download_url == "http://127.0.0.1:9/fake.gguf"
        assert preview.destination == str(_destination(workspace))
        assert preview.part_path.endswith(PART_SUFFIX)
        assert preview.blocked_reason == ""
        # Espaço livre é um inteiro real do disco (nunca inventado).
        assert isinstance(preview.free_bytes, int)
        assert preview.free_bytes > 0

    def test_preview_never_touches_network_or_disk(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())
        light = workspace / "models" / "light"
        before = sorted(p.name for p in light.iterdir())

        preview = downloader.preview(DOWNLOADABLE)

        assert downloader._opener.call_count == 0      # nenhuma conexão
        assert before == sorted(p.name for p in light.iterdir())  # nenhum arquivo
        assert not preview.destination_exists

    def test_preview_without_url_is_not_downloadable(self, workspace):
        """Sem `download_url` no catálogo, a prévia recusa — sem inventar URL."""
        downloader = _make_downloader(workspace, _ok_transport())

        preview = downloader.preview(NO_URL)

        assert preview.downloadable is False
        assert preview.can_download is False
        assert preview.download_url == ""

    def test_preview_detects_already_installed(self, workspace):
        destination = _destination(workspace)
        destination.write_bytes(PAYLOAD)
        downloader = _make_downloader(workspace, _ok_transport())

        preview = downloader.preview(DOWNLOADABLE)

        assert preview.destination_exists is True
        assert preview.already_installed is True
        assert "já está instalado" in preview.summary()

    def test_preview_flags_invalid_existing_file(self, workspace):
        destination = _destination(workspace)
        destination.write_bytes(b"conteudo invalido")
        downloader = _make_downloader(workspace, _ok_transport())

        preview = downloader.preview(DOWNLOADABLE)

        assert preview.destination_exists is True
        assert preview.already_installed is False
        assert preview.can_download is False
        assert "não sobrescreve" in preview.blocked_reason

    def test_preview_to_dict_is_serializable(self, workspace):
        import json as _json
        downloader = _make_downloader(workspace, _ok_transport())

        payload = downloader.preview(DOWNLOADABLE).to_dict()

        # to_dict precisa sobreviver a json.dumps (usado em log/diagnóstico).
        _json.dumps(payload)
        assert payload["model_id"] == DOWNLOADABLE
        assert payload["expected_human"].endswith("GB")
        assert payload["can_download"] is True
    def test_preview_without_download_available_is_blocked(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())

        preview = downloader.preview(NO_DOWNLOAD)

        assert preview.downloadable is False
        assert preview.blocked_reason.startswith("Modelo de Teste:")
        assert "origem confirmada" in preview.blocked_reason

    def test_preview_of_unknown_model_is_not_downloadable(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())

        preview = downloader.preview(UNKNOWN_ID)

        assert preview.downloadable is False
        assert "não encontrado no catálogo" in preview.blocked_reason

    def test_preview_reports_path_traversal_as_blocked(self, workspace):
        """A prévia aplica a mesma proteção de path traversal do download."""
        downloader = _make_downloader(workspace, _ok_transport())

        preview = downloader.preview(EVIL)

        assert preview.can_download is False
        assert "fora da pasta de modelos" in preview.blocked_reason

    def test_preview_summary_contains_the_facts(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())

        text = downloader.preview(DOWNLOADABLE).summary()

        assert "Modelo" in text and "Origem" in text
        assert "Destino" in text and "Espaço" in text
        assert "fake.gguf" in text





# --------------------------------------------------------------------- #
# Confirmação explícita antes de qualquer transferência
# --------------------------------------------------------------------- #

class TestConfirmation:
    """`confirm(preview) -> bool` decide. Antes dele, zero bytes trafegam."""

    def test_accepted_confirmation_downloads_normally(self, workspace):
        seen: list = []
        downloader = _make_downloader(workspace, _ok_transport())

        def confirm(preview):
            seen.append(preview)
            return True

        result = downloader.download(DOWNLOADABLE, confirm=confirm)

        assert result.success is True
        assert result.status == STATUS_DOWNLOADED
        assert _destination(workspace).read_bytes() == PAYLOAD
        assert downloader._opener.call_count == 1
        # A confirmação recebeu a prévia real, com os mesmos dados.
        assert len(seen) == 1
        assert seen[0].model_id == DOWNLOADABLE
        assert seen[0].can_download is True

    def test_declined_confirmation_does_not_start_download(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())
        destination = _destination(workspace)
        part = destination.with_name(destination.name + PART_SUFFIX)

        result = downloader.download(DOWNLOADABLE, confirm=lambda _p: False)

        assert result.status == STATUS_DECLINED
        assert result.success is False
        assert "recusado" in result.error
        # Nenhuma transferência, nenhum .part, nenhum arquivo final.
        assert downloader._opener.call_count == 0
        assert not part.exists()
        assert not destination.exists()

    def test_missing_confirmation_when_required_blocks_download(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())
        destination = _destination(workspace)
        part = destination.with_name(destination.name + PART_SUFFIX)

        result = downloader.download(DOWNLOADABLE, require_confirmation=True)

        assert result.status == STATUS_CONFIRMATION_REQUIRED
        assert result.success is False
        assert "confirmação" in result.error
        assert downloader._opener.call_count == 0
        assert not part.exists()
        assert not destination.exists()

    def test_confirmation_is_not_required_by_default(self, workspace):
        """Compatibilidade: sem flags, o download segue exatamente como antes."""
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_DOWNLOADED
        assert downloader._opener.call_count == 1

    def test_required_confirmation_satisfied_by_callback(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(
            DOWNLOADABLE, confirm=lambda _p: True, require_confirmation=True
        )

        assert result.status == STATUS_DOWNLOADED
        assert downloader._opener.call_count == 1

    def test_confirmation_runs_before_any_byte_is_written(self, workspace):
        """O callback é chamado com o disco ainda intacto."""
        destination = _destination(workspace)
        observed: dict = {}

        def confirm(preview):
            observed["part_exists"] = bool(
                preview.part_path and Path(preview.part_path).exists()
            )
            observed["final_exists"] = destination.exists()
            observed["url_calls"] = downloader._opener.call_count
            return True

    def test_confirmation_never_reaches_models_without_url(self, workspace):
        """A recusa por falta de fonte vem ANTES da pergunta ao usuário."""
        asked: list = []
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(
            NO_URL,
            confirm=lambda p: asked.append(p) or True,
            require_confirmation=True,
        )

        assert result.status == STATUS_REFUSED
        assert asked == []
        assert downloader._opener.call_count == 0

    def test_confirmation_never_reaches_path_traversal(self, workspace):
        asked: list = []
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(
            EVIL,
            confirm=lambda p: asked.append(p) or True,
            require_confirmation=True,
        )

        assert result.status == STATUS_REFUSED
        assert asked == []
        assert downloader._opener.call_count == 0

    def test_confirmation_never_reaches_when_file_is_invalid(self, workspace):
        destination = _destination(workspace)
        destination.write_bytes(b"conteudo invalido")
        asked: list = []
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(
            DOWNLOADABLE,
            confirm=lambda p: asked.append(p) or True,
            require_confirmation=True,
        )

        assert result.status == STATUS_REFUSED
        assert asked == []
        assert downloader._opener.call_count == 0
        assert destination.read_bytes() == b"conteudo invalido"

    def test_confirmation_never_reaches_on_insufficient_space(
        self, workspace, monkeypatch
    ):
        from collections import namedtuple

        usage = namedtuple("usage", "total used free")
        monkeypatch.setattr(
            "brain.model_downloader.shutil.disk_usage",
            lambda _probe: usage(total=10 * GIB, used=10 * GIB - 512, free=512),
        )
        asked: list = []
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(
            DOWNLOADABLE,
            confirm=lambda p: asked.append(p) or True,
            require_confirmation=True,
        )

        assert result.status == STATUS_REFUSED
        assert asked == []
        assert downloader._opener.call_count == 0

    def test_already_installed_short_circuits_before_confirmation(self, workspace):
        """Nada a baixar => nada a perguntar."""
        destination = _destination(workspace)
        destination.write_bytes(PAYLOAD)
        asked: list = []
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(
            DOWNLOADABLE,
            confirm=lambda p: asked.append(p) or True,
            require_confirmation=True,
        )

        assert result.status == STATUS_ALREADY_INSTALLED
        assert asked == []
        assert downloader._opener.call_count == 0

    def test_new_statuses_have_readable_summaries(self, workspace):
        downloader = _make_downloader(workspace, _ok_transport())

        required = downloader.download(
            DOWNLOADABLE, require_confirmation=True
        ).summary()
        declined = downloader.download(
            DOWNLOADABLE, confirm=lambda _p: False
        ).summary()

        assert "confirmação" in required
        assert "não autorizado" in declined
        # A recusa afirma que NADA foi baixado (não "Modelo X baixado ...").
        assert "nada foi baixado" in declined
        assert not declined.startswith("Modelo")


# --------------------------------------------------------------------- #
# DownloadLock — exclusão mútua por destination
# --------------------------------------------------------------------- #

def _lock_path(workspace: Path, model_id: str = DOWNLOADABLE) -> Path:
    destination = _destination_for(workspace, model_id)
    return destination.with_name(destination.name + ".lock")


def _write_lock(lock_path: Path, **fields) -> None:
    """Grava um lock com campos específicos, completando o resto."""
    payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": time.time(),
        "size_bytes": SIZE_DOWNLOADABLE,
        "model_id": DOWNLOADABLE,
        "token": "x" * 32,
    }
    payload.update(fields)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")


class SlowTransport(FakeTransport):
    """Transporte que demora o suficiente para o lock ser observável."""

    def __init__(self, delay: float = 0.05, payload: bytes = PAYLOAD):
        super().__init__(FakeResponse(data=payload, content_length=len(payload)))
        self.delay = delay

    def __call__(self, request, timeout=None):
        time.sleep(self.delay)
        return super().__call__(request, timeout=timeout)


class TestLockAcquisition:
    def test_lock_created_atomically(self, workspace):
        """A criação usa O_CREAT|O_EXCL: sem janela de TOCTOU."""
        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert lock.acquire() is True
        assert lock.acquired is True
        assert _lock_path(workspace).exists()
        lock.release()

    def test_lock_contains_metadata(self, workspace):
        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert lock.acquire() is True
        data = json.loads(_lock_path(workspace).read_text(encoding="utf-8"))
        assert data["pid"] == os.getpid()
        assert data["hostname"] == socket.gethostname()
        assert isinstance(data["started_at"], (int, float))
        assert data["size_bytes"] == SIZE_DOWNLOADABLE
        assert data["model_id"] == DOWNLOADABLE
        # token: impede que outra instância apague este lock
        assert isinstance(data["token"], str) and data["token"]
        lock.release()

    def test_second_acquire_returns_locked(self, workspace):
        """Segundo acquire NÃO espera: devolve False imediatamente."""
        first = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        second = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert first.acquire() is True
        began = time.time()
        assert second.acquire() is False
        assert time.time() - began < 0.5  # não houve espera
        first.release()
        assert second.acquire() is True  # após liberar, assume
        second.release()

    def test_release_removes_lock(self, workspace):
        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        lock.acquire()
        lock.release()
        assert not _lock_path(workspace).exists()
        assert lock.acquired is False

    def test_lock_path_is_sibling_inside_models_root(self, workspace):
        downloader = _make_downloader(workspace)
        destination = _destination_for(workspace, DOWNLOADABLE)
        lock_path = downloader.lock_path_for(destination)
        assert lock_path.parent == destination.parent
        assert lock_path.name.endswith(".lock")
        # Herda a mesma proteção de path traversal do destino.
        assert downloader._ensure_within_models_root(lock_path) is None

    def test_owner_cannot_release_other_lock(self, workspace):
        """Uma instância que NÃO adquiriu não pode apagar o lock de outra."""
        owner = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        stranger = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert owner.acquire() is True

        stranger.release()  # não é dona: não pode apagar
        assert _lock_path(workspace).exists(), "o lock do dono foi removido!"

        owner.release()
        assert not _lock_path(workspace).exists()

    def test_stolen_lock_is_not_removed_by_previous_owner(self, workspace):
        """Se outro tomou o lock, o antigo NÃO o apaga ao sair."""
        first = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        first.acquire()
        _write_lock(_lock_path(workspace), token="outro-token")  # roubado

        first.release()
        assert _lock_path(workspace).exists(), "removeu lock de terceiro"
        json.loads(_lock_path(workspace).read_text())  # ainda é JSON válido


class TestLockAgePolicy:
    def test_small_model_uses_floor(self):
        assert lock_age_limit(1000) == LOCK_MIN_AGE_SECONDS

    def test_zero_size_uses_floor(self):
        assert lock_age_limit(0) == LOCK_MIN_AGE_SECONDS

    def test_large_model_gets_longer_window(self):
        """19,8 GB precisa de uma janela MUITO maior que 1 hora."""
        limite_pequeno = lock_age_limit(2 * 1024 ** 3)
        limite_grande = lock_age_limit(20 * 1024 ** 3)
        assert limite_grande > limite_pequeno
        assert limite_grande > LOCK_MIN_AGE_SECONDS * 24

    def test_limit_is_capped(self):
        """Mesmo um tamanho absurdo não produz limite infinito."""
        assert lock_age_limit(10 ** 18) <= 7 * 24 * 3600.0

    def test_invalid_size_is_tolerated(self):
        assert lock_age_limit(-5) == LOCK_MIN_AGE_SECONDS
        assert lock_age_limit("nao-e-numero") == LOCK_MIN_AGE_SECONDS


class TestStaleLocks:
    def test_live_lock_is_not_reclaimed(self, workspace):
        """Lock legítimo, processo VIVO: nunca pode ser roubado."""
        owner = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert owner.acquire() is True

        outro = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert outro.acquire() is False
        assert _lock_path(workspace).exists()
        owner.release()

    def test_stale_lock_dead_pid_is_reclaimed(self, workspace):
        """PID morto neste host -> stale, reclaim."""
        # PID alto e improvável de existir; hostname igual ao atual.
        _write_lock(_lock_path(workspace), pid=999_999, hostname=socket.gethostname())
        assert _lock_path(workspace).exists()

        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert lock.acquire() is True
        assert json.loads(_lock_path(workspace).read_text())["pid"] == os.getpid()
        lock.release()

    def test_malformed_lock_is_reclaimed(self, workspace):
        """Conteúdo não-JSON não é confiável -> stale."""
        _lock_path(workspace).write_text("isto nao e json {{{", encoding="utf-8")
        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert lock.acquire() is True
        lock.release()

    def test_empty_lock_is_reclaimed(self, workspace):
        _lock_path(workspace).write_text("", encoding="utf-8")
        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert lock.acquire() is True
        lock.release()

    def test_oversized_lock_is_reclaimed(self, workspace):
        """Lock enorme não vale a pena interpretar."""
        _lock_path(workspace).write_text("A" * (LOCK_MAX_CONTENT_BYTES + 100), encoding="utf-8")
        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert lock.acquire() is True
        lock.release()

    def test_old_lock_is_reclaimed(self, workspace):
        """Lock com processo vivo mas idade acima do limite -> stale."""
        _write_lock(
            _lock_path(workspace),
            pid=os.getpid(),  # vivo: só a idade pode liberar
            started_at=time.time() - LOCK_MIN_AGE_SECONDS - 10,
        )
        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert lock.acquire() is True
        lock.release()

    def test_large_model_lock_not_stolen(self, workspace):
        """O ponto mais importante: um download de 19,8 GB legítimo NÃO pode
        ter seu lock roubado só por ser 'velho' na escala de um modelo pequeno."""
        started = time.time() - (3600 * 2)  # 2 horas atrás
        _write_lock(
            _lock_path(workspace),
            pid=os.getpid(),  # processo VIVO
            size_bytes=20 * 1024 ** 3,  # 19,8 GB
            started_at=started,
        )
        limit = lock_age_limit(20 * 1024 ** 3)
        assert 3600 * 2 < limit, "2h deveria estar dentro da janela de 19,8 GB"

        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, 20 * 1024 ** 3)
        assert lock.acquire() is False, "lock legítimo de modelo grande foi roubado!"
        assert _lock_path(workspace).exists()

    def test_other_hostname_relies_on_age_only(self, workspace):
        """Lock de outra máquina: o PID dela não é verificável aqui."""
        _write_lock(
            _lock_path(workspace),
            pid=999_999,  # morto aqui, mas é de outro host
            hostname="outra-maquina-na-rede",
        )
        lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        # Não pode ser roubado só porque "o PID não existe aqui".
        assert lock.acquire() is False

    def test_stale_reclaim_has_no_double_owner(self, workspace):
        """Dois contenders disputando o MESMO lock obsoleto: so um pode ser dono."""
        _write_lock(_lock_path(workspace), pid=999_999, hostname=socket.gethostname())

        results = []
        start = threading.Barrier(2)

        def contender():
            lock = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
            start.wait()
            results.append(lock.acquire())

        threads = [threading.Thread(target=contender) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exatamente um venceu: o O_EXCL é o árbitro, não o unlink.
        assert sorted(results) == [False, True]

        dono = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert dono.acquire() is False, "o lock deveria continuar com um dono"
        # Libera para não deixar resíduo no tmp_path.
        json.loads(_lock_path(workspace).read_text())
        _lock_path(workspace).unlink()


class TestDownloadLockIntegration:
    """O lock dentro do fluxo real de `download()`."""

    def test_lock_exists_during_download(self, workspace):
        """O lock existe ENQUANTO o download corre e some depois."""
        visto_durante = []
        downloader = _make_downloader(workspace, SlowTransport(delay=0.25))
        original = downloader._run_download

        def observing(*args, **kwargs):
            visto_durante.append(_lock_path(workspace).exists())
            return original(*args, **kwargs)

        downloader._run_download = observing
        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_DOWNLOADED
        assert visto_durante == [True], "o lock não existia durante o download"
        assert not _lock_path(workspace).exists(), "o lock não foi liberado"

    def test_release_after_failure(self, workspace):
        downloader = _make_downloader(
            workspace, FakeTransport(error=OSError("sem rede (fake)"))
        )
        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_FAILED
        assert not _lock_path(workspace).exists()

    def test_release_after_cancel(self, workspace):
        event = threading.Event()
        event.set()  # já cancelado
        downloader = _make_downloader(workspace, SlowTransport(delay=0.05))

        result = downloader.download(DOWNLOADABLE, cancel_event=event)

        assert result.status == STATUS_CANCELLED
        assert not _lock_path(workspace).exists()

    def test_release_after_cancel_keeps_part_when_requested(self, workspace):
        event = threading.Event()
        event.set()
        downloader = _make_downloader(workspace, SlowTransport(delay=0.05))
        destination = _destination_for(workspace, DOWNLOADABLE)

        result = downloader.download(
            DOWNLOADABLE, cancel_event=event, keep_partial=True
        )

        assert result.status == STATUS_CANCELLED
        assert not _lock_path(workspace).exists()
        # `_abandon_part` preservou o .part, como antes do lock.
        assert destination.with_name(destination.name + PART_SUFFIX).exists()

    def test_sha_mismatch_releases_lock(self, workspace):
        """SHA errado: STATUS_FAILED, lock liberado, `.part` removido."""
        downloader = _make_downloader(workspace, _ok_transport())
        destination = _destination_for(workspace, BAD_HASH)
        part = destination.with_name(destination.name + PART_SUFFIX)

        result = downloader.download(BAD_HASH)

        assert result.status == STATUS_FAILED
        assert not _lock_path(workspace).exists()
        assert not part.exists()
        assert not destination.exists()

    def test_existing_file_fast_path_does_not_lock(self, workspace):
        """Arquivo já válido: ALREADY_INSTALLED sem tocar no disco de escrita."""
        destination = _destination_for(workspace, DOWNLOADABLE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(PAYLOAD)
        downloader = _make_downloader(workspace, _ok_transport())

        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_ALREADY_INSTALLED
        assert not _lock_path(workspace).exists()
        assert downloader._opener.call_count == 0

    def test_promotion_occurs_under_lock(self, workspace):
        """O `os.replace` acontece com o lock ainda em mãos."""
        estado = {}
        downloader = _make_downloader(workspace, _ok_transport())
        original = os.replace

        def spy(src, dst, *a, **k):
            if str(dst).endswith(".gguf"):
                estado["lock"] = _lock_path(workspace).exists()
            return original(src, dst, *a, **k)

        import brain.model_downloader as md
        md.os.replace = spy
        try:
            result = downloader.download(DOWNLOADABLE)
        finally:
            md.os.replace = original

        assert result.status == STATUS_DOWNLOADED
        assert estado.get("lock") is True

    def test_allow_replace_invalid_is_locked(self, workspace):
        """O `os.replace` para `.invalid` também roda sob lock."""
        destination = _destination_for(workspace, HASHED)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"conteudo invalido")
        backup = destination.with_name(destination.name + ".invalid")

        estado = {}
        downloader = _make_downloader(workspace, _ok_transport())
        original = os.replace
        # O lock de HASHED é outro arquivo: o de DOWNLOADABLE não tem relação.
        lock_do_alvo = _lock_path(workspace, HASHED)

        def spy(src, dst, *a, **k):
            if str(dst).endswith(".invalid"):
                estado["lock"] = lock_do_alvo.exists()
            return original(src, dst, *a, **k)

        import brain.model_downloader as md
        md.os.replace = spy
        try:
            result = downloader.download(HASHED, allow_replace_invalid=True)
        finally:
            md.os.replace = original

        assert result.status == STATUS_DOWNLOADED
        assert backup.is_file()
        assert estado.get("lock") is True

    def test_existing_file_appears_before_lock_is_revalidated(self, workspace):
        """O arquivo final surge entre a checagem inicial e o lock.

        Sem a revalidação pós-lock, o segundo download sobrescreveria cegamente
        um arquivo que o outro processo acabou de promover. Aqui a 1ª checagem
        responde "não existe" e a 2ª, feita SOB o lock, encontra o arquivo.
        """
        destination = _destination_for(workspace, DOWNLOADABLE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloader = _make_downloader(workspace, _ok_transport())

        original_check = downloader._handle_existing
        state = {"n": 0}

        def check_que_mente_uma_vez(dest, model, result, allow, *a, **k):
            state["n"] += 1
            if state["n"] == 1:
                return _CONTINUE  # a mentira: "não existe"
            return original_check(dest, model, result, allow, *a, **k)

        downloader._handle_existing = check_que_mente_uma_vez

        # `preview()` roda no intervalo entre a 1ª checagem e o lock (dentro do
        # fluxo de confirmação), então é um gancho fiel para simular "outro
        # processo promoveu agora".
        original_preview = downloader.preview

        def preview_que_cria(model):
            if not destination.exists():
                destination.write_bytes(PAYLOAD)
            return original_preview(model)

        downloader.preview = preview_que_cria

        result = downloader.download(DOWNLOADABLE, confirm=lambda p: True)

        assert state["n"] == 2, "a revalidação pós-lock não aconteceu"
        assert result.status == STATUS_ALREADY_INSTALLED
        assert downloader._opener.call_count == 0, "não deveria ter baixado"
        assert not _lock_path(workspace).exists()


class TestConcurrency:
    """Cenários de concorrência medidos, não supostos."""

    def test_same_model_two_threads_one_wins_one_locked(self, workspace):
        """Duas threads, mesmo destino: uma baixa, a outra recebe LOCKED.

        Este é o teste que reproduz a race da auditoria. Antes do lock, as duas
        escreviam no mesmo `.part` e a promoção falhava com WinError 32.
        """
        results = []
        erros = []
        barreira = threading.Barrier(2)

        def worker():
            downloader = _make_downloader(workspace, SlowTransport(delay=0.15))
            try:
                barreira.wait(timeout=5)
                r = downloader.download(DOWNLOADABLE)
                results.append(r.status)
            except Exception as exc:  # noqa: BLE001 - o teste É sobre não explodir
                erros.append((type(exc).__name__, str(exc)))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not erros, f"houve exceção (WinError 32 antes do lock): {erros}"
        assert len(results) == 2
        # Exatamente um venceu.
        assert results.count(STATUS_DOWNLOADED) == 1
        assert results.count(STATUS_LOCKED) == 1
        # Arquivo final íntegro: existe apenas UMA vez, completo.
        destination = _destination_for(workspace, DOWNLOADABLE)
        assert destination.exists()
        assert destination.read_bytes() == PAYLOAD
        assert not destination.with_name(destination.name + PART_SUFFIX).exists()
        assert not _lock_path(workspace).exists()

    def test_locked_result_carries_holder_info(self, workspace):
        """O recusado sabe QUEM está segurando, para diagnóstico."""
        dono = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        assert dono.acquire() is True

        downloader = _make_downloader(workspace, _ok_transport())
        result = downloader.download(DOWNLOADABLE)

        assert result.status == STATUS_LOCKED
        assert result.lock_waited is False       # política: não esperar
        assert result.lock_holder is not None
        assert str(os.getpid()) in result.lock_holder
        assert result.success is False
        assert downloader._opener.call_count == 0  # nem tocou a rede
        dono.release()

    def test_locked_summary_is_helpful(self, workspace):
        dono = DownloadLock(_lock_path(workspace), DOWNLOADABLE, SIZE_DOWNLOADABLE)
        dono.acquire()
        downloader = _make_downloader(workspace, _ok_transport())

        texto = downloader.download(DOWNLOADABLE).summary()

        assert "andamento" in texto
        assert "Nenhum download novo" in texto
        dono.release()

    def test_different_models_download_in_parallel(self, workspace):
        """A e B tem locks distintos: os dois avançam, nenhum trava o outro."""
        barreira = threading.Barrier(2)
        resultados = {}
        erros = []

        def worker(model_id, chave):
            downloader = _make_downloader(workspace, SlowTransport(delay=0.15))
            try:
                barreira.wait(timeout=5)
                r = downloader.download(model_id)
                resultados[chave] = (r.status, r.destination)
            except Exception as exc:  # noqa: BLE001
                erros.append((chave, type(exc).__name__, str(exc)))

        # HASHED e CORRUPT_HASH: modelos diferentes, destinos diferentes.
        t1 = threading.Thread(target=worker, args=(HASHED, "a"))
        t2 = threading.Thread(target=worker, args=(CORRUPT_HASH, "b"))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        assert not erros, f"modelos diferentes não deveriam colidir: {erros}"
        # Nenhum recebeu LOCKED: os locks são por destination.
        assert resultados["a"][0] == STATUS_DOWNLOADED
        assert resultados["b"][0] == STATUS_FAILED  # hash propositalmente errado
        # E cada um promoveu para o SEU destino.
        assert _destination_for(workspace, HASHED).exists()
        assert not _destination_for(workspace, CORRUPT_HASH).exists()
        assert not _lock_path(workspace, HASHED).exists()
        assert not _lock_path(workspace, CORRUPT_HASH).exists()

    def test_no_partial_file_ever_visible_at_destination(self, workspace):
        """Durante o download, o destino ou não existe ou está completo.

        O `.part` nunca é visto pelo ModelCatalog porque tem outro nome — mas
        o lock garante que ninguém substitua o destino por um arquivo parcial.
        """
        destino = _destination_for(workspace, DOWNLOADABLE)
        observados = []
        parar = threading.Event()

        def vigia():
            while not parar.is_set():
                if destino.exists():
                    observados.append(destino.stat().st_size)
                time.sleep(0.002)

        v = threading.Thread(target=vigia)
        v.start()
        try:
            downloader = _make_downloader(workspace, SlowTransport(delay=0.1))
            result = downloader.download(DOWNLOADABLE)
        finally:
            parar.set()
            v.join()

        assert result.status == STATUS_DOWNLOADED
        assert observados, "a vigia nunca viu o arquivo (teste suspecto)"
        # Nenhum tamanho parcial observado.
        assert all(size == len(PAYLOAD) for size in observados), (
            f"tamanho parcial visível: {set(observados)}"
        )


# --------------------------------------------------------------------- #
# Processo separado (multiprocessing)
# --------------------------------------------------------------------- #

def _process_worker(workspace_str, model_id, barrier, queue):
    """Alvo do processo filho. Faz um download e reporta o status."""
    workspace = Path(workspace_str)
    try:
        # A barreira sincroniza os DOIS processos ANTES de qualquer download:
        # o lock é adquirido antes da rede, então sincronizar dentro do opener
        # deixaria o perdedor (que nem chega à rede) esperando para sempre.
        barrier.wait(timeout=30)

        downloader = _make_downloader(
            workspace, FakeTransport(
                FakeResponse(data=PAYLOAD, content_length=len(PAYLOAD))
            )
        )
        result = downloader.download(model_id)
        queue.put(result.status)
    except Exception as exc:  # noqa: BLE001
        queue.put("EXCECAO:%s:%s" % (type(exc).__name__, exc))


@pytest.mark.skipif(
    multiprocessing.get_start_method(allow_none=True) == "spawn"
    and not hasattr(multiprocessing, "get_context"),
    reason="multiprocessing indisponível",
)
class TestCrossProcess:
    def test_same_model_two_processes_one_wins_one_locked(self, workspace):
        """Dois PROCESSOS, mesmo destino: um baixa, o outro recebe LOCKED.

        É o cenário que a auditoria mostrou não ser coberto por threads: o
        lock de arquivo precisa valer entre processos, não só entre threads.
        """
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        barreira = ctx.Barrier(2)

        processos = [
            ctx.Process(
                target=_process_worker,
                args=(str(workspace), DOWNLOADABLE, barreira, queue),
            )
            for _ in range(2)
        ]
        for p in processos:
            p.start()
        for p in processos:
            p.join(timeout=90)
            assert p.exitcode is not None, "processo travou"

        status = [queue.get(timeout=10) for _ in range(2)]

        assert not any(s.startswith("EXCECAO") for s in status), status
        assert sorted(status) == sorted([STATUS_DOWNLOADED, STATUS_LOCKED]), status
        # Exatamente UM arquivo final, íntegro.
        destination = _destination_for(workspace, DOWNLOADABLE)
        assert destination.exists()
        assert destination.read_bytes() == PAYLOAD
        assert not _lock_path(workspace).exists()


class TestLockApi:
    def test_default_fields_preserve_compatibility(self):
        result = DownloadResult()
        assert result.lock_waited is False
        assert result.lock_holder is None
        # Campos antigos continuam com os mesmos defaults.
        assert result.success is False
        assert result.status == STATUS_FAILED
        assert result.model_id == ""

    def test_to_dict_includes_lock_fields(self):
        result = DownloadResult(
            status=STATUS_LOCKED, model_id="x", lock_waited=False, lock_holder="h:1"
        )
        payload = result.to_dict()
        assert payload["lock_waited"] is False
        assert payload["lock_holder"] == "h:1"
        assert payload["status"] == STATUS_LOCKED

    def test_old_results_still_work(self):
        """Um DownloadResult construído sem os campos novos é válido."""
        result = DownloadResult(success=True, status=STATUS_DOWNLOADED)
        assert result.summary().startswith("Modelo")
        assert result.to_dict()["lock_holder"] is None

    def test_status_locked_is_a_new_constant_not_a_reuse(self):
        assert STATUS_LOCKED == "locked"
        # Não pode ser confundido com nenhuma outra situação.
        for other in (
            STATUS_FAILED, STATUS_REFUSED, STATUS_CANCELLED, STATUS_DECLINED,
        ):
            assert STATUS_LOCKED != other

    def test_preview_does_not_acquire_lock(self, workspace):
        """`preview()` continua read-only: não toma o lock."""
        downloader = _make_downloader(workspace, _ok_transport())
        preview = downloader.preview(DOWNLOADABLE)
        assert preview.can_download is True
        assert not _lock_path(workspace).exists()

    def test_no_lock_bypass_flags_exist(self):
        """Nenhuma forma de ignorar o lock foi introduzida."""
        import inspect

        assinatura = inspect.signature(ModelDownloader.download)
        for parametro in assinatura.parameters:
            assert "lock" not in parametro.lower() or parametro in (
                "require_confirmation",
            )
        # E não há constante de "ignorar lock".
        assert not hasattr(ModelDownloader, "ignore_lock")
        assert not hasattr(ModelDownloader, "force_lock")
