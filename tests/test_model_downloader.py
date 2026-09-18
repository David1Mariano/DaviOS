"""Testes do ModelDownloader (camada isolada de rede/disco).

Regras desta suíte:
- NENHUM download real e NENHUM acesso à internet: o transporte HTTP é um
  fake injetado em `opener`, que serve bytes de memória e simula erros;
- catálogo, configuração e arquivos vivem em `tmp_path` (isolamento total);
- NÃO se inicia llama-server, NÃO se altera `active_model_id` e NÃO se chama
  o provider — há uma classe de testes dedicada a provar isso.
"""

from __future__ import annotations

import json
import threading
import urllib.error
from pathlib import Path

import pytest

from brain.model_catalog import ModelCatalog
from brain.model_downloader import (
    PART_SUFFIX,
    STATUS_ALREADY_INSTALLED,
    STATUS_CANCELLED,
    STATUS_DOWNLOADED,
    STATUS_FAILED,
    STATUS_REFUSED,
    DownloadResult,
    ModelDownloader,
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