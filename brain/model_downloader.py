"""Downloader de modelos do DaviOS — camada ISOLADA de rede e de disco.

Responsabilidade única: pegar um modelo do catálogo, baixar o arquivo com
streaming, validar e promover o temporário para o nome definitivo.

    ModelCatalog   ->  o que existe e de onde vem (fonte da verdade)
    ModelManager   ->  qual modelo usar (decisão)
    ModelDownloader -> baixar + validar + finalizar arquivo   <-- este módulo
    LocalLlamaCppProvider -> executar llama-server

O que este módulo NÃO faz (por contrato):
- NÃO decide qual modelo o usuário deve usar;
- NÃO altera `active_model_id` nem chama `set_active_model()`;
- NÃO inicia, para ou conversa com o llama-server / provider;
- NÃO edita o catálogo em memória nem marca modelos como instalados.

Quem descobre que um arquivo está instalado é o `ModelCatalog`
(`is_model_installed()`), olhando o disco — nunca este módulo.

Regras de segurança:
1. Só baixa o que o CATÁLOGO autoriza: `download_available=True` E
   `download_url` não vazia. Jamais monta URL a partir do nome do modelo.
2. Escreve em `<arquivo>.part` e só renomeia para `.gguf` após validar.
   Um `.part` nunca é promovido por interrupção, erro ou falha de validação.
3. Nunca sobrescreve silenciosamente um `.gguf` existente.
4. O destino é derivado do catálogo, resolvido e confinado à pasta de
   modelos do projeto (proteção contra path traversal).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from brain.model_catalog import PROJECT_ROOT, ModelCatalog, ModelInfo
from config.davios_config import DaviosConfig

logger = logging.getLogger("davios.models.downloader")

# Sufixos. O temporário fica FORA do caminho final de propósito: assim o
# ModelCatalog não o enxerga como modelo instalado.
PART_SUFFIX = ".part"
GGUF_SUFFIX = ".gguf"

# Tamanho do bloco de leitura da resposta HTTP (streaming).
CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Identificação enviada ao servidor. Alguns hosts recusam requisições sem UA.
USER_AGENT = "DaviOS-ModelDownloader/1.0"

# Intervalo entre callbacks de progresso, em bytes recebidos. Evita inundar
# o chamador com um callback por chunk em arquivos de vários GB.
PROGRESS_INTERVAL_BYTES = 8 * 1024 * 1024  # 8 MiB

# Fração mínima do tamanho DECLARADO no catálogo que um arquivo precisa ter
# para ser considerado completo. O catálogo usa estimativas, então exigir
# igualdade exata rejeitaria arquivos bons; 0.80 detecta download truncado
# sem transformar estimativa em requisito absoluto.
MIN_SIZE_RATIO = 0.80

# Fator de segurança do disco: cobre o `.part` escrito em paralelo e
# fragmentação. Não é margem de "quase cabe" — serve só para recusar
# downloads que claramente não cabem.
DISK_SAFETY_FACTOR = 1.05

# Status possíveis de um DownloadResult.
STATUS_DOWNLOADED = "downloaded"                 # baixou agora e promoveu
STATUS_ALREADY_INSTALLED = "already_installed"   # já havia arquivo válido
STATUS_REFUSED = "refused"                       # catálogo não autoriza
STATUS_FAILED = "failed"                         # erro de rede/validação/disco
STATUS_CANCELLED = "cancelled"                   # abortado pelo chamador

# Sentinela interno: "nada a decidir aqui, siga o fluxo normal do download".
# Usamos um objeto único (e comparação por identidade) para não confundir
# "nenhum erro" com um DownloadResult válido.
_CONTINUE = object()

ProgressCallback = Callable[[int, int], None]


def _human_gb(num_bytes: Optional[int]) -> str:
    """Formata bytes como GB legível; '?' quando não há valor."""
    if num_bytes is None or num_bytes <= 0:
        return "?"
    return f"{num_bytes / (1024 ** 3):.2f} GB"


@dataclass
class DownloadResult:
    """Resultado padronizado de uma tentativa de download.

    `success=True` significa que existe um `.gguf` válido no destino final —
    seja porque acabou de ser baixado, seja porque já estava lá. O chamador
    não precisa inspecionar o disco para saber se pode usar o modelo.
    """

    success: bool = False
    status: str = STATUS_FAILED
    model_id: str = ""
    destination: Optional[str] = None
    part_path: Optional[str] = None
    downloaded_bytes: int = 0
    expected_bytes: int = 0
    already_installed: bool = False
    cancelled: bool = False
    error: str = ""
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    # True quando `expected_bytes` veio de estimativa do catálogo (e não do
    # Content-Length do servidor). Permite ao chamador explicar a diferença.
    expected_is_estimate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "model_id": self.model_id,
            "destination": self.destination,
            "part_path": self.part_path,
            "downloaded_bytes": self.downloaded_bytes,
            "expected_bytes": self.expected_bytes,
            "expected_is_estimate": self.expected_is_estimate,
            "already_installed": self.already_installed,
            "cancelled": self.cancelled,
            "error": self.error,
            "validation_errors": list(self.validation_errors),
            "validation_warnings": list(self.validation_warnings),
        }

    def summary(self) -> str:
        """Mensagem curta, em português, para mostrar ao usuário."""
        if self.status == STATUS_ALREADY_INSTALLED:
            return f"O modelo {self.model_id} já está instalado em {self.destination}."
        if self.status == STATUS_DOWNLOADED:
            return (
                f"Modelo {self.model_id} baixado ({_human_gb(self.downloaded_bytes)}) "
                f"para {self.destination}."
            )
        if self.status == STATUS_CANCELLED:
            return f"Download de {self.model_id} cancelado."
        if self.status == STATUS_REFUSED:
            return self.error or f"Download de {self.model_id} não está disponível."
        return self.error or f"Falha ao baixar {self.model_id}."


class ModelDownloader:
    """Baixa, valida e finaliza arquivos de modelo autorizados pelo catálogo."""

    def __init__(
        self,
        config: Optional[DaviosConfig] = None,
        catalog: Optional[ModelCatalog] = None,
        project_root: Optional[Path] = None,
        chunk_size: int = CHUNK_SIZE,
        timeout: float = 30.0,
        progress_interval: int = PROGRESS_INTERVAL_BYTES,
        opener: Optional[Callable[..., Any]] = None,
    ):
        self.config = config or DaviosConfig.load()
        # `is not None` (e não `or`): um catálogo vazio é falsy por causa de
        # __len__ e seria substituído pelo catálogo do projeto em silêncio.
        self.catalog = catalog if catalog is not None else ModelCatalog(config=self.config)
        self.project_root = (
            Path(project_root).resolve() if project_root else Path(PROJECT_ROOT)
        )
        self.chunk_size = max(1024, int(chunk_size))
        self.timeout = float(timeout)
        self.progress_interval = max(0, int(progress_interval))
        # `opener` permite injetar um transporte fake nos testes sem tocar em
        # rede. Default: urllib.request.urlopen.
        self._opener = opener or urllib.request.urlopen

    # ------------------------------------------------------------------ #
    # Destino e segurança de path
    # ------------------------------------------------------------------ #

    def models_root(self) -> Path:
        """Diretório raiz onde os modelos do DaviOS podem ser gravados."""
        root = self.config.models_path(self.project_root)
        return Path(root).resolve()

    def _ensure_within_models_root(self, path: Path) -> Optional[str]:
        """Confina um destino à pasta de modelos. Devolve erro, ou None se ok.

        Proteção contra path traversal: mesmo que o catálogo (ou um catálogo
        editado à mão) declare algo como "../../etc/passwd", o destino nunca
        escapa de `models_root()`.

        Exceção deliberada: uma configuração explícita de `models_dir` em
        formato absoluto é tratada como raiz confiável — o usuário pode
        legitimamente guardar modelos em outro volume.
        """
        root = self.models_root()
        try:
            candidate = path.resolve()
        except OSError as exc:
            return f"Caminho de destino inválido: {exc}"
        if os.environ.get("DAVIOS_ALLOW_MODELS_OUTSIDE_ROOT") == "1":
            return None
        if Path(self.config.models_dir).is_absolute():
            return None
        if root == candidate or root in candidate.parents:
            return None
        return (
            f"Destino fora da pasta de modelos ({root}): {candidate}. "
            "O DaviOS não grava modelos fora dessa pasta."
        )

    def destination_path(self, model: ModelInfo) -> Path:
        """Caminho final do `.gguf`, derivado do catálogo (nunca do usuário)."""
        if model.path:
            path = Path(model.path)
            return path if path.is_absolute() else self.project_root / path
        filename = model.filename or f"{model.id}{GGUF_SUFFIX}"
        return self.models_root() / filename

    def part_path_for(self, destination: Path) -> Path:
        """Caminho do arquivo temporário: `<destino>.part`.

        Fica FORA do nome final para que o catálogo nunca o considere um
        modelo instalado enquanto o download não terminar.
        """
        return destination.with_name(destination.name + PART_SUFFIX)

    # ------------------------------------------------------------------ #
    # Validação
    # ------------------------------------------------------------------ #

    def validate_file(
        self,
        path: Path,
        model: ModelInfo,
    ) -> tuple[list[str], list[str]]:
        """Validação mínima de um arquivo de modelo (erros, avisos).

        Erros impedem a promoção do `.part`; avisos não. O tamanho declarado
        no catálogo é uma ESTIMATIVA, então é usado apenas para detectar
        arquivo OBVIAMENTE incompleto (proporção mínima), nunca para exigir
        igualdade exata.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not path.exists():
            return [f"Arquivo não encontrado: {path}"], warnings

        if not path.is_file():
            return [f"O destino não é um arquivo regular: {path}"], warnings

        try:
            size = path.stat().st_size
        except OSError as exc:
            return [f"Não foi possível medir o arquivo: {exc}"], warnings

        if size <= 0:
            errors.append("O arquivo está vazio (0 bytes).")
            return errors, warnings

        # Extensão esperada. O catálogo declara o formato; aceitamos o sufixo
        # do formato, não o nome exato do arquivo. Ao validar um `.part`,
        # comparamos com o sufixo do nome FINAL (removendo o `.part`), senão
        # todo temporário seria reprovado na sua própria extensão.
        expected_suffix = f".{model.format.lower()}" if model.format else GGUF_SUFFIX
        if expected_suffix != PART_SUFFIX:
            candidate = path.name
            if candidate.lower().endswith(PART_SUFFIX):
                candidate = candidate[: -len(PART_SUFFIX)]
            actual_suffix = Path(candidate).suffix.lower()
            if actual_suffix != expected_suffix:
                errors.append(
                    f"Extensão inesperada ({actual_suffix or 'sem extensão'}); "
                    f"era esperado {expected_suffix}."
                )

        # Tamanho plausível. Só quando o catálogo declara um tamanho.
        expected = int(model.size_bytes or 0)
        if expected > 0:
            if size < expected * MIN_SIZE_RATIO:
                errors.append(
                    f"Arquivo incompleto: {_human_gb(size)} baixados contra "
                    f"~{_human_gb(expected)} esperados."
                )
            elif size < expected:
                warnings.append(
                    f"Arquivo menor que o estimado ({_human_gb(size)} de "
                    f"~{_human_gb(expected)}); pode estar truncado."
                )
            elif size > expected * 1.5:
                warnings.append(
                    f"Arquivo bem maior que o estimado ({_human_gb(size)} de "
                    f"~{_human_gb(expected)})."
                )

        return errors, warnings

    # ------------------------------------------------------------------ #
    # Espaço em disco
    # ------------------------------------------------------------------ #

    def check_space(
        self,
        model: ModelInfo,
        target_dir: Optional[Path] = None,
    ) -> tuple[bool, str]:
        """Verifica se cabe antes de começar. (ok, mensagem)

        Nunca inicia um download que claramente não cabe. A margem cobre o
        arquivo temporário + fragmentação.
        """
        expected = int(model.size_bytes or 0)
        if expected <= 0:
            # Sem estimativa não há como afirmar; deixamos o próprio sistema
            # operacional recusar a escrita se o disco encher.
            return True, "Tamanho não declarado no catálogo; espaço não verificado."
        target = Path(target_dir) if target_dir else self.destination_path(model).parent
        probe = target if target.exists() else self.project_root
        try:
            usage = shutil.disk_usage(str(probe))
        except OSError as exc:
            return True, f"Não foi possível medir o disco livre ({exc}); prosseguindo."
        needed = int(expected * DISK_SAFETY_FACTOR)
        if usage.free < needed:
            return False, (
                f"Esse modelo ocupa aproximadamente {_human_gb(expected)} e há apenas "
                f"{_human_gb(usage.free)} livres em {probe}."
            )
        if usage.free < needed * 2:
            return True, (
                f"Cabe, mas o armazenamento está apertado: ~{_human_gb(expected)} "
                f"necessários e {_human_gb(usage.free)} livres."
            )
        return True, f"Espaço suficiente: {_human_gb(usage.free)} livres."

    # ------------------------------------------------------------------ #
    # Resolução (busca de dados, não decisão)
    # ------------------------------------------------------------------ #

    def resolve(self, model: Any) -> Optional[ModelInfo]:
        """Aceita ModelInfo, ID exato ou alias; consulta apenas o catálogo.

        Isto é busca de DADOS, não decisão: quem escolhe qual modelo usar é o
        ModelManager.
        """
        if model is None:
            return None
        if isinstance(model, ModelInfo):
            return model
        text = str(model).strip()
        if not text:
            return None
        found = self.catalog.get_model_by_id(text)
        return found if found is not None else self.catalog.resolve_alias(text)

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #

    def download(
        self,
        model: Any,
        progress_callback: Optional[ProgressCallback] = None,
        cancel_event: Optional[threading.Event] = None,
        keep_partial: bool = False,
        allow_replace_invalid: bool = False,
    ) -> DownloadResult:
        """Baixa o modelo para `<arquivo>.part` e promove para `.gguf`.

        Parâmetros:
        - progress_callback(baixados, total) — total=0 quando desconhecido;
        - cancel_event — qualquer objeto com `is_set()`; checado a cada chunk
          (base para um cancelamento futuro, sem UI agora);
        - keep_partial — se True, preserva o `.part` em falha/cancelamento
          para permitir retomada; default False (remove o temporário);
        - allow_replace_invalid — se True, move um `.gguf` existente e
          INVÁLIDO para `<arquivo>.gguf.invalid` antes de baixar de novo.
          Default False: nunca destruímos arquivo existente em silêncio.

        Nunca altera `active_model_id`, nunca fala com o provider e nunca
        edita o catálogo.
        """
        resolved = self.resolve(model)
        if resolved is None:
            return DownloadResult(
                status=STATUS_REFUSED,
                model_id=str(model),
                error=f"Modelo não encontrado no catálogo: {model!r}",
            )
        model = resolved
        result = DownloadResult(model_id=model.id)

        # --- 1. O catálogo autoriza este download? ---------------------- #
        if not model.download_available or not model.download_url:
            note = model.download_note or (
                "O catálogo não declara uma origem de download confirmada "
                "para este modelo."
            )
            logger.warning("[MODEL] download recusado (sem fonte): %s", model.id)
            result.status = STATUS_REFUSED
            result.error = f"{model.name}: {note}"
            return result

        # --- 2. Destino seguro ----------------------------------------- #
        destination = self.destination_path(model)
        result.destination = str(destination)
        safety_error = self._ensure_within_models_root(destination)
        if safety_error:
            logger.warning("[MODEL][ERROR] destino recusado: %s", safety_error)
            result.status = STATUS_REFUSED
            result.error = safety_error
            return result

        expected = int(model.size_bytes or 0)
        result.expected_bytes = expected

        # --- 3. Já existe um arquivo final? ---------------------------- #
        existing_error = self._handle_existing(
            destination, model, result, allow_replace_invalid
        )
        if existing_error is not _CONTINUE:
            return result

        # --- 4. Cabe no disco? ----------------------------------------- #
        fits, space_message = self.check_space(model, destination.parent)
        if not fits:
            result.status = STATUS_REFUSED
            result.error = space_message
            return result
        if "apertado" in space_message or "não verificado" in space_message:
            result.validation_warnings.append(space_message)

        part = self.part_path_for(destination)
        result.part_path = str(part)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            result.status = STATUS_FAILED
            result.error = f"Não foi possível criar a pasta de destino: {exc}"
            return result

        logger.info("[MODEL] download iniciado: %s -> %s", model.id, destination)
        return self._run_download(
            model, destination, part, result, progress_callback,
            cancel_event, keep_partial,
        )

    # ------------------------------------------------------------------ #
    # Fluxo interno
    # ------------------------------------------------------------------ #

    def _handle_existing(
        self,
        destination: Path,
        model: ModelInfo,
        result: DownloadResult,
        allow_replace_invalid: bool,
    ) -> Any:
        """O que fazer quando já existe algo no destino FINAL.

        Política (explícita e coberta por testes):
        - nada existe         -> _CONTINUE, segue o download;
        - existe e é válido   -> already_installed; NÃO sobrescreve;
        - existe e é inválido -> recusa explicando o motivo. Só move o arquivo
          para `<nome>.invalid` se `allow_replace_invalid=True`. Nunca
          destruímos um arquivo existente em silêncio.
        """
        if not destination.exists():
            return _CONTINUE

        errors, warnings = self.validate_file(destination, model)
        if not errors:
            result.status = STATUS_ALREADY_INSTALLED
            result.success = True
            result.already_installed = True
            result.validation_warnings.extend(warnings)
            try:
                result.downloaded_bytes = destination.stat().st_size
            except OSError:
                pass
            logger.info("[MODEL] já instalado, não sobrescrito: %s", model.id)
            return result

        result.validation_errors.extend(errors)
        if not allow_replace_invalid:
            result.status = STATUS_REFUSED
            result.error = (
                f"Já existe um arquivo em {destination}, mas ele parece inválido "
                f"({'; '.join(errors)}). O DaviOS não sobrescreve arquivos "
                "existentes automaticamente: remova ou mova o arquivo e tente "
                "de novo."
            )
            logger.warning("[MODEL] arquivo existente inválido: %s", destination)
            return result

        backup = destination.with_name(destination.name + ".invalid")
        try:
            os.replace(str(destination), str(backup))
        except OSError as exc:
            result.status = STATUS_FAILED
            result.error = f"Não foi possível mover o arquivo inválido: {exc}"
            return result
        result.validation_warnings.append(f"Arquivo inválido preservado em {backup}.")
        logger.warning("[MODEL] arquivo inválido movido para %s", backup)
        return _CONTINUE

    def _run_download(
        self,
        model: ModelInfo,
        destination: Path,
        part: Path,
        result: DownloadResult,
        progress_callback: Optional[ProgressCallback],
        cancel_event: Optional[threading.Event],
        keep_partial: bool,
    ) -> DownloadResult:
        """Streaming para `.part`, validação e promoção atômica para `.gguf`.

        O arquivo final só passa a existir depois da validação: em erro,
        cancelamento ou falha de validação o `.part` é removido (ou preservado
        quando `keep_partial=True`) e NUNCA é promovido.
        """
        expected = result.expected_bytes
        result.expected_is_estimate = bool(model.size_is_estimate)
        progress_total = expected
        downloaded = 0
        last_notified = 0
        response = None

        # --- Conexão --------------------------------------------------- #
        try:
            response = self._open_stream(model.download_url)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
            result.status = STATUS_FAILED
            result.error = f"Falha ao abrir a conexão: {exc}"
            logger.error("[MODEL][ERROR] falha ao iniciar download: %s", exc)
            return result

        try:
            declared_total = self._declared_total(response)
            if declared_total > 0:
                # O Content-Length vem do servidor; a estimativa do catálogo é
                # só um palpite. Para o progresso, o número do servidor manda.
                progress_total = declared_total
                result.expected_bytes = declared_total
                result.expected_is_estimate = False

            try:
                with open(part, "wb") as handle:
                    while True:
                        if cancel_event is not None and cancel_event.is_set():
                            result.status = STATUS_CANCELLED
                            result.cancelled = True
                            result.error = "Download cancelado."
                            logger.info("[MODEL] download cancelado: %s", model.id)
                            return self._abandon_part(part, result, keep_partial)

                        chunk = response.read(self.chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        result.downloaded_bytes = downloaded
                        if (
                            progress_callback is not None
                            and self.progress_interval > 0
                            and downloaded - last_notified >= self.progress_interval
                        ):
                            last_notified = downloaded
                            self._notify(progress_callback, downloaded, progress_total)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
                result.status = STATUS_FAILED
                result.error = f"Download interrompido: {exc}"
                logger.error("[MODEL][ERROR] download interrompido: %s", exc)
                return self._abandon_part(part, result, keep_partial)
        finally:
            self._close(response)

        if cancel_event is not None and cancel_event.is_set():
            # Cancelamento pedido no exato último chunk: ainda assim não
            # promovemos, para que quem cancelou não receba um modelo instalado.
            result.status = STATUS_CANCELLED
            result.cancelled = True
            result.error = "Download cancelado."
            return self._abandon_part(part, result, keep_partial)

        self._notify(progress_callback, downloaded, progress_total or downloaded)

        # --- Validação do temporário ------------------------------------ #
        # Nada é promovido antes de passar por aqui: tamanho, extensão e
        # plausibilidade. O `.gguf` só existe se for utilizável.
        errors, warnings = self.validate_file(part, model)
        result.validation_warnings.extend(warnings)
        if errors:
            result.status = STATUS_FAILED
            result.validation_errors.extend(errors)
            result.error = (
                f"O arquivo baixado não passou na validação: {'; '.join(errors)}"
            )
            logger.error("[MODEL][ERROR] validação falhou: %s", errors)
            return self._abandon_part(part, result, keep_partial)

        # --- Promoção atômica ------------------------------------------ #
        # `os.replace` no mesmo volume é atômico: ou o `.gguf` final aparece
        # completo, ou não aparece. Não existe estado intermediário visível
        # para o ModelCatalog.
        try:
            os.replace(str(part), str(destination))
        except OSError as exc:
            result.status = STATUS_FAILED
            result.error = f"Não foi possível finalizar o arquivo: {exc}"
            logger.error("[MODEL][ERROR] falha ao promover .part: %s", exc)
            return self._abandon_part(part, result, keep_partial)

        result.status = STATUS_DOWNLOADED
        result.success = True
        result.part_path = None  # já não existe mais
        try:
            result.downloaded_bytes = destination.stat().st_size
        except OSError:
            result.downloaded_bytes = downloaded
        logger.info(
            "[MODEL] download concluído: %s (%s)", model.id, _human_gb(result.downloaded_bytes)
        )
        return result

    # ------------------------------------------------------------------ #
    # Acesso à rede
    # ------------------------------------------------------------------ #
    # Ficam isoladas aqui para que os testes possam injetar um transporte
    # fake em `opener` (ver __init__) sem tocar em rede real.

    def _open_stream(self, url: str) -> Any:
        """Abre o stream binário do arquivo, com timeout e User-Agent.

        Alguns hosts recusam requisições sem User-Agent. O timeout evita um
        download que ficaria pendurado para sempre numa conexão morta.
        """
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/octet-stream, */*",
            },
        )
        return self._opener(request, timeout=self.timeout)

    def _declared_total(self, response: Any) -> int:
        """Content-Length declarado pelo servidor, ou 0 quando indisponível.

        Este é o único número de tamanho CONFIÁVEL: a estimativa do catálogo é
        um palpite de fórmula. Quando o servidor informa, ele prevalece.
        """
        headers = getattr(response, "headers", None)
        if headers is None:
            return 0
        getter = getattr(headers, "get", None)
        if getter is None:
            return 0
        raw = getter("Content-Length")
        if raw is None:
            return 0
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            return 0
        return value if value > 0 else 0

    def _close(self, response: Any) -> None:
        """Fecha o stream; falha de fechamento nunca derruba o fluxo."""
        if response is None:
            return
        close = getattr(response, "close", None)
        if close is None:
            return
        try:
            close()
        except Exception:  # noqa: BLE001 - fechar é melhor esforço
            logger.debug("[MODEL] falha ao fechar o stream HTTP", exc_info=True)

    def _notify(
        self,
        callback: Optional[ProgressCallback],
        downloaded: int,
        total: int,
    ) -> None:
        """Chama o callback de progresso; erro do chamador não vaza.

        `total=0` significa "tamanho total desconhecido" — não inventamos um
        denominador, o que faria o percentual mentir.
        """
        if callback is None:
            return
        try:
            callback(downloaded, total)
        except Exception:  # noqa: BLE001 - o callback é do chamador
            logger.debug("[MODEL] callback de progresso falhou", exc_info=True)

    def _abandon_part(
        self,
        part: Path,
        result: DownloadResult,
        keep_partial: bool,
    ) -> DownloadResult:
        """Política explícita do `.part` quando o download não se completa.

        - keep_partial=False (default): remove o temporário. Não deixa lixo e
          elimina qualquer chance de confusão com um modelo instalado.
        - keep_partial=True: preserva o `.part` para uma retomada futura. O
          arquivo FINAL continua inexistente e o `ModelCatalog` continua vendo
          "não instalado", porque `.part` nunca é promovido.

        Em nenhum caminho o `.part` vira `.gguf`: a promoção só existe no
        `os.replace` do fim de `_run_download`, depois da validação.
        """
        result.part_path = str(part)
        if not part.exists():
            return result
        if keep_partial:
            result.validation_warnings.append(
                f"Temporário preservado em {part} (keep_partial=True); "
                "o modelo continua NÃO instalado."
            )
            return result
        try:
            part.unlink()
            result.part_path = None
        except OSError as exc:
            result.validation_warnings.append(
                f"Não foi possível remover o temporário {part}: {exc}"
            )
        return result
