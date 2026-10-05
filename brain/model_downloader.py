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

import hashlib
import json
import logging
import math
import os
import secrets
import shutil
import socket
import threading
import time
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
# Outro download já está usando o MESMO destino. Não é falha nem recusa: o
# recurso está temporariamente ocupado. Nenhum byte foi transferido por este
# downloader, e nada foi alterado.
STATUS_LOCKED = "locked"
# Confirmação pedida mas não atendida: `require_confirmation=True` e nenhum
# `confirm` fornecido. NADA foi baixado — nem `.part`, nem arquivo final.
STATUS_CONFIRMATION_REQUIRED = "confirmation_required"
# O usuário recebeu a prévia e respondeu NÃO. Também não baixa nada.
STATUS_DECLINED = "declined"

# =============================================================================
# POLÍTICA DE DOWNLOADLOCK
# =============================================================================
# Um lock protege o `.part` e o destino de UM modelo. O objetivo é evitar que
# dois downloader manipulem o mesmo arquivo — a auditoria reproduziu o caso em
# que um processo crashando apagava o `.part` de outro, desperdiçando o download
# inteiro.
#
# A regra que governa toda esta política: NUNCA roubar o lock de um download
# legítimo em andamento. Um falso positivo aqui é pior que o problema que o
# lock resolve, porque reintroduz exatamente a corrupção que tentamos evitar.

LOCK_SUFFIX = ".lock"

# Velocidade MÍNIMA assumida para um download, em bytes por segundo.
# 50.000 B/s ≈ 0,4 Mbps: uma conexão deliberadamente pessimista, mas possível
# (móvel em zona rural, hotspot congestionado, uplink de servidor remoto).
# Uma velocidade alta tornaria o TTL curto demais e causaria falsos positivos
# em justamente as máquinas que mais demorarão.
LOCK_ASSUMED_MIN_BYTES_PER_SEC = 50_000

# Multiplicador de segurança sobre o tempo estimado. Cobre picos de latência,
# pausas por GC, e a validação + SHA-256 após o download (~10 s para 19,8 GB).
LOCK_AGE_SAFETY_FACTOR = 3.0

# Piso absoluto, em segundos (1 hora). Modelos sem `size_bytes`, ou minúsculos,
# ainda precisam de uma janela que não roube um download de verdade.
LOCK_MIN_AGE_SECONDS = 3600.0

# Teto absoluto, em segundos (7 dias). Um lock de 19,8 GB a 50 KB/s levaria
# ~4,8 dias; o teto evita que um lock órfão vire permanente só porque o modelo
# é grande. Passado o teto, PID/hostname ainda precisam indicar processo morto:
# o teto sozinho NÃO rouba lock ativo.
LOCK_MAX_AGE_SECONDS = 7 * 24 * 3600.0

# Tamanho máximo do arquivo de lock. O conteúdo é um JSON pequeno e conhecido;
# um lock muito maior significa que algo estranho foi escrito ali.
LOCK_MAX_CONTENT_BYTES = 4096


def lock_age_limit(size_bytes: int) -> float:
    """Idade máxima (segundos) antes de um lock poder ser considerado velho.

    `size_bytes` é o tamanho ESPERADO do modelo. O cálculo é:

        estimado = size_bytes / LOCK_ASSUMED_MIN_BYTES_PER_SEC
        limite   = estimado * LOCK_AGE_SAFETY_FACTOR

    limitado a [LOCK_MIN_AGE_SECONDS, LOCK_MAX_AGE_SECONDS].

    Todas as unidades são explícitas: bytes, bytes/segundo e segundos.
    """
    try:
        size = int(size_bytes or 0)
    except (TypeError, ValueError):
        size = 0
    if size <= 0:
        return LOCK_MIN_AGE_SECONDS
    estimated = size / float(LOCK_ASSUMED_MIN_BYTES_PER_SEC)
    return min(
        max(estimated * LOCK_AGE_SAFETY_FACTOR, LOCK_MIN_AGE_SECONDS),
        LOCK_MAX_AGE_SECONDS,
    )


def _process_is_alive(pid: int) -> bool:
    """True se um processo com este PID existe AGORA neste host.

    PID sozinho nunca decide: o SO reutiliza PIDs, e um lock antigo pode
    "encontrar" vivo um processo sem nenhuma relação. Por isso esta função é
    só UMA das três condições de stale (ver `DownloadLock._is_stale`).

    No Windows, `os.kill(pid, 0)` NÃO é seguro: sinal 0 não existe lá e a
    chamada pode encerrar o processo. Usa-se `OpenProcess` via ctypes, que
    apenas consulta. `psutil` resolveria, mas adicionaria uma dependência que
    o resto do projeto não tem.
    """
    if pid <= 0:
        return False

    if os.name == "nt":
        try:
            import ctypes

            # PROCESS_QUERY_LIMITED_INFORMATION: o menos privilegiado que ainda
            # responde "existe?".
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            # ERROR_ACCESS_DENIED (5) = existe, mas sem permissão para
            # inspecionar. Isso é "vivo", não "morto".
            return ctypes.windll.kernel32.GetLastError() == 5
        except Exception:  # noqa: BLE001 - nunca derruba o download por isso
            logger.debug("[MODEL] não foi possível verificar o PID %s", pid)
            return True  # na dúvida, tratamos como vivo (não rouba)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # existe, mas é de outro usuário
    except OSError:
        return False
    return True


def _read_lock_content(lock_path: Path) -> Optional[dict]:
    """Lê e valida o JSON de um lock. Devolve None se não for confiável."""
    try:
        if lock_path.stat().st_size > LOCK_MAX_CONTENT_BYTES:
            return None
        with open(str(lock_path), "rb", buffering=0) as fh:
            # Lê APENAS a região do JSON, e SEM buffer. Um handle com buffer
            # pode pedir mais bytes do que pedimos e cruzar o byte sentinela,
            # travado — e o Windows nega o arquivo inteiro com
            # PermissionError, fazendo um lock vivo parecer ilegível.
            bruto = fh.read(LOCK_SENTINEL_OFFSET)
        texto = bruto.decode("utf-8", "replace").strip()
        if not texto:
            return None
        data = json.loads(texto)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _has_valid_metadata(data: Optional[dict]) -> bool:
    """True se o lock carrega TODA a metadata de que a política depende.

    Um lock sem metadata completa não foi escrito por esta implementação
    (ou foi corrompido), e portanto não tem dono que se possa verificar.
    Classificá-lo como "vivo" o deixaria bloqueando o destino para sempre —
    sem nenhum caminho de recuperação. Por isso a ausência é tratada como
    stale, nunca como lock legítimo.

    Obrigatórios:
      pid         > 0        → verificar liveness
      hostname    não vazio  → saber se o PID é local
      started_at  finito     → calcular a idade
      size_bytes  >= 0       → definir a janela de idade
      token       não vazio  → provar posse no release

    `model_id` é diagnóstico e NÃO é obrigatório.
    """
    if not isinstance(data, dict):
        return False

    pid = data.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False

    hostname = data.get("hostname")
    if not isinstance(hostname, str) or not hostname.strip():
        return False

    started_at = data.get("started_at")
    if isinstance(started_at, bool) or not isinstance(started_at, (int, float)):
        return False
    if not math.isfinite(float(started_at)):
        return False

    size = data.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        return False

    token = data.get("token")
    if not isinstance(token, str) or not token.strip():
        return False

    return True


def _lock_byte_offset(handle, unlock: bool) -> None:
        """Trava/libera o byte SENTINELA, fora da região do JSON.

        Por que um byte separado: `msvcrt.locking` (e `LockFile`) nega
        leitura/escrita do range travado a QUALQUER handle, inclusive do mesmo
        processo. Se travassemos o offset 0, o `read_text()` do próprio dono
        falharia com PermissionError, e o lock pareceria "ilegível" para o
        diagnóstico — abrindo caminho para classificar um lock vivo como
        órfão. Por isso o JSON ocupa [0, LOCK_SENTINEL_OFFSET) e o byte
        travado é logo depois dele.
        """
        if os.name == "nt":
            import msvcrt

            handle.seek(LOCK_SENTINEL_OFFSET)
            op = msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK
            msvcrt.locking(handle.fileno(), op, 1)
        else:
            import fcntl

            if unlock:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle) -> None:
    """Libera o lock do SO. Idempotente e nunca levanta."""
    try:
        _lock_byte_offset(handle, unlock=True)
    except Exception:  # noqa: BLE001
        # Fechar o descritor também libera o lock; nunca mascarar erro de release.
        pass


def _try_lock_file(handle) -> bool:
    """Trava EXCLUSIVA e NÃO-BLOQUEANTE o byte sentinela.

    True  = adquirido (esta instância é a dona)
    False = já está travado por outro processo

    O SO é o árbitro: para um mesmo inode, apenas um handle obtém a trava.
    """
    try:
        _lock_byte_offset(handle, unlock=False)
    except (OSError, IOError):
        return False
    return True


# Offset do byte que o sistema operacional efetivamente trava.
#
# O JSON de metadata ocupa a região [0, LOCK_SENTINEL_OFFSET). O byte travado
# fica logo depois, para que a trava NÃO bloqueie a leitura do próprio JSON:
# `msvcrt.locking`/`LockFile` nega acesso ao range travado a qualquer handle,
# até no mesmo processo, e ler o lock do próprio dono falharia.
#
# Deve ser > qualquer tamanho de JSON válido e <= LOCK_MAX_CONTENT_BYTES.
LOCK_SENTINEL_OFFSET = 512

# Tolerância (segundos) para deriva de relógio entre a gravação e a leitura do
# lock. Um `started_at` além disso não veio de um relógio coerente e, sem
# correção, bloquearia o destino para sempre.
LOCK_CLOCK_SKEW_TOLERANCE_SECONDS = 300.0


class DownloadLock:
    """Exclusão mútua sobre o `.part`/destino de UM modelo.

    Um lock por `destination` — nunca global e nunca por URL — porque o recurso
    realmente disputado é o arquivo, e dois modelos distintos precisam baixar
    ao mesmo tempo.

    A aquisição é ATÔMICA: `os.open(O_CREAT|O_EXCL)` falha se o arquivo já
    existe, sem janela de TOCTOU entre "existe?" e "cria?".

    A liberação só acontece se esta instância for a DONA, confirmada por um
    token aleatório gravado no conteúdo. Uma instância que nunca adquiriu o
    lock não tem como apagá-lo.
    """

    def __init__(self, lock_path: Path, model_id: str = "", size_bytes: int = 0):
        self.lock_path = Path(lock_path)
        self.model_id = model_id
        self.size_bytes = int(size_bytes or 0)
        self._token: Optional[str] = None
        self._acquired = False

    @property
    def acquired(self) -> bool:
        return self._acquired

    def holder(self) -> Optional[str]:
        """Descrição legível de quem está com o lock: 'host:pid'.

        Diagnóstico apenas — nunca usado para decidir segurança.
        """
        data = _read_lock_content(self.lock_path)
        if not data:
            return None
        host = data.get("hostname") or "?"
        pid = data.get("pid")
        return str(host) if pid is None else "%s:%s" % (host, pid)

    def _is_stale(self) -> tuple[bool, str]:
        """(stale, motivo). Três condições independentes; qualquer uma basta.

        IMPORTANTE: desde a adoção do lock do sistema operacional, esta função
        NÃO decide mais posse. Ela responde apenas "esta metadata describe um
        dono que ainda pode ser verificado?". Um lock realmente mantido pelo SO
        nunca deve ser roubado com base em JSON — o SO é a autoridade.
        """
        data = _read_lock_content(self.lock_path)
        if data is None:
            # (C) ilegível, grande demais, ou não-JSON: não é confiável.
            return True, "conteudo ilegivel ou invalido"

        # (C2) JSON válido porém SEM metadata completa. Antes, `{}` ou um
        # `pid` sem `started_at` caíam em "não stale" e bloqueavam o destino
        # para sempre, sem caminho de recuperação.
        if not _has_valid_metadata(data):
            return True, "metadata obrigatoria ausente ou invalida"

        host = data.get("hostname")
        pid = data["pid"]
        started_at = float(data["started_at"])
        size = data["size_bytes"]

        # (D) Relógio adiantado ou lixo: um `started_at` no futuro não é um
        # download em andamento, é um registro inconsistente.
        agora = time.time()
        if started_at > agora + LOCK_CLOCK_SKEW_TOLERANCE_SECONDS:
            return True, "timestamp no futuro (%.0fs a frente)" % (
                started_at - agora
            )

        # (A) o dono não está mais vivo NESTE host.
        if host == socket.gethostname() and not _process_is_alive(pid):
            return True, "processo %s nao existe mais" % pid
        # Hostname diferente: não podemos verificar o PID de lá, então a idade
        # (condição B) é a única que pode liberar este lock. Nunca confiar em
        # um PID que veio de outra máquina.

        # (B) idade acima do limite derivado do tamanho.
        age = agora - started_at
        limit = lock_age_limit(size)
        if age > limit:
            return True, "idade %.0fs acima do limite %.0fs" % (age, limit)

        return False, ""

    def _open_lock_file(self):
        """Abre (criando se preciso) o arquivo de lock e garante 1 byte.

        O arquivo NUNCA é removido durante o ciclo de vida normal. Isso é
        essencial para a garantia de exclusividade: enquanto o inode existe,
        o lock do SO recai sempre sobre o MESMO byte do MESMO objeto. Se o
        arquivo fosse apagado e recriado, um contender poderia travar um inode
        diferente do que o dono atual possui — double-owner.

        No Windows, `msvcrt.locking` trava um RANGE de bytes e falha se o
        arquivo estiver vazio; por isso garantimos pelo menos 1 byte.
        """
        try:
            handle = open(str(self.lock_path), "r+b")
        except FileNotFoundError:
            # Criar sem `a+b`: em modo append o Python ignora `seek()` para
            # escrita, e o byte sentinela acabaria no offset 0 — exatamente
            # onde o JSON mora.
            handle = open(str(self.lock_path), "w+b")
        try:
            # Garante que exista o byte sentinela em LOCK_SENTINEL_OFFSET.
            # Sem ele, `msvcrt.locking` falha e nada trava.
            handle.seek(0, os.SEEK_END)
            if handle.tell() < LOCK_SENTINEL_OFFSET + 1:
                handle.seek(LOCK_SENTINEL_OFFSET)
                handle.write(b"\x00")
                handle.flush()
        except OSError:
            handle.close()
            raise
        return handle

    def _write_metadata(self, handle) -> None:
        """Grava a metadata de diagnóstico no arquivo que já está travado.

        NÃO decide posse — a posse é a do SO.

        Não usamos `truncate()`: no Windows, truncar pode invalidar o range
        que `msvcrt.locking` travou, e então outro processo conseguiria travar
        o mesmo byte. Em vez disso sobrescrevemos a partir do offset 0 e, se o
        conteúdo novo for mais curto que o antigo, preenchemos o resto com
        espaços — preservando o tamanho do arquivo e, portanto, a trava.
        """
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": time.time(),  # segundos desde a época (wall clock)
            "size_bytes": self.size_bytes,
            "model_id": self.model_id,
            "token": self._token,
        }
        data = json.dumps(payload).encode("utf-8")
        try:
            handle.seek(0)
            handle.write(data)
            # Preenche até o byte sentinela com espaços: mantém o tamanho do
            # arquivo estável (a trava do SO continua válida) e deixa um
            # leitor de diagnóstico com JSON parseável.
            if len(data) < LOCK_SENTINEL_OFFSET:
                handle.write(b" " * (LOCK_SENTINEL_OFFSET - len(data)))
            handle.flush()
        except OSError as exc:
            logger.warning("[MODEL] falha ao escrever metadata do lock: %s", exc)

    def acquire(self) -> bool:
        """Tenta adquirir. NÃO ESPERA: devolve False se outro tem o lock.

        A posse é decidida EXCLUSIVAMENTE pelo sistema operacional (flock no
        POSIX, LockFile/msvcrt.locking no Windows). O JSON é apenas
        diagnóstico.

        A ausência de espera é deliberada. Um download de 19,8 GB pode levar
        dezenas de minutos; segurar o lock em espera prenderia a thread sem
        dar nenhum sinal ao usuário. Recusar com `STATUS_LOCKED` é mais honesto
        e infinitamente mais simples de testar.

        INVARIANTE: para um destino, no máximo uma instância tem
        `acquired == True`. Isso vale porque `_acquired` só é setado depois de
        o SO confirmar a trava, e o SO serializa contenders sobre o mesmo
        inode. Não existe `unlink() -> create()` neste caminho.
        """
        if self._acquired:
            return True

        try:
            handle = self._open_lock_file()
        except OSError as exc:
            # Sem lock do SO não há garantia de exclusividade. Falhar de forma
            # segura é preferível a aceitar uma corrida silenciosa.
            logger.warning("[MODEL] nao foi possivel abrir o lock: %s", exc)
            return False

        if not _try_lock_file(handle):
            # Outro processo/thread possui o byte travado.
            handle.close()
            return False

        # A partir daqui SOMENTE esta instância possui o lock. Um processo que
        # morre aqui devolve o lock ao SO automaticamente (o descritor fecha),
        # sem precisar de heurística de PID.
        self._handle = handle
        self._token = secrets.token_hex(16)
        self._acquired = True
        self._write_metadata(handle)
        return True

    def release(self) -> None:
        """Libera o lock devolvendo-o ao SO.

        O arquivo `.lock` permanece no disco (com a última metadata, para
        diagnóstico). Removê-lo permitiria que um contender travasse um inode
        novo e se auto-declarasse dono.
        """
        if not self._acquired:
            return
        handle = getattr(self, "_handle", None)
        if handle is not None:
            _unlock_file(handle)
            try:
                handle.close()
            except OSError:
                pass
        self._handle = None
        self._acquired = False
        self._token = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


# Sentinela interno: "nada a decidir aqui, siga o fluxo normal do download".
# Usamos um objeto único (e comparação por identidade) para não confundir
# "nenhum erro" com um DownloadResult válido.
_CONTINUE = object()

ProgressCallback = Callable[[int, int], None]

# Função que recebe a prévia e devolve a decisão do chamador. `True` autoriza
# o download; qualquer outra coisa recusa. Um chamador não interactivo (CLI,
# teste, serviço) decide como perguntar — este módulo só guarda a resposta.
ConfirmCallback = Callable[["DownloadPreview"], bool]


@dataclass
class DownloadPreview:
    """Prévia READ-ONLY de um download: o que BAIXARIA, não o que baixou.

    Existe para que nada seja transferido antes de o usuário saber o quê. Todos
    os campos vêm de fontes reais (catálogo, disco, destino derivado) — nenhum
    valor é inventado ou preenchido com URL montada a partir do nome.

    `download()` usa esta mesma prévia internamente, então o que o chamador vê
    é exatamente o que o downloader checou.
    """

    model_id: str = ""
    name: str = ""
    expected_bytes: int = 0
    expected_is_estimate: bool = False
    download_url: str = ""
    destination: str = ""
    part_path: str = ""
    free_bytes: Optional[int] = None
    space_message: str = ""
    # Estado já observado no disco, antes de qualquer escrita.
    already_installed: bool = False
    destination_exists: bool = False
    # Motivo que impede o download agora (catálogo sem fonte, destino fora da
    # pasta de modelos, arquivo inválido no lugar, disco cheio). Vazio = pode
    # baixar, sujeito à confirmação.
    blocked_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    # Existe alguma prévia válida para este modelo? Quando False, os campos
    # acima não devem ser interpretados como uma oferta de download.
    downloadable: bool = False

    @property
    def can_download(self) -> bool:
        """True quando nada impede o download AGORA (não significa autorizado)."""
        return self.downloadable and not self.blocked_reason

    @property
    def expected_human(self) -> str:
        return _human_gb(self.expected_bytes)

    @property
    def free_human(self) -> str:
        return _human_gb(self.free_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "expected_bytes": self.expected_bytes,
            "expected_is_estimate": self.expected_is_estimate,
            "expected_human": self.expected_human,
            "download_url": self.download_url,
            "destination": self.destination,
            "part_path": self.part_path,
            "free_bytes": self.free_bytes,
            "free_human": self.free_human,
            "space_message": self.space_message,
            "already_installed": self.already_installed,
            "destination_exists": self.destination_exists,
            "blocked_reason": self.blocked_reason,
            "warnings": list(self.warnings),
            "downloadable": self.downloadable,
            "can_download": self.can_download,
        }

    def summary(self) -> str:
        """Descrição curta e legível da prévia, para o usuário."""
        if not self.downloadable:
            return self.blocked_reason or f"Download de {self.model_id} indisponível."
        if self.already_installed:
            return f"{self.name} já está instalado em {self.destination}."
        lines = [
            f"Modelo   : {self.name} ({self.model_id})",
            f"Tamanho  : ~{self.expected_human}",
            f"Origem   : {self.download_url}",
            f"Destino  : {self.destination}",
            f"Espaço   : {self.free_human} livres",
        ]
        return "\n".join(lines)


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
    # --- Integridade SHA-256 ------------------------------------------- #
    # Três campos juntos, porque `hash_verified=False` é ambíguo sozinho:
    # pode ser "o catálogo não traz hash" ou "o hash não bateu". Os três
    # juntos deixam a diferença explícita:
    #   sem hash   -> expected=None, actual=<hex>, verified=False
    #   hash ok    -> expected=<hex>, actual=<hex>, verified=True
    #   hash errado-> expected=<hex>, actual=<outro>, verified=False
    expected_sha256: Optional[str] = None
    actual_sha256: Optional[str] = None
    hash_verified: bool = False
    # --- DownloadLock -------------------------------------------------- #
    # True se este download fez alguma espera pelo lock. Hoje é sempre False:
    # a política é NÃO esperar (um download de 19,8 GB pode levar dezenas de
    # minutos, e prender a thread sem sinal ao usuário é pior que recusar).
    # O campo existe para que a camada acima possa distinguir esse caso se a
    # política mudar, sem breaking change.
    lock_waited: bool = False
    # Descrição de quem ocupava o lock quando houve conflito: "host:pid".
    # Informativo, para diagnóstico. NUNCA usado para decidir segurança — um
    # PID recycled e um hostname forjado não diriam nada sobre o dono real.
    lock_holder: Optional[str] = None

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
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "hash_verified": self.hash_verified,
            "lock_waited": self.lock_waited,
            "lock_holder": self.lock_holder,
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
        if self.status == STATUS_CONFIRMATION_REQUIRED:
            return (
                f"O download de {self.model_id} precisa de confirmação antes "
                "de começar."
            )
        if self.status == STATUS_DECLINED:
            return f"Download de {self.model_id} não autorizado; nada foi baixado."
        if self.status == STATUS_LOCKED:
            who = self.lock_holder or "outro processo"
            return (
                f"Já existe um download em andamento para {self.model_id} "
                f"({who}). Nenhum download novo foi iniciado."
            )
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

    def lock_path_for(self, destination: Path) -> Path:
        """Caminho do lock associado a um destino: `<destino>.lock`.

        Fica ao lado do `.part` e do `.gguf`, portanto dentro de `models_root()`
        — nunca aceita um caminho vindo do chamador. `destination` sempre vem
        de `destination_path()` (derivado do catálogo) e já passou por
        `_ensure_within_models_root()`.

        O sufixo `.lock` é diferente tanto de `.part` quanto de `.gguf`, então
        nem o `ModelCatalog` nem o `validate_file()` o confundem com um modelo.
        """
        return Path(destination).with_name(Path(destination).name + LOCK_SUFFIX)

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
    # Integridade SHA-256
    # ------------------------------------------------------------------ #

    def _compute_sha256(self, path: Path) -> str:
        """SHA-256 do arquivo, lido em chunks (nunca o arquivo inteiro).

        O tamanho dos modelos é de|GiBs, então carregar o arquivo em memória
        seria inviável. A leitura usa `CHUNK_SIZE` e o hash é acumulado
        incrementalmente, o que mantém o pico de memória em poucos MiB,
        independentemente do tamanho do `.gguf`.
        """
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(self.chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def verify_sha256(
        self,
        path: Path,
        model: ModelInfo,
        result: Optional[DownloadResult] = None,
    ) -> tuple[list[str], list[str]]:
        """Confere a integridade do arquivo contra o SHA-256 do catálogo.

        Devolve (erros, avisos) no mesmo formato de `validate_file()`, para que
        o chamador trate os dois validadores igual.

        - catálogo sem hash -> nenhum erro; UM aviso dizendo que a integridade
          NÃO foi verificada. O silêncio seria uma falsa garantia.
        - hash confere     -> nenhum erro, nenhum aviso.
        - hash não confere -> um erro, com o esperado e o obtido, e nada mais.

        Os valores esperados/observados são escritos em `result` quando um
        DownloadResult é fornecido, para que a camada de cima possa explicar a
        falha sem recalcular nada.
        """
        errors: list[str] = []
        warnings: list[str] = []
        expected = model.sha256

        if result is not None:
            result.expected_sha256 = expected

        if not expected:
            warnings.append(
                f"O catálogo não traz SHA-256 para {model.id}: a integridade "
                "criptográfica deste arquivo não foi verificada."
            )
            return errors, warnings

        try:
            actual = self._compute_sha256(path)
        except OSError as exc:
            errors.append(f"Não foi possível calcular o SHA-256: {exc}")
            return errors, warnings

        if result is not None:
            result.actual_sha256 = actual

        # `expected` já foi normalizado em minúsculas pelo catálogo, e
        # `hexdigest()` também devolve minúsculas: a comparação é direta.
        if actual != expected:
            errors.append(
                f"SHA-256 incompatível: esperado {expected}, obtido {actual}. "
                "O arquivo pode estar corrompido ou ter sido adulterado."
            )
            return errors, warnings

        if result is not None:
            result.hash_verified = True
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

    def _free_space(self, target_dir: Optional[Path] = None) -> Optional[int]:
        """Bytes livres no volume do destino, ou None se não foi possível medir.

        Usado pela prévia. Nunca levanta exceção: um disco ilegível é apenas
        "espaço desconhecido", que o chamador deve mostrar como tal em vez de
        assumir zero (o que sugeriria disco cheio sem evidência).
        """
        probe = Path(target_dir) if target_dir else self.models_root()
        if not probe.exists():
            probe = self.project_root
        try:
            return int(shutil.disk_usage(str(probe)).free)
        except OSError as exc:
            logger.debug("[MODEL] não foi possível medir o disco livre: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Prévia (read-only)
    # ------------------------------------------------------------------ #

    def preview(self, model: Any) -> DownloadPreview:
        """Descreve o que um `download()` FARIA, sem baixar nem criar arquivo.

        Puramente informativo e seguro: não abre conexão, não escreve nada e não
        altera `active_model_id` nem o catálogo. Reaproveita exatamente as
        mesmas validações do download (fonte no catálogo, destino confinado,
        arquivo existente, espaço) para que a prévia não possa prometer algo que
        a execução faria de modo diferente.

        `downloadable=False` significa que o modelo não existe no catálogo ou o
        catálogo não declara origem — nunca que a URL foi "construída" aqui.
        """
        resolved = self.resolve(model)
        if resolved is None:
            return DownloadPreview(
                model_id=str(model),
                blocked_reason=f"Modelo não encontrado no catálogo: {model!r}",
            )
        model = resolved

        preview = DownloadPreview(
            model_id=model.id,
            name=model.name,
            expected_bytes=int(model.size_bytes or 0),
            expected_is_estimate=bool(model.size_is_estimate),
            download_url=model.download_url or "",
        )

        # 1. O catálogo autoriza? (mesma condição do download, passo 1)
        if not model.download_available or not model.download_url:
            preview.blocked_reason = f"{model.name}: {model.download_note or (
                'O catálogo não declara uma origem de download confirmada para '
                'este modelo.'
            )}"
            return preview
        preview.downloadable = True

        # 2. Destino confinado (mesma proteção contra path traversal).
        destination = self.destination_path(model)
        preview.destination = str(destination)
        preview.part_path = str(self.part_path_for(destination))
        safety_error = self._ensure_within_models_root(destination)
        if safety_error:
            preview.blocked_reason = safety_error
            return preview

        # 3. O que já existe no destino? (aviso, nunca sobrescrita)
        if destination.exists():
            preview.destination_exists = True
            errors, warnings = self.validate_file(destination, model)
            preview.warnings.extend(warnings)
            if not errors:
                # A prévia promete o que a execução fará, então precisa aplicar
                # a mesma verificação de integridade: um arquivo já instalado
                # com o hash errado FAZ `download()` recusar.
                hash_errors, hash_warnings = self.verify_sha256(
                    destination, model
                )
                preview.warnings.extend(hash_warnings)
                if hash_errors:
                    preview.blocked_reason = " ".join(hash_errors)
                    return preview
                preview.already_installed = True
            else:
                # `download()` recusaria aqui sem `allow_replace_invalid`.
                preview.blocked_reason = (
                    f"Já existe um arquivo em {destination}, mas ele parece "
                    f"inválido ({'; '.join(errors)}). O DaviOS não sobrescreve "
                    "arquivos existentes automaticamente."
                )
                return preview

        # 4. Cabe no disco?
        fits, space_message = self.check_space(model, destination.parent)
        preview.space_message = space_message
        preview.free_bytes = self._free_space(destination.parent)
        if "apertado" in space_message or "não verificado" in space_message:
            preview.warnings.append(space_message)
        if not fits:
            preview.blocked_reason = space_message
            return preview

        return preview

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
        confirm: Optional[ConfirmCallback] = None,
        require_confirmation: bool = False,
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
        - confirm — `confirm(preview) -> bool`, chamado com a prévia ANTES de
          qualquer byte trafegar. True autoriza; False recusa.
        - require_confirmation — se True, a confirmação é obrigatória: sem
          `confirm` o download NÃO começa e volta STATUS_CONFIRMATION_REQUIRED.
          Default False para não mudar o comportamento dos chamadores atuais.

        A confirmação é a ÚLTIMA etapa antes da rede. Todas as recusas mais
        fortes (catálogo sem fonte, destino inseguro, arquivo inválido no lugar,
        disco insuficiente) vêm ANTES dela: o usuário não precisa responder a uma
        pergunta sobre algo que já está errado.

        `preview()` é read-only e não é afetado por nenhuma destas flags.

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
        # Checagem inicial, SÓ LEITURA: `defer_write=True` adia qualquer
        # escrita (o `os.replace` para `.invalid`) para a revalidação sob o
        # lock. A decisão final é confirmada lá, com o estado real.
        existing_error = self._handle_existing(
            destination, model, result, allow_replace_invalid, defer_write=True
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

        # --- 5. Confirmação do chamador --------------------------------- #
        # Só aqui: todas as recusas mais fortes já passaram. A prévia vem da
        # MESMA função que `preview()` expõe, então o chamador decide vendo
        # exatamente o que será feito. Nenhum byte trafegou até aqui.
        if require_confirmation or confirm is not None:
            preview = self.preview(model)
            for warning in preview.warnings:
                if warning not in result.validation_warnings:
                    result.validation_warnings.append(warning)

            if confirm is None:
                result.status = STATUS_CONFIRMATION_REQUIRED
                result.error = (
                    "Este download exige confirmação explícita e nenhuma foi "
                    "fornecida. Passe confirm(preview) -> bool para decidir."
                )
                logger.info("[MODEL] download aguardando confirmação: %s", model.id)
                return result

            if not confirm(preview):
                result.status = STATUS_DECLINED
                result.error = f"Download de {model.name} recusado pelo chamador."
                logger.info("[MODEL] download recusado na confirmação: %s", model.id)
                return result

        # --- 6. DownloadLock: exclusão mútua sobre o `.part` ------------ #
        # Adquirido AQUI, depois da confirmação: a confirmação do usuário não
        # pode prender o lock enquanto ele lê o prompt. Modelos diferentes têm
        # destinos diferentes e, portanto, locks diferentes — nunca se bloqueiam.
        lock = DownloadLock(
            self.lock_path_for(destination),
            model_id=model.id,
            size_bytes=expected,
        )
        if not lock.acquire():
            result.status = STATUS_LOCKED
            result.lock_waited = False      # política: não esperar
            result.lock_holder = lock.holder()
            logger.info(
                "[MODEL] download já em andamento para %s (holder=%s)",
                model.id,
                result.lock_holder,
            )
            return result

        # --- 7. Revalidação pós-lock ------------------------------------ #
        # A decisão de `_handle_existing()` foi tomada ANTES do lock, e pode
        # ter ficado obsoleta enquanto esperávamos. Sob o lock, o estado real é
        # o único que vale: se outro processo acabou de promover este modelo,
        # não devemos sobrescrevê-lo cegamente.
        #
        # A liberação do lock é feita em TODOS os caminhos a partir daqui — o
        # `return` do recheck está dentro do `try` cujo `finally` libera.
        try:
            recheck = self._handle_existing(
                destination, model, result, allow_replace_invalid
            )
            if recheck is not _CONTINUE:
                return result

            # --- 8. Agora sim: rede e disco de escrita ------------------- #
            # Tudo daqui roda sob o lock: `_run_download()` escreve o `.part`,
            # valida, calcula o SHA-256 e promove com `os.replace()`. O
            # `finally` cobre sucesso, falha, cancelamento, SHA mismatch e
            # qualquer exceção inesperada.
            logger.info("[MODEL] download iniciado: %s -> %s", model.id, destination)
            return self._run_download(
                model, destination, part, result, progress_callback,
                cancel_event, keep_partial,
            )
        finally:
            lock.release()

    # ------------------------------------------------------------------ #
    # Fluxo interno
    # ------------------------------------------------------------------ #

    def _handle_existing(
        self,
        destination: Path,
        model: ModelInfo,
        result: DownloadResult,
        allow_replace_invalid: bool,
        defer_write: bool = False,
    ) -> Any:
        """O que fazer quando já existe algo no destino FINAL.

        Política (explícita e coberta por testes):
        - nada existe         -> _CONTINUE, segue o download;
        - existe e é válido   -> already_installed; NÃO sobrescreve;
        - existe e é inválido -> recusa explicando o motivo. Só move o arquivo
          para `<nome>.invalid` se `allow_replace_invalid=True`. Nunca
          destruímos um arquivo existente em silêncio.

        A integridade SHA-256 é verificada AQUI TAMBÉM, e não só em
        `_run_download`, porque um arquivo já presente no disco nunca passa
        pelo fluxo de download.
        """
        if not destination.exists():
            return _CONTINUE

        errors, warnings = self.validate_file(destination, model)
        if not errors:
            # A integridade SHA-256 entra AQUI, e não só no `_run_download`:
            # sem isso, um `.gguf` adulterado localmente e do MESMO tamanho
            # passaria despercebido e seria aceito como "já instalado" — a
            # verificação só protegeria downloads novos, nunca os arquivos que
            # já estão no disco.
            hash_errors, hash_warnings = self.verify_sha256(destination, model, result)
            result.validation_warnings.extend(warnings)
            result.validation_warnings.extend(hash_warnings)

            if hash_errors:
                result.status = STATUS_REFUSED
                result.validation_errors.extend(hash_errors)
                result.error = (
                    f"Já existe um arquivo em {destination}, mas ele NÃO confere "
                    f"com o SHA-256 oficial de {model.id}. O DaviOS não aceita "
                    "nem sobrescreve esse arquivo automaticamente: remova-o ou "
                    "mova-o para .invalid e baixe de novo."
                )
                logger.warning(
                    "[MODEL] arquivo existente com SHA-256 incompatível: %s",
                    destination,
                )
                return result

            result.status = STATUS_ALREADY_INSTALLED
            result.success = True
            result.already_installed = True
            try:
                result.downloaded_bytes = destination.stat().st_size
            except OSError:
                pass
            logger.info(
                "[MODEL] já instalado, não sobrescrito: %s (sha256=%s)",
                model.id,
                "verificado" if result.hash_verified else "sem verificação",
            )
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

        if defer_write:
            # Chamada ANTES do lock (checagem inicial, só leitura): o arquivo
            # inválido será movido para `.invalid` na revalidação SOB o lock.
            # Mover aqui escreveria no destino sem exclusão mútua, que é
            # exatamente o que o DownloadLock existe para impedir.
            return _CONTINUE

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

        # --- Integridade SHA-256 ----------------------------------------- #
        # Calculado SOBRE O `.part`, ANTES da promoção: é o único momento em que
        # a decisão de publicar o arquivo ainda não foi tomada. Calcular depois
        # do `os.replace` exigiria reverter um `.gguf` que o ModelCatalog já
        # enxergaria como instalado.
        hash_errors, hash_warnings = self.verify_sha256(part, model, result)
        result.validation_warnings.extend(hash_warnings)
        if hash_errors:
            result.status = STATUS_FAILED
            result.validation_errors.extend(hash_errors)
            result.error = (
                "O arquivo baixado não passou na verificação de integridade: "
                + "; ".join(hash_errors)
            )
            logger.error("[MODEL][ERROR] SHA-256 falhou: %s", hash_errors)
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
