"""Contrato mínimo e funções puras compartilhadas pelos canais do DaviOS.

Este módulo é deliberadamente agnóstico de I/O e de transporte: não conhece
Telegram, console, rede nem o núcleo. Ele define apenas:

    - ``IncomingMessage``: mensagem normalizada recebida de qualquer fonte.
    - ``Channel``: classe base abstrata que todo canal concreto implementa.
    - ``is_authorized``: decisão PURA de whitelist por ``sender_id``.

REGRA DE SEGURANÇA (whitelist): uma lista de ids VAZIA nunca autoriza ninguém.
O canal só conversa com quem estiver explicitamente na lista.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class IncomingMessage:
    """Mensagem já normalizada, independente da fonte (console/Telegram).

    ``sender_id`` identifica quem enviou (usado na whitelist); ``chat_id`` é
    para onde a resposta deve voltar; ``text`` é o conteúdo textual.
    """

    sender_id: int
    chat_id: int
    text: str


# Uma entrada de whitelist em forma de string só vale se for um "-" opcional
# seguido APENAS de dígitos ASCII. fullmatch ancora a string inteira, então
# "123\n" (newline ao final) NÃO é aceito — o que `\d` + `$` permitiria por
# casar antes do newline. A conversão int() pode levantar ValueError para
# strings com mais de ~4300 dígitos (limite do CPython) — capturado no caller.
_ALLOWED_ID_RE = re.compile(r"-?[0-9]+")


def _coerce_allowed_id(candidate):
    """Converte uma entrada da whitelist em ``int``, ou devolve ``None``.

    Aceita:
      - ``int`` de verdade (``bool`` é REJEITADO, pois é subclasse de int);
      - ``str`` com um "-" opcional + somente dígitos ASCII.
    Rejeita (devolve ``None``, sem autorizar): ``bool``, ``float``, ``None``,
    objetos e qualquer outra string. NUNCA levanta exceção.
    """
    if isinstance(candidate, bool):
        return None
    if isinstance(candidate, int):
        return candidate
    if isinstance(candidate, str) and _ALLOWED_ID_RE.fullmatch(candidate):
        try:
            return int(candidate)
        except ValueError:
            # ex.: string com mais de ~4300 dígitos (limite do int()).
            return None
    return None


def is_authorized(sender_id: int, allowed_ids) -> bool:
    """Retorna ``True`` SOMENTE se ``sender_id`` estiver explicitamente na lista.

    Regras (função pura, sem I/O e que NUNCA levanta exceção):
      - ``allowed_ids`` precisa ser ``list``/``tuple``/``set``/``frozenset``.
        Qualquer outro tipo (``str``, ``dict``, ``None``, ``int``) -> ``False``
        — em particular uma ``str`` NÃO é iterada como whitelist (evita que
        "123456789" autorize os ids 1..9 caractere a caractere).
      - ``sender_id`` só vale se ``type(sender_id) is int`` (rejeita ``bool``,
        ``float``, ``str``, ``None``).
      - cada entrada da lista: aceita ``int`` (não ``bool``) e ``str`` de dígitos
        com "-" opcional; o resto é ignorado (não autoriza por ele).
      - lista vazia -> ``False`` (nunca "liberar tudo" por omissão).
    """
    if not isinstance(allowed_ids, (list, tuple, set, frozenset)):
        return False
    if not allowed_ids:
        return False
    if type(sender_id) is not int:  # rejeita bool/float/str/None de forma estrita
        return False

    for candidate in allowed_ids:
        value = _coerce_allowed_id(candidate)
        if value is not None and value == sender_id:
            return True
    return False


class Channel(ABC):
    """Classe base abstrata de um canal de conversa.

    Um canal concreto (ex.: Telegram) normaliza a entrada em
    ``IncomingMessage``, aplica ``is_authorized`` ANTES de qualquer
    processamento e devolve a resposta ao remetente. O núcleo
    (ConversationEngine) é injetado/fornecido pela subclasse — nada aqui
    depende dele.
    """

    #: Nome curto e estável do canal (ex.: "telegram", "console").
    name: str = "channel"

    @abstractmethod
    def start(self) -> None:
        """Inicia o recebimento de mensagens (bloqueante ou não, a critério)."""

    @abstractmethod
    def stop(self) -> None:
        """Encerra o canal e libera recursos associados."""
