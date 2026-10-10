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


def is_authorized(sender_id: int, allowed_ids) -> bool:
    """Retorna ``True`` somente se ``sender_id`` estiver explicitamente na lista.

    Regras (função pura, sem I/O):
      - lista vazia/``None``  -> SEMPRE ``False`` (nunca "liberar tudo" por
        omissão; ausência de whitelist = ninguém autorizado).
      - comparação tolerante a tipo: ``sender_id`` e os ids da lista são
        comparados como ``int``. Ids de tipo errado (ex.: ``"123"``) são
        convertidos quando possível; se não der para converter, não autoriza.
    """
    if not allowed_ids:
        return False

    try:
        target = int(sender_id)
    except (TypeError, ValueError):
        return False

    for candidate in allowed_ids:
        try:
            if int(candidate) == target:
                return True
        except (TypeError, ValueError):
            # Elemento da whitelist com tipo inválido é ignorado (não autoriza
            # por ele), mantendo a verificação estrita e segura.
            continue
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
