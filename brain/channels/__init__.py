"""Canais de entrada do DaviOS (console, Telegram, ...).

Um canal é um adaptador fino por cima do núcleo (ConversationEngine): recebe
mensagens de uma fonte externa e devolve a resposta final. O núcleo NÃO é
alterado por nada daqui.
"""

from brain.channels.base import Channel, IncomingMessage, is_authorized

__all__ = ["Channel", "IncomingMessage", "is_authorized"]
