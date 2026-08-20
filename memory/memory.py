"""Modelo de uma memória persistida."""

from __future__ import annotations

from typing import Any, Optional


class Memory:
    """Envelope de uma entrada de memória.

    ``importance`` continua disponível por compatibilidade. ``base_importance``
    guarda o valor antes de decay; o valor efetivo é calculado pelo
    :class:`MemoryManager` no momento da recuperação.
    """

    def __init__(
        self,
        content: str,
        memory_type: str,
        importance: float,
        emotion: str,
        emotional_intensity: float,
        facts=None,
        *,
        id: Optional[int] = None,
        confidence: float = 0.7,
        base_importance: Optional[float] = None,
        status: str = "active",
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        last_accessed_at: Optional[str] = None,
        access_count: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.id = id
        self.content = content
        self.memory_type = memory_type
        self.importance = float(importance)
        self.base_importance = (
            float(base_importance)
            if base_importance is not None
            else float(importance)
        )
        self.emotion = emotion
        self.emotional_intensity = float(emotional_intensity)
        self.confidence = float(confidence)
        self.status = status
        self.created_at = created_at
        self.updated_at = updated_at
        self.last_accessed_at = last_accessed_at
        self.access_count = int(access_count or 0)
        self.metadata = metadata or {}
        self.facts = facts if facts is not None else []

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    def __repr__(self):
        return (
            f"Memory(id={self.id}, content={self.content!r}, "
            f"memory_type={self.memory_type!r}, importance={self.importance}, "
            f"confidence={self.confidence}, status={self.status!r}, "
            f"facts={self.facts!r})"
        )
