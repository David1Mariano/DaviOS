"""Modelos para as afirmações atômicas armazenadas na memória.

Uma ``Memory`` pode conter vários ``MemoryFact``. Um fato não é identificado
apenas pelo alvo: duas afirmações sobre ``pizza`` podem coexistir, pois cada uma
tem seu próprio id, contexto temporal, evidências e estado de resolução.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MemoryFact:
    """Uma afirmação atômica sobre uma entidade.

    Os primeiros seis campos preservam o construtor usado pelo projeto antes da
    evolução do modelo. Os demais campos tornam o fato auditável e permitem que
    preferências, episódios e fatos independentes coexistam.
    """

    target: str
    relation: str
    emotion: str = "neutral"
    emotional_intensity: float = 0.0
    temporal_context: str = "current"
    negation: bool = False
    subject: str = "user"
    value: Optional[str] = None
    confidence: float = 0.7
    importance: float = 0.5
    status: str = "active"
    source: str = "user_statement"
    fact_type: Optional[str] = None
    id: Optional[int] = None
    memory_id: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_accessed_at: Optional[str] = None
    access_count: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_preference(self) -> bool:
        return self.relation.lower() in {
            "like",
            "dislike",
            "love",
            "hate",
            "prefer",
            "avoid",
            "favorite",
        }

    @property
    def relation_family(self) -> str:
        """Grupo semântico usado para localizar possíveis conflitos."""

        relation = (self.relation or "").strip().lower()
        if relation in {
            "like",
            "dislike",
            "love",
            "hate",
            "prefer",
            "avoid",
            "favorite",
        }:
            return "preference"
        return relation

    def semantic_key(self) -> tuple[str, str, str]:
        return (
            (self.subject or "user").strip().casefold(),
            (self.target or "").strip().casefold(),
            self.relation_family,
        )

    def snapshot(self) -> dict[str, Any]:
        """Representação serializável completa usada no histórico."""

        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "subject": self.subject,
            "target": self.target,
            "relation": self.relation,
            "value": self.value,
            "emotion": self.emotion,
            "emotional_intensity": self.emotional_intensity,
            "temporal_context": self.temporal_context,
            "negation": bool(self.negation),
            "confidence": self.confidence,
            "importance": self.importance,
            "status": self.status,
            "source": self.source,
            "fact_type": self.fact_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }
