"""Persistência SQLite para o subsistema de memória.

O banco trata fatos como afirmações atômicas, e não como uma coluna mutável por
alvo. Isso permite manter preferências passadas, episódios repetidos e fatos
independentes sobre a mesma entidade sem apagar evidências ou contexto.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Optional

from memory.memory import Memory
from memory.memory_fact import MemoryFact


class Database:
    """Repositório SQLite com migração incremental e histórico append-only."""

    def __init__(
        self,
        db_path: str = "memory.db",
        now_provider: Optional[Callable[[], datetime]] = None,
    ):
        self.db_path = db_path
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.cursor = self.connection.cursor()
        self.cursor.execute("PRAGMA foreign_keys = ON")
        self.initialize()

    # ------------------------------------------------------------------
    # Schema and migration
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self._create_base_tables()
        self._ensure_memory_columns()
        self._ensure_fact_columns()
        self._ensure_legacy_history_columns()
        self._remove_legacy_content_uniqueness()
        self._create_auxiliary_tables()
        self._backfill_legacy_values()
        self.connection.commit()

    def _create_base_tables(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'episodic',
                importance REAL NOT NULL DEFAULT 0.5,
                emotion TEXT NOT NULL DEFAULT 'neutral',
                emotional_intensity REAL NOT NULL DEFAULT 0,
                base_importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.7,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT,
                updated_at TEXT,
                last_accessed_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                target TEXT NOT NULL,
                relation TEXT NOT NULL,
                emotion TEXT NOT NULL DEFAULT 'neutral',
                emotional_intensity REAL NOT NULL DEFAULT 0,
                temporal_context TEXT NOT NULL DEFAULT 'current',
                negation INTEGER NOT NULL DEFAULT 0,
                subject TEXT NOT NULL DEFAULT 'user',
                value TEXT,
                confidence REAL NOT NULL DEFAULT 0.7,
                importance REAL NOT NULL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'active',
                source TEXT NOT NULL DEFAULT 'user_statement',
                fact_type TEXT,
                created_at TEXT,
                updated_at TEXT,
                last_accessed_at TEXT,
                access_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
            """
        )
        # A tabela antiga é preservada para compatibilidade e migrada para a
        # tabela append-only abaixo quando possível.
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_fact_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                target TEXT,
                relation TEXT,
                emotion TEXT,
                emotional_intensity REAL,
                changed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
            """
        )

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self.cursor.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            self.cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {definition}"
            )

    def _ensure_memory_columns(self) -> None:
        for column, definition in (
            ("base_importance", "REAL"),
            ("confidence", "REAL"),
            ("status", "TEXT"),
            ("created_at", "TEXT"),
            ("updated_at", "TEXT"),
            ("last_accessed_at", "TEXT"),
            ("access_count", "INTEGER DEFAULT 0"),
            ("metadata", "TEXT DEFAULT '{}'"),
        ):
            self._ensure_column("memories", column, definition)

    def _ensure_fact_columns(self) -> None:
        for column, definition in (
            ("temporal_context", "TEXT"),
            ("negation", "INTEGER DEFAULT 0"),
            ("subject", "TEXT DEFAULT 'user'"),
            ("value", "TEXT"),
            ("confidence", "REAL"),
            ("importance", "REAL"),
            ("status", "TEXT"),
            ("source", "TEXT"),
            ("fact_type", "TEXT"),
            ("created_at", "TEXT"),
            ("updated_at", "TEXT"),
            ("last_accessed_at", "TEXT"),
            ("access_count", "INTEGER DEFAULT 0"),
            ("metadata", "TEXT DEFAULT '{}'"),
        ):
            self._ensure_column("memory_facts", column, definition)

    def _ensure_legacy_history_columns(self) -> None:
        # Registros antigos não tinham como reconstituir estes campos. A partir
        # desta migração, todos os novos eventos passam pela tabela de revisões.
        for column, definition in (
            ("fact_id", "INTEGER"),
            ("subject", "TEXT"),
            ("value", "TEXT"),
            ("temporal_context", "TEXT"),
            ("negation", "INTEGER"),
            ("confidence", "REAL"),
            ("importance", "REAL"),
            ("status", "TEXT"),
            ("revision_type", "TEXT"),
            ("reason", "TEXT"),
            ("evidence_snapshot", "TEXT"),
        ):
            self._ensure_column("memory_fact_history", column, definition)

    def _remove_legacy_content_uniqueness(self) -> None:
        """Remove ``UNIQUE(content)`` de bancos antigos sem perder IDs.

        Episódios repetidos podem ter o mesmo texto; o controle de duplicidade
        agora é uma decisão da camada de domínio, não uma restrição física.
        """

        unique_content = False
        for index in self.cursor.execute("PRAGMA index_list(memories)"):
            if not index["unique"]:
                continue
            index_columns = [
                row["name"]
                for row in self.cursor.execute(
                    f"PRAGMA index_info({index['name']})"
                )
            ]
            if index_columns == ["content"]:
                unique_content = True
                break

        if not unique_content:
            return

        self.connection.commit()
        self.connection.execute("PRAGMA foreign_keys = OFF")
        try:
            self.cursor.execute("BEGIN")
            self.cursor.execute(
                """
                CREATE TABLE memories_rebuilt (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'episodic',
                    importance REAL NOT NULL DEFAULT 0.5,
                    emotion TEXT NOT NULL DEFAULT 'neutral',
                    emotional_intensity REAL NOT NULL DEFAULT 0,
                    base_importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 0.7,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT,
                    updated_at TEXT,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self.cursor.execute(
                """
                INSERT INTO memories_rebuilt (
                    id, content, memory_type, importance, emotion,
                    emotional_intensity, base_importance, confidence, status,
                    created_at, updated_at, last_accessed_at, access_count,
                    metadata
                )
                SELECT
                    id, content, COALESCE(memory_type, 'episodic'),
                    COALESCE(importance, 0.5), COALESCE(emotion, 'neutral'),
                    COALESCE(emotional_intensity, 0),
                    COALESCE(base_importance, importance, 0.5),
                    COALESCE(confidence, 0.7), COALESCE(status, 'active'),
                    created_at, updated_at, last_accessed_at,
                    COALESCE(access_count, 0), COALESCE(metadata, '{}')
                FROM memories
                """
            )
            self.cursor.execute("DROP TABLE memories")
            self.cursor.execute("ALTER TABLE memories_rebuilt RENAME TO memories")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            self.connection.execute("PRAGMA foreign_keys = ON")

    def _create_auxiliary_tables(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_fact_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER,
                memory_id INTEGER NOT NULL,
                revision_type TEXT NOT NULL,
                subject TEXT,
                target TEXT,
                relation TEXT,
                value TEXT,
                emotion TEXT,
                emotional_intensity REAL,
                temporal_context TEXT,
                negation INTEGER,
                confidence REAL,
                importance REAL,
                status TEXT,
                source TEXT,
                fact_type TEXT,
                evidence_snapshot TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                reason TEXT,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id),
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS fact_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL DEFAULT 'statement',
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'user_statement',
                confidence REAL NOT NULL DEFAULT 0.7,
                observed_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id)
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_fact_conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id INTEGER NOT NULL,
                fact_b_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                winner_fact_id INTEGER,
                reason TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT,
                UNIQUE(fact_a_id, fact_b_id),
                FOREIGN KEY(fact_a_id) REFERENCES memory_facts(id),
                FOREIGN KEY(fact_b_id) REFERENCES memory_facts(id),
                FOREIGN KEY(winner_fact_id) REFERENCES memory_facts(id)
            )
            """
        )
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_node TEXT NOT NULL,
                target_node TEXT NOT NULL,
                relation TEXT NOT NULL,
                fact_id INTEGER,
                weight REAL NOT NULL DEFAULT 0.5,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id)
            )
            """
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_facts_target "
            "ON memory_facts(target COLLATE NOCASE)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_facts_memory "
            "ON memory_facts(memory_id)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_revisions_fact "
            "ON memory_fact_revisions(fact_id, recorded_at)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_fact_evidence_fact "
            "ON fact_evidence(fact_id)"
        )

    def _backfill_legacy_values(self) -> None:
        now = self._now()
        self.cursor.execute(
            """
            UPDATE memories
            SET base_importance = COALESCE(base_importance, importance, 0.5),
                confidence = COALESCE(confidence, 0.7),
                status = COALESCE(status, 'active'),
                created_at = COALESCE(created_at, ?),
                updated_at = COALESCE(updated_at, created_at, ?),
                access_count = COALESCE(access_count, 0),
                metadata = COALESCE(metadata, '{}')
            """,
            (now, now),
        )
        self.cursor.execute(
            """
            UPDATE memory_facts
            SET temporal_context = COALESCE(temporal_context, 'unknown'),
                negation = COALESCE(negation, 0),
                subject = COALESCE(subject, 'user'),
                confidence = COALESCE(confidence, 0.7),
                importance = COALESCE(importance, 0.5),
                status = COALESCE(status, 'active'),
                source = COALESCE(source, 'legacy_migration'),
                created_at = COALESCE(created_at, ?),
                updated_at = COALESCE(updated_at, created_at, ?),
                access_count = COALESCE(access_count, 0),
                metadata = COALESCE(metadata, '{}')
            """,
            (now, now),
        )
        self.cursor.execute(
            """
            UPDATE memory_fact_history
            SET temporal_context = COALESCE(temporal_context, 'unknown'),
                negation = COALESCE(negation, 0),
                subject = COALESCE(subject, 'user'),
                confidence = COALESCE(confidence, 0.7),
                importance = COALESCE(importance, 0.5),
                status = COALESCE(status, 'legacy'),
                revision_type = COALESCE(revision_type, 'legacy_history'),
                evidence_snapshot = COALESCE(evidence_snapshot, '[]')
            """
        )

        # Cria uma fotografia inicial para cada fato pré-existente. O passado
        # que o schema antigo nunca registrou é marcado como "unknown", em vez
        # de inventar dados.
        for row in self.cursor.execute(
            """
            SELECT f.*
            FROM memory_facts f
            WHERE NOT EXISTS (
                SELECT 1 FROM memory_fact_revisions r
                WHERE r.fact_id = f.id
            )
            """
        ).fetchall():
            fact = self._fact_from_row(row, include_evidence=False)
            self._write_fact_revision(
                fact,
                revision_type="migration_snapshot",
                reason="Snapshot criado durante a migração do histórico.",
                recorded_at=fact.created_at or now,
            )

    # ------------------------------------------------------------------
    # Serialization and row mapping
    # ------------------------------------------------------------------

    def _now(self, value: Optional[datetime | str] = None) -> str:
        if isinstance(value, str):
            return value
        moment = value or self._now_provider()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _dump(value: Any, fallback: str) -> str:
        return json.dumps(value if value is not None else json.loads(fallback), ensure_ascii=False)

    @staticmethod
    def _load(value: Optional[str], fallback: Any) -> Any:
        if not value:
            return fallback
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, float(value)))

    def _memory_from_row(self, row: sqlite3.Row, facts=None) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            importance=row["importance"],
            base_importance=row["base_importance"],
            emotion=row["emotion"],
            emotional_intensity=row["emotional_intensity"],
            confidence=row["confidence"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"],
            metadata=self._load(row["metadata"], {}),
            facts=facts,
        )

    def _fact_from_row(
        self,
        row: sqlite3.Row,
        *,
        include_evidence: bool = True,
    ) -> MemoryFact:
        fact = MemoryFact(
            id=row["id"],
            memory_id=row["memory_id"],
            subject=row["subject"] or "user",
            target=row["target"],
            relation=row["relation"],
            value=row["value"],
            emotion=row["emotion"] or "neutral",
            emotional_intensity=row["emotional_intensity"] or 0.0,
            temporal_context=row["temporal_context"] or "unknown",
            negation=bool(row["negation"]),
            confidence=row["confidence"] if row["confidence"] is not None else 0.7,
            importance=row["importance"] if row["importance"] is not None else 0.5,
            status=row["status"] or "active",
            source=row["source"] or "user_statement",
            fact_type=row["fact_type"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=row["access_count"] or 0,
            metadata=self._load(row["metadata"], {}),
        )
        if include_evidence and fact.id is not None:
            fact.evidence = self.get_fact_evidence(fact.id)
        return fact

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    def save_memory(
        self,
        memory: Memory,
        *,
        evidence: Optional[Iterable[dict[str, Any] | str]] = None,
        now: Optional[datetime | str] = None,
        allow_duplicate_content: bool = False,
    ) -> Optional[int]:
        """Persiste uma memória e seus fatos.

        O comportamento antigo de ignorar conteúdo exatamente igual continua
        como padrão. Chamadores que representam episódios repetidos podem optar
        por ``allow_duplicate_content=True``.
        """

        if not allow_duplicate_content and self.find_memory(memory) is not None:
            return None

        timestamp = self._now(now)
        self.cursor.execute(
            """
            INSERT INTO memories (
                content, memory_type, importance, emotion, emotional_intensity,
                base_importance, confidence, status, created_at, updated_at,
                last_accessed_at, access_count, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.content,
                memory.memory_type,
                self._clamp(memory.importance),
                memory.emotion,
                memory.emotional_intensity,
                self._clamp(memory.base_importance),
                self._clamp(memory.confidence),
                memory.status,
                timestamp,
                timestamp,
                memory.last_accessed_at,
                memory.access_count,
                self._dump(memory.metadata, "{}"),
            ),
        )
        memory_id = self.cursor.lastrowid
        memory.id = memory_id
        memory.created_at = timestamp
        memory.updated_at = timestamp

        for fact in memory.facts:
            fact.fact_type = fact.fact_type or memory.memory_type
            fact_evidence = fact.evidence or evidence
            self.add_memory_fact(
                memory_id=memory_id,
                target=fact.target,
                relation=fact.relation,
                emotion=fact.emotion,
                emotional_intensity=fact.emotional_intensity,
                temporal_context=fact.temporal_context,
                negation=fact.negation,
                subject=fact.subject,
                value=fact.value,
                confidence=fact.confidence,
                importance=fact.importance,
                status=fact.status,
                source=fact.source,
                fact_type=fact.fact_type,
                evidence=fact_evidence,
                metadata=fact.metadata,
                now=timestamp,
                commit=False,
                fact_object=fact,
            )

        self.connection.commit()
        return memory_id

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        row = self.cursor.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._memory_from_row(row, facts=self.get_memory_facts(memory_id))

    def find_memory(self, memory: Memory | str) -> Optional[Memory]:
        content = memory.content if isinstance(memory, Memory) else str(memory)
        row = self.cursor.execute(
            "SELECT * FROM memories WHERE content = ? ORDER BY id DESC LIMIT 1",
            (content,),
        ).fetchone()
        if row is None:
            return None
        return self._memory_from_row(row, facts=self.get_memory_facts(row["id"]))

    def find_memories(self, *, include_archived: bool = True) -> list[Memory]:
        sql = "SELECT * FROM memories"
        if not include_archived:
            sql += " WHERE status = 'active'"
        sql += " ORDER BY created_at DESC, id DESC"
        rows = self.cursor.execute(sql).fetchall()
        return [
            self._memory_from_row(row, facts=self.get_memory_facts(row["id"]))
            for row in rows
        ]

    def get_memories(self) -> list[Memory]:
        return self.find_memories()

    def update_memory_content(
        self,
        memory_id: int,
        content: str,
        *,
        now: Optional[datetime | str] = None,
    ) -> bool:
        self.cursor.execute(
            "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
            (content, self._now(now), memory_id),
        )
        self.connection.commit()
        return self.cursor.rowcount > 0

    def touch_memory(
        self,
        memory_id: int,
        *,
        now: Optional[datetime | str] = None,
        commit: bool = True,
    ) -> bool:
        timestamp = self._now(now)
        self.cursor.execute(
            """
            UPDATE memories
            SET last_accessed_at = ?, access_count = access_count + 1
            WHERE id = ?
            """,
            (timestamp, memory_id),
        )
        if commit:
            self.connection.commit()
        return self.cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Atomic facts, evidence and immutable revisions
    # ------------------------------------------------------------------

    def add_memory_fact(
        self,
        memory_id: int,
        target: str,
        relation: str,
        emotion: str = "neutral",
        emotional_intensity: float = 0.0,
        temporal_context: str = "current",
        negation: bool = False,
        *,
        subject: str = "user",
        value: Optional[str] = None,
        confidence: float = 0.7,
        importance: float = 0.5,
        status: str = "active",
        source: str = "user_statement",
        fact_type: Optional[str] = None,
        evidence: Optional[Iterable[dict[str, Any] | str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        now: Optional[datetime | str] = None,
        commit: bool = True,
        fact_object: Optional[MemoryFact] = None,
    ) -> int:
        """Adiciona uma afirmação sem fundi-la a fatos do mesmo alvo."""

        timestamp = self._now(now)
        self.cursor.execute(
            """
            INSERT INTO memory_facts (
                memory_id, target, relation, emotion, emotional_intensity,
                temporal_context, negation, subject, value, confidence,
                importance, status, source, fact_type, created_at, updated_at,
                last_accessed_at, access_count, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                target,
                relation,
                emotion or "neutral",
                emotional_intensity or 0.0,
                temporal_context or "unknown",
                int(bool(negation)),
                subject or "user",
                value,
                self._clamp(confidence),
                self._clamp(importance),
                status or "active",
                source or "user_statement",
                fact_type,
                timestamp,
                timestamp,
                None,
                0,
                self._dump(metadata, "{}"),
            ),
        )
        fact_id = self.cursor.lastrowid
        fact = fact_object or MemoryFact(
            target=target,
            relation=relation,
            emotion=emotion or "neutral",
            emotional_intensity=emotional_intensity or 0.0,
            temporal_context=temporal_context or "unknown",
            negation=bool(negation),
            subject=subject or "user",
            value=value,
            confidence=self._clamp(confidence),
            importance=self._clamp(importance),
            status=status or "active",
            source=source or "user_statement",
            fact_type=fact_type,
            metadata=metadata or {},
        )
        fact.id = fact_id
        fact.memory_id = memory_id
        fact.created_at = timestamp
        fact.updated_at = timestamp
        fact.evidence = []

        for item in evidence or []:
            self._add_fact_evidence(
                fact_id,
                item,
                now=timestamp,
                write_revision=False,
                commit=False,
            )
        fact.evidence = self.get_fact_evidence(fact_id)
        self._write_fact_revision(fact, revision_type="created", recorded_at=timestamp)
        self._sync_graph_for_fact(fact, timestamp)
        self.touch_memory(memory_id, now=timestamp, commit=False)
        if commit:
            self.connection.commit()
        return fact_id

    def get_memory_facts(
        self,
        memory_id: int,
        *,
        include_inactive: bool = True,
    ) -> list[MemoryFact]:
        sql = "SELECT * FROM memory_facts WHERE memory_id = ?"
        if not include_inactive:
            sql += " AND status = 'active'"
        sql += " ORDER BY created_at ASC, id ASC"
        return [
            self._fact_from_row(row)
            for row in self.cursor.execute(sql, (memory_id,)).fetchall()
        ]

    def get_fact(self, fact_id: int) -> Optional[MemoryFact]:
        row = self.cursor.execute(
            "SELECT * FROM memory_facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return self._fact_from_row(row) if row is not None else None

    def find_facts_by_target(
        self,
        target: str,
        *,
        include_inactive: bool = True,
    ) -> list[MemoryFact]:
        sql = "SELECT * FROM memory_facts WHERE LOWER(target) = LOWER(?)"
        if not include_inactive:
            sql += " AND status = 'active'"
        sql += " ORDER BY created_at DESC, id DESC"
        return [
            self._fact_from_row(row)
            for row in self.cursor.execute(sql, (target,)).fetchall()
        ]

    def find_memories_by_target(
        self,
        target: str,
        *,
        include_archived: bool = True,
    ) -> list[Memory]:
        sql = """
            SELECT DISTINCT m.id
            FROM memories m
            INNER JOIN memory_facts f ON f.memory_id = m.id
            WHERE LOWER(f.target) = LOWER(?)
        """
        if not include_archived:
            sql += " AND m.status = 'active'"
        ids = [row["id"] for row in self.cursor.execute(sql, (target,)).fetchall()]
        return [memory for memory_id in ids if (memory := self.get_memory(memory_id))]

    def find_memory_by_fact_target(self, target: str) -> Optional[Memory]:
        memories = self.find_memories_by_target(target)
        return memories[0] if memories else None
        return memories[0] if memories else None

    def find_memory_by_semantic_key(
        self,
        subject: str,
        target: str,
        relation_family: str,
        *,
        include_inactive: bool = True,
    ) -> Optional[Memory]:
        """Localiza uma memória baseada na chave semântica.

        Retorna a primeira memória que contenha um fato com
        subject, target e relation_family correspondentes.
        """
        facts = self.find_facts_for_semantic_key(
            subject,
            target,
            relation_family,
            include_inactive=include_inactive,
        )
        if not facts:
            return None
        return self.get_memory(facts[0].memory_id)

    def update_memory_fact(
        self,
        memory_id: int,
        target: Optional[str] = None,
        relation: Optional[str] = None,
        emotion: Optional[str] = None,
        emotional_intensity: Optional[float] = None,
        temporal_context: Optional[str] = None,
        negation: Optional[bool] = None,
        *,
        fact_id: Optional[int] = None,
        subject: Optional[str] = None,
        value: Optional[str] = None,
        confidence: Optional[float] = None,
        importance: Optional[float] = None,
        status: Optional[str] = None,
        source: Optional[str] = None,
        fact_type: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        reason: Optional[str] = None,
        revision_type: str = "updated",
        now: Optional[datetime | str] = None,
    ) -> bool:
        """Atualiza um fato específico e registra uma revisão completa.

        Para compatibilidade, quando ``fact_id`` não é informado escolhe a
        afirmação mais recente daquele alvo na memória. Novos fluxos devem usar
        ``fact_id`` para evitar ambiguidade.
        """

        if fact_id is None:
            if not target:
                return False
            row = self.cursor.execute(
                """
                SELECT id FROM memory_facts
                WHERE memory_id = ? AND LOWER(target) = LOWER(?)
                ORDER BY created_at DESC, id DESC LIMIT 1
                """,
                (memory_id, target),
            ).fetchone()
            if row is None:
                return False
            fact_id = row["id"]

        previous = self.get_fact(fact_id)
        if previous is None or previous.memory_id != memory_id:
            return False

        timestamp = self._now(now)
        next_values = {
            "target": target if target is not None else previous.target,
            "relation": relation if relation is not None else previous.relation,
            "emotion": emotion if emotion is not None else previous.emotion,
            "emotional_intensity": (
                emotional_intensity
                if emotional_intensity is not None
                else previous.emotional_intensity
            ),
            "temporal_context": (
                temporal_context
                if temporal_context is not None
                else previous.temporal_context
            ),
            "negation": int(
                bool(negation) if negation is not None else previous.negation
            ),
            "subject": subject if subject is not None else previous.subject,
            "value": value if value is not None else previous.value,
            "confidence": self._clamp(
                confidence if confidence is not None else previous.confidence
            ),
            "importance": self._clamp(
                importance if importance is not None else previous.importance
            ),
            "status": status if status is not None else previous.status,
            "source": source if source is not None else previous.source,
            "fact_type": fact_type if fact_type is not None else previous.fact_type,
            "metadata": self._dump(
                metadata if metadata is not None else previous.metadata,
                "{}",
            ),
        }
        self.cursor.execute(
            """
            UPDATE memory_facts
            SET target = ?, relation = ?, emotion = ?, emotional_intensity = ?,
                temporal_context = ?, negation = ?, subject = ?, value = ?,
                confidence = ?, importance = ?, status = ?, source = ?,
                fact_type = ?, metadata = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                next_values["target"],
                next_values["relation"],
                next_values["emotion"],
                next_values["emotional_intensity"],
                next_values["temporal_context"],
                next_values["negation"],
                next_values["subject"],
                next_values["value"],
                next_values["confidence"],
                next_values["importance"],
                next_values["status"],
                next_values["source"],
                next_values["fact_type"],
                next_values["metadata"],
                timestamp,
                fact_id,
            ),
        )
        fact = self.get_fact(fact_id)
        if fact is None:
            self.connection.rollback()
            return False
        self._write_fact_revision(
            fact,
            revision_type=revision_type,
            reason=reason,
            recorded_at=timestamp,
        )
        self._sync_graph_for_fact(fact, timestamp)
        self.touch_memory(memory_id, now=timestamp, commit=False)
        self.connection.commit()
        return True

    def set_fact_status(
        self,
        fact_id: int,
        status: str,
        *,
        reason: Optional[str] = None,
        now: Optional[datetime | str] = None,
    ) -> bool:
        fact = self.get_fact(fact_id)
        if fact is None:
            return False
        return self.update_memory_fact(
            memory_id=fact.memory_id,
            fact_id=fact_id,
            status=status,
            reason=reason,
            revision_type="status_changed",
            now=now,
        )

    def _write_fact_revision(
        self,
        fact: MemoryFact,
        *,
        revision_type: str,
        reason: Optional[str] = None,
        recorded_at: Optional[str] = None,
    ) -> int:
        self.cursor.execute(
            """
            INSERT INTO memory_fact_revisions (
                fact_id, memory_id, revision_type, subject, target, relation,
                value, emotion, emotional_intensity, temporal_context, negation,
                confidence, importance, status, source, fact_type,
                evidence_snapshot, metadata, reason, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact.id,
                fact.memory_id,
                revision_type,
                fact.subject,
                fact.target,
                fact.relation,
                fact.value,
                fact.emotion,
                fact.emotional_intensity,
                fact.temporal_context,
                int(bool(fact.negation)),
                fact.confidence,
                fact.importance,
                fact.status,
                fact.source,
                fact.fact_type,
                self._dump(fact.evidence, "[]"),
                self._dump(fact.metadata, "{}"),
                reason,
                recorded_at or self._now(),
            ),
        )
        return self.cursor.lastrowid

    def get_fact_history(self, fact_id: int) -> list[dict[str, Any]]:
        rows = self.cursor.execute(
            """
            SELECT * FROM memory_fact_revisions
            WHERE fact_id = ?
            ORDER BY recorded_at ASC, id ASC
            """,
            (fact_id,),
        ).fetchall()
        return [self._revision_to_dict(row) for row in rows]

    def get_memory_fact_history(self, memory_id: int) -> list[dict[str, Any]]:
        """Retorna revisões completas e entradas legadas ainda existentes."""

        history = [
            self._revision_to_dict(row)
            for row in self.cursor.execute(
                """
                SELECT * FROM memory_fact_revisions
                WHERE memory_id = ?
                ORDER BY recorded_at ASC, id ASC
                """,
                (memory_id,),
            ).fetchall()
        ]
        legacy_rows = self.cursor.execute(
            """
            SELECT * FROM memory_fact_history
            WHERE memory_id = ?
            ORDER BY changed_at ASC, id ASC
            """,
            (memory_id,),
        ).fetchall()
        for row in legacy_rows:
            history.append(
                {
                    "id": f"legacy-{row['id']}",
                    "fact_id": row["fact_id"],
                    "memory_id": row["memory_id"],
                    "revision_type": row["revision_type"] or "legacy_history",
                    "target": row["target"],
                    "relation": row["relation"],
                    "subject": row["subject"] or "user",
                    "value": row["value"],
                    "emotion": row["emotion"],
                    "emotional_intensity": row["emotional_intensity"],
                    "temporal_context": row["temporal_context"] or "unknown",
                    "negation": bool(row["negation"]),
                    "confidence": row["confidence"],
                    "importance": row["importance"],
                    "status": row["status"] or "legacy",
                    "source": "legacy_history",
                    "fact_type": None,
                    "evidence": self._load(row["evidence_snapshot"], []),
                    "metadata": {},
                    "reason": row["reason"],
                    "recorded_at": row["changed_at"],
                    "changed_at": row["changed_at"],
                    "legacy": True,
                }
            )
        return sorted(
            history,
            key=lambda item: (item.get("recorded_at") or "", str(item["id"])),
        )

    def _revision_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "fact_id": row["fact_id"],
            "memory_id": row["memory_id"],
            "revision_type": row["revision_type"],
            "target": row["target"],
            "relation": row["relation"],
            "subject": row["subject"],
            "value": row["value"],
            "emotion": row["emotion"],
            "emotional_intensity": row["emotional_intensity"],
            "temporal_context": row["temporal_context"],
            "negation": bool(row["negation"]),
            "confidence": row["confidence"],
            "importance": row["importance"],
            "status": row["status"],
            "source": row["source"],
            "fact_type": row["fact_type"],
            "evidence": self._load(row["evidence_snapshot"], []),
            "metadata": self._load(row["metadata"], {}),
            "reason": row["reason"],
            "recorded_at": row["recorded_at"],
            "changed_at": row["recorded_at"],
            "legacy": False,
        }

    def get_fact_evidence(self, fact_id: int) -> list[dict[str, Any]]:
        rows = self.cursor.execute(
            """
            SELECT * FROM fact_evidence
            WHERE fact_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (fact_id,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "fact_id": row["fact_id"],
                "type": row["evidence_type"],
                "content": row["content"],
                "source": row["source"],
                "confidence": row["confidence"],
                "observed_at": row["observed_at"],
                "created_at": row["created_at"],
                "metadata": self._load(row["metadata"], {}),
            }
            for row in rows
        ]

    def _add_fact_evidence(
        self,
        fact_id: int,
        evidence: dict[str, Any] | str,
        *,
        now: Optional[datetime | str] = None,
        write_revision: bool = True,
        commit: bool = True,
    ) -> int:
        item = {"content": evidence} if isinstance(evidence, str) else dict(evidence)
        content = str(item.get("content", "")).strip()
        if not content:
            raise ValueError("A evidência precisa conter texto em 'content'.")
        timestamp = self._now(now)
        self.cursor.execute(
            """
            INSERT INTO fact_evidence (
                fact_id, evidence_type, content, source, confidence,
                observed_at, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fact_id,
                item.get("type", "statement"),
                content,
                item.get("source", "user_statement"),
                self._clamp(item.get("confidence", 0.7)),
                item.get("observed_at", timestamp),
                self._dump(item.get("metadata", {}), "{}"),
                timestamp,
            ),
        )
        evidence_id = self.cursor.lastrowid
        if write_revision:
            fact = self.get_fact(fact_id)
            if fact is not None:
                fact.evidence = self.get_fact_evidence(fact_id)
                self.cursor.execute(
                    "UPDATE memory_facts SET updated_at = ? WHERE id = ?",
                    (timestamp, fact_id),
                )
                fact.updated_at = timestamp
                self._write_fact_revision(
                    fact,
                    revision_type="evidence_added",
                    recorded_at=timestamp,
                )
                self.touch_memory(fact.memory_id, now=timestamp, commit=False)
        if commit:
            self.connection.commit()
        return evidence_id

    def add_fact_evidence(
        self,
        fact_id: int,
        evidence: dict[str, Any] | str,
        *,
        now: Optional[datetime | str] = None,
    ) -> int:
        return self._add_fact_evidence(fact_id, evidence, now=now)

    # ------------------------------------------------------------------
    # Conflict and graph persistence
    # ------------------------------------------------------------------

    def find_facts_for_semantic_key(
        self,
        subject: str,
        target: str,
        relation_family: str,
        *,
        include_inactive: bool = False,
        exclude_fact_id: Optional[int] = None,
    ) -> list[MemoryFact]:
        sql = """
            SELECT * FROM memory_facts
            WHERE LOWER(subject) = LOWER(?) AND LOWER(target) = LOWER(?)
        """
        params: list[Any] = [subject, target]
        if not include_inactive:
            sql += " AND status = 'active'"
        if exclude_fact_id is not None:
            sql += " AND id != ?"
            params.append(exclude_fact_id)
        facts = [
            self._fact_from_row(row)
            for row in self.cursor.execute(sql, params).fetchall()
        ]
        return [fact for fact in facts if fact.relation_family == relation_family]

    def find_active_preferences(
        self,
        subject: str = "user",
        relation_family: str = "preference",
    ) -> list[MemoryFact]:
        """Busca todos os fatos ativos de uma familia semantica.

        Util para perguntas genericas como 'do que eu gosto?' onde o alvo
        nao e conhecido antecipadamente.
        """
        sql = """
            SELECT * FROM memory_facts
            WHERE LOWER(subject) = LOWER(?) AND status = 'active'
        """
        params: list[Any] = [subject]
        return [
            fact
            for fact in (
                self._fact_from_row(row)
                for row in self.cursor.execute(sql, params).fetchall()
            )
            if fact.relation_family == relation_family
        ]

    def ensure_semantic_uniqueness_index(self) -> bool:
        """Cria o indice unico semantico para evitar fatos ativos duplicados.

        O indice e parcial (WHERE status = 'active' AND fact_type IS NOT NULL),
        permitindo que fatos inativos/historicos coexistam sem violar a regra.
        """
        try:
            self.cursor.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_facts_semantic
                ON memory_facts (subject, target, fact_type)
                WHERE status = 'active' AND fact_type IS NOT NULL
            """)
            self.connection.commit()
            return True
        except Exception:
            self.connection.rollback()
            return False

    def normalize_all_fact_targets(self, normalize_fn) -> int:
        """Corrige targets invalidos heredados de normalizaciones antiguas.

        Recorre todos los hechos y aplica ``normalize_fn`` al target,
        persistiendo el nuevo valor cuando cambie.

        Returns:
            Cantidad de targets corregidos.
        """
        rows = self.cursor.execute(
            "SELECT id, target FROM memory_facts"
        ).fetchall()
        fixed = 0
        for row in rows:
            current = row["target"] or ""
            if not current:
                continue
            normalized = normalize_fn(current) if callable(normalize_fn) else current
            if normalized and normalized != current:
                self.cursor.execute(
                    "UPDATE memory_facts SET target = ? WHERE id = ?",
                    (normalized, row["id"]),
                )
                fixed += 1
        if fixed:
            self.connection.commit()
        return fixed

    def consolidate_duplicate_memories(self) -> dict[str, Any]:
        """Consolida memorias duplicadas por concepto (chave semantica).

        Agrupa hechos por (subject, target, fact_type), elige una memoria
        canonica (la mas reciente y activa), mueve las evidencias de las
        duplicatas al hecho canonico y marca duplicatas como inactivas.

        Nunca borra: historial y evidencias se preservan.

        Returns:
            Dict con resumen de la consolidacion.
        """
        from collections import defaultdict

        groups = defaultdict(list)
        rows = self.cursor.execute("SELECT * FROM memory_facts").fetchall()
        for row in rows:
            fact = self._fact_from_row(row)
            subject = (fact.subject or "user").strip().casefold()
            target = (fact.target or "").strip().casefold()
            fact_type = fact.fact_type or "episodic"
            groups[(subject, target, fact_type)].append(fact)

        total_duplicates = 0
        total_evidence_moved = 0
        consolidated_memories = []

        for group in groups.values():
            if len({f.memory_id for f in group if f.memory_id}) < 2:
                continue

            # Memoria canonica: hecho activo mas reciente (mayor id).
            active = [f for f in group if f.status == "active"]
            ordered = sorted(
                active if active else group,
                key=lambda f: (f.id or 0),
                reverse=True,
            )
            canonical_fact = ordered[0]
            canonical_memory_id = canonical_fact.memory_id

            for fact in group:
                if fact.id is None or fact.id == canonical_fact.id:
                    continue
                # Mover evidencias de la duplicata al hecho canonico
                for ev in self.get_fact_evidence(fact.id):
                    self._add_fact_evidence(
                        canonical_fact.id,
                        {
                            "content": ev.get("content"),
                            "source": ev.get("source"),
                            "confidence": ev.get("confidence"),
                            "type": ev.get("type", "statement"),
                        },
                        write_revision=False,
                        commit=False,
                    )
                    total_evidence_moved += 1

                if fact.status != "inactive":
                    self.set_fact_status(fact.id, "inactive", reason="dup consolidada")
                if fact.memory_id and fact.memory_id != canonical_memory_id:
                    if fact.memory_id not in consolidated_memories:
                        consolidated_memories.append(fact.memory_id)
                    self.cursor.execute(
                        "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                        ("inactive", self._now(), fact.memory_id),
                    )
                    total_duplicates += 1

        self.connection.commit()
        return {
            "action": "consolidate",
            "consolidated_memories": total_duplicates,
            "evidence_moved": total_evidence_moved,
        }

    def record_conflict(

        fact_a_id: int,
        fact_b_id: int,
        *,
        reason: str,
        winner_fact_id: Optional[int] = None,
        status: str = "open",
        now: Optional[datetime | str] = None,
        commit: bool = True,
    ) -> int:
        timestamp = self._now(now)
        row = self.cursor.execute(
            "SELECT * FROM memory_fact_conflicts WHERE fact_a_id = ? AND fact_b_id = ? AND status = 'open'",
            (fact_a_id, fact_b_id),
        ).fetchone()
        if row is None:
            self.cursor.execute(
                """
                INSERT INTO memory_fact_conflicts
                    (fact_a_id, fact_b_id, reason, status, winner_fact_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fact_a_id, fact_b_id, reason, status, winner_fact_id, timestamp),
            )
            conflict_id = self.cursor.lastrowid
        else:
            conflict_id = row["id"]
            self.cursor.execute(
                """
                UPDATE memory_fact_conflicts
                SET status = ?, winner_fact_id = ?, reason = ?,
                    resolved_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    winner_fact_id,
                    reason,
                    timestamp if status == "resolved" else None,
                    conflict_id,
                ),
            )
        if commit:
            self.connection.commit()
        return conflict_id

    def resolve_conflict(
        self,
        conflict_id: int,
        winner_fact_id: int,
        *,
        reason: str,
        now: Optional[datetime | str] = None,
    ) -> bool:
        self.cursor.execute(
            """
            UPDATE memory_fact_conflicts
            SET status = 'resolved', winner_fact_id = ?, reason = ?,
                resolved_at = ?
            WHERE id = ?
            """,
            (winner_fact_id, reason, self._now(now), conflict_id),
        )
        self.connection.commit()
        return self.cursor.rowcount > 0

    def get_conflicts(self, *, fact_id: Optional[int] = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM memory_fact_conflicts"
        params: tuple[Any, ...] = ()
        if fact_id is not None:
            sql += " WHERE fact_a_id = ? OR fact_b_id = ?"
            params = (fact_id, fact_id)
        sql += " ORDER BY created_at ASC, id ASC"
        return [dict(row) for row in self.cursor.execute(sql, params).fetchall()]

    def _upsert_graph_edge(
        self,
        source_node: str,
        target_node: str,
        relation: str,
        *,
        fact_id: Optional[int] = None,
        weight: float = 0.5,
        metadata: Optional[dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> int:
        timestamp = timestamp or self._now()
        row = self.cursor.execute(
            """
            SELECT id FROM memory_graph_edges
            WHERE source_node = ? AND target_node = ? AND relation = ?
              AND COALESCE(fact_id, -1) = COALESCE(?, -1)
            """,
            (source_node, target_node, relation, fact_id),
        ).fetchone()
        if row is None:
            self.cursor.execute(
                """
                INSERT INTO memory_graph_edges (
                    source_node, target_node, relation, fact_id, weight,
                    metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_node,
                    target_node,
                    relation,
                    fact_id,
                    self._clamp(weight),
                    self._dump(metadata, "{}"),
                    timestamp,
                    timestamp,
                ),
            )
            return self.cursor.lastrowid
        self.cursor.execute(
            """
            UPDATE memory_graph_edges
            SET weight = ?, metadata = ?, updated_at = ?
            WHERE id = ?
            """,
            (self._clamp(weight), self._dump(metadata, "{}"), timestamp, row["id"]),
        )
        return row["id"]

    def _sync_graph_for_fact(self, fact: MemoryFact, timestamp: str) -> None:
        fact_node = f"fact:{fact.id}"
        memory_node = f"memory:{fact.memory_id}"
        subject_node = f"entity:{(fact.subject or 'user').strip().casefold()}"
        target_node = f"entity:{(fact.target or '').strip().casefold()}"
        self._upsert_graph_edge(
            memory_node,
            fact_node,
            "contains",
            fact_id=fact.id,
            weight=1.0,
            timestamp=timestamp,
        )
        self._upsert_graph_edge(
            fact_node,
            target_node,
            "about",
            fact_id=fact.id,
            weight=fact.confidence,
            timestamp=timestamp,
        )
        self._upsert_graph_edge(
            subject_node,
            target_node,
            fact.relation,
            fact_id=fact.id,
            weight=fact.confidence,
            metadata={
                "negation": bool(fact.negation),
                "temporal_context": fact.temporal_context,
                "status": fact.status,
            },
            timestamp=timestamp,
        )

    def get_memory_graph(
        self,
        *,
        memory_id: Optional[int] = None,
        target: Optional[str] = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Retorna uma projeção leve de grafo sem trocar SQLite por outro DB."""

        facts: list[MemoryFact]
        if memory_id is not None:
            facts = self.get_memory_facts(memory_id)
        elif target is not None:
            facts = self.find_facts_by_target(target)
        else:
            facts = [
                self._fact_from_row(row)
                for row in self.cursor.execute("SELECT * FROM memory_facts").fetchall()
            ]
        fact_ids = {fact.id for fact in facts if fact.id is not None}
        nodes: dict[str, dict[str, Any]] = {}
        for fact in facts:
            memory = self.get_memory(fact.memory_id)
            if memory is not None:
                nodes[f"memory:{memory.id}"] = {
                    "id": f"memory:{memory.id}",
                    "type": "memory",
                    "label": memory.content,
                    "memory_type": memory.memory_type,
                }
            nodes[f"fact:{fact.id}"] = {
                "id": f"fact:{fact.id}",
                "type": "fact",
                "label": f"{fact.subject} {fact.relation} {fact.target}",
                "status": fact.status,
                "confidence": fact.confidence,
            }
            for entity, label in (
                (f"entity:{fact.subject.casefold()}", fact.subject),
                (f"entity:{fact.target.casefold()}", fact.target),
            ):
                nodes[entity] = {"id": entity, "type": "entity", "label": label}

        if not fact_ids:
            return {"nodes": [], "edges": []}
        placeholders = ", ".join("?" for _ in fact_ids)
        edges = [
            {
                "id": row["id"],
                "source": row["source_node"],
                "target": row["target_node"],
                "relation": row["relation"],
                "fact_id": row["fact_id"],
                "weight": row["weight"],
                "metadata": self._load(row["metadata"], {}),
            }
            for row in self.cursor.execute(
                f"SELECT * FROM memory_graph_edges WHERE fact_id IN ({placeholders})",
                tuple(fact_ids),
            ).fetchall()
        ]
        for conflict in self.get_conflicts():
            if conflict["fact_a_id"] not in fact_ids and conflict["fact_b_id"] not in fact_ids:
                continue
            edges.append(
                {
                    "id": f"conflict:{conflict['id']}",
                    "source": f"fact:{conflict['fact_a_id']}",
                    "target": f"fact:{conflict['fact_b_id']}",
                    "relation": "conflicts_with",
                    "fact_id": None,
                    "weight": 1.0,
                    "metadata": {
                        "status": conflict["status"],
                        "winner_fact_id": conflict["winner_fact_id"],
                        "reason": conflict["reason"],
                    },
                }
            )
        return {"nodes": list(nodes.values()), "edges": edges}

    def close(self) -> None:
        self.connection.close()
