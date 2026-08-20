import sqlite3

from memory.memory import Memory
from memory.memory_fact import MemoryFact


class Database:
    def __init__(self, db_path="memory.db"):
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.initialize()

    def initialize(self):
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT UNIQUE,
                memory_type TEXT,
                importance REAL,
                emotion TEXT,
                emotional_intensity REAL
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                target TEXT,
                relation TEXT,
                emotion TEXT,
                emotional_intensity REAL,
                temporal_context TEXT,
                negation INTEGER,
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
            """
        )

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_fact_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                target TEXT,
                relation TEXT,
                emotion TEXT,
                emotional_intensity REAL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            )
            """
        )

        existing_columns = {
            row[1]
            for row in self.cursor.execute(
                "PRAGMA table_info(memory_facts)"
            )
        }

        if "temporal_context" not in existing_columns:
            self.cursor.execute(
                "ALTER TABLE memory_facts ADD COLUMN temporal_context TEXT"
            )

        if "negation" not in existing_columns:
            self.cursor.execute(
                "ALTER TABLE memory_facts ADD COLUMN negation BOOLEAN DEFAULT 0"
            )

        self.connection.commit()

    def save_memory(self, memory):
        existing_memory = self.find_memory(memory)
        if existing_memory is not None:
            print(f"[DB] Memória já existe no banco: '{memory.content}'. Inserção ignorada.")
            return None

        try:
            self.cursor.execute(
                """
                INSERT INTO memories (
                    content,
                    memory_type,
                    importance,
                    emotion,
                    emotional_intensity
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory.content,
                    memory.memory_type,
                    memory.importance,
                    memory.emotion,
                    memory.emotional_intensity,
                ),
            )

            memory_id = self.cursor.lastrowid

            for fact in memory.facts:
                self.cursor.execute(
                    """
                    INSERT INTO memory_facts (
                        memory_id,
                        target,
                        relation,
                        emotion,
                        emotional_intensity,
                        temporal_context,
                        negation
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id,
                        getattr(fact, "target", None),
                        getattr(fact, "relation", None),
                        getattr(fact, "emotion", None),
                        getattr(fact, "emotional_intensity", None),
                        getattr(fact, "temporal_context", None),
                        int(getattr(fact, "negation", False)),
                    ),
                )

            self.connection.commit()
            print(f"[DB] Nova memória salva com ID {memory_id}.")
            return memory_id

        except sqlite3.IntegrityError:
            print(f"[DB] Bloqueado pelo SQLite: Memória duplicada '{memory.content}'.")
            self.connection.rollback()
            return None

    def get_memory_facts(self, memory_id):
        self.cursor.execute(
            """
            SELECT
                target,
                relation,
                emotion,
                emotional_intensity,
                temporal_context,
                negation
            FROM memory_facts
            WHERE memory_id = ?
            """,
            (memory_id,),
        )

        rows = self.cursor.fetchall()

        return [
            MemoryFact(
                target=row[0],
                relation=row[1],
                emotion=row[2],
                emotional_intensity=row[3],
                temporal_context=row[4],
                negation=bool(row[5]),
            )
            for row in rows
        ]

    def find_facts_by_target(self, target):
        self.cursor.execute(
            """
           SELECT
            target,
            relation,
            emotion,
            emotional_intensity,
            temporal_context,
            negation
        FROM memory_facts
        WHERE target = ?
                    """,
                    (target,),
                )

        rows = self.cursor.fetchall()
        return [
            MemoryFact(
                target=row[0],
                relation=row[1],
                emotion=row[2],
                emotional_intensity=row[3],
                temporal_context=row[4],
                negation=row[5],
            )
            for row in rows
        ]

    def find_memories_by_target(self, target):
        self.cursor.execute(
            """
            SELECT DISTINCT m.id
            FROM memories m
            INNER JOIN memory_facts f ON m.id = f.memory_id
            WHERE f.target = ?
            """,
            (target,),
        )

        memories = []
        for row in self.cursor.fetchall():
            memory_id = row[0]
            self.cursor.execute(
                """
                SELECT content, memory_type, importance, emotion, emotional_intensity
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            )
            memory_row = self.cursor.fetchone()
            if memory_row is None:
                continue

            facts = self.get_memory_facts(memory_id)
            memories.append(
                Memory(
                    content=memory_row[0],
                    memory_type=memory_row[1],
                    importance=memory_row[2],
                    emotion=memory_row[3],
                    emotional_intensity=memory_row[4],
                    facts=facts,
                )
            )

        return memories

    def find_memory(self, memory):
        self.cursor.execute(
            """
            SELECT id, content, memory_type, importance, emotion, emotional_intensity
            FROM memories
            WHERE content = ?
            """,
            (memory.content,),
        )

        result = self.cursor.fetchone()
        if result is None:
            return None

        memory_id = result[0]
        facts = self.get_memory_facts(memory_id)

        found = Memory(
            content=result[1],
            memory_type=result[2],
            importance=result[3],
            emotion=result[4],
            emotional_intensity=result[5],
            facts=facts,
        )
        found.id = memory_id
        return found

    def find_memories(self):
        self.cursor.execute("SELECT * FROM memories")
        rows = self.cursor.fetchall()

        memories = []
        for row in rows:
            memory_id = row[0]
            facts = self.get_memory_facts(memory_id)
            record = Memory(
                content=row[1],
                memory_type=row[2],
                importance=row[3],
                emotion=row[4],
                emotional_intensity=row[5],
                facts=facts,
            )
            record.id = memory_id
            memories.append(record)

        return memories

    def get_memories(self):
        return self.find_memories()

    def update_memory_fact(
        self,
        memory_id,
        target,
        relation,
        emotion,
        emotional_intensity,
        temporal_context=None,
        negation=None,
    ):
        self.cursor.execute(
            """
            SELECT target, relation, emotion, emotional_intensity
            FROM memory_facts
            WHERE memory_id = ? AND LOWER(target) = LOWER(?)
            """,
            (memory_id, target),
        )

        previous = self.cursor.fetchone()

        if previous is None:
            return False

        previous_target = previous[0]
        previous_relation = previous[1]
        previous_emotion = previous[2]
        previous_intensity = previous[3]

        self.cursor.execute(
            """
            INSERT INTO memory_fact_history (
                memory_id,
                target,
                relation,
                emotion,
                emotional_intensity
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                previous_target,
                previous_relation,
                previous_emotion,
                previous_intensity,
            ),
        )

        self.cursor.execute(
            """
            UPDATE memory_facts
            SET
                relation = ?,
                emotion = ?,
                emotional_intensity = ?,
                temporal_context = COALESCE(?, temporal_context),
                negation = COALESCE(?, negation)
            WHERE memory_id = ?
              AND LOWER(target) = LOWER(?)
            """,
            (
                relation,
                emotion,
                emotional_intensity,
                temporal_context,
                int(negation) if negation is not None else None,
                memory_id,
                target,
            ),
        )

        self.connection.commit()

        return True

    def update_existing_memory(self, existing_memory, new_memory):
        for new_fact in new_memory.facts:
            for existing_fact in existing_memory.facts:

                if new_fact.target != existing_fact.target:
                    continue

                if (
                    new_fact.relation == existing_fact.relation
                    and new_fact.emotion == existing_fact.emotion
                ):
                    continue

                updated = self.update_memory_fact(
                    memory_id=existing_memory.id,
                    target=new_fact.target,
                    relation=new_fact.relation,
                    emotion=new_fact.emotion,
                    emotional_intensity=new_fact.emotional_intensity,
                )

                if updated:
                    print(
                        f"[MEMORY] Fato atualizado: "
                        f"{new_fact.target} -> {new_fact.relation}"
                    )

        return self.find_memory(existing_memory)

    def add_memory_fact(
        self,
        memory_id,
        target,
        relation,
        emotion,
        emotional_intensity,
        temporal_context="current",
        negation=False,
    ):
        self.cursor.execute(
            """
            INSERT INTO memory_facts (
                memory_id,
                target,
                relation,
                emotion,
                emotional_intensity,
                temporal_context,
                negation
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                target,
                relation,
                emotion,
                emotional_intensity,
                temporal_context,
                int(negation),
            ),
        )

        self.connection.commit()

        return True

    def find_memory_by_fact_target(self, target):
        self.cursor.execute(
            """
            SELECT DISTINCT m.id
            FROM memories m
            INNER JOIN memory_facts f
                ON m.id = f.memory_id
            WHERE LOWER(f.target) = LOWER(?)
            """,
            (target,),
        )

        row = self.cursor.fetchone()

        if row is None:
            return None

        memory_id = row[0]

        self.cursor.execute(
            """
            SELECT content, memory_type, importance, emotion, emotional_intensity
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )

        memory_row = self.cursor.fetchone()

        if memory_row is None:
            return None

        facts = self.get_memory_facts(memory_id)

        memory = Memory(
            content=memory_row[0],
            memory_type=memory_row[1],
            importance=memory_row[2],
            emotion=memory_row[3],
            emotional_intensity=memory_row[4],
            facts=facts,
        )

        memory.id = memory_id

        return memory

    def get_memory_fact_history(self, memory_id):
        self.cursor.execute(
            """
            SELECT
                target,
                relation,
                emotion,
                emotional_intensity,
                changed_at
            FROM memory_fact_history
            WHERE memory_id = ?
            ORDER BY changed_at ASC
            """,
            (memory_id,),
        )

        rows = self.cursor.fetchall()

        return [
            {
                "target": row[0],
                "relation": row[1],
                "emotion": row[2],
                "emotional_intensity": row[3],
                "changed_at": row[4],
            }
            for row in rows
        ]
    
    def update_memory_content(self, memory_id, content):
        self.cursor.execute(
            """
            UPDATE memories
            SET content = ?
            WHERE id = ?
            """,
            (
                content,
                memory_id,
            ),
        )

        self.connection.commit()
        return self.cursor.rowcount > 0