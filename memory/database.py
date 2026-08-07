import sqlite3

from memory.memory import Memory


class Database:

    def __init__(self):

        self.connection = sqlite3.connect("davios.db")
        self.cursor = self.connection.cursor()

    def initialize(self):

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories(

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                content TEXT,

                memory_type TEXT,

                importance INTEGER

            )
        """)

        self.connection.commit()

    def save_memory(self, memory):

        self.cursor.execute(
            "INSERT INTO memories(content, memory_type, importance) VALUES (?, ?, ?)",
            (memory.content, memory.memory_type, memory.importance)
        )

        self.connection.commit()

    def memory_exists(self, memory):

        self.cursor.execute(
            "SELECT content, memory_type, importance FROM memories WHERE content = ?",
            (memory.content,)
        )

        result = self.cursor.fetchone()


        return result is not None

    def get_memories(self):

        self.cursor.execute(
            "SELECT * FROM memories"
        )

        rows = self.cursor.fetchall()

        memories = [
            Memory(
                content=row[1],
                memory_type=row[2],
                importance=row[3]
            )
            for row in rows
        ]

        return memories