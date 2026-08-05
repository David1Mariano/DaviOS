import sqlite3


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

    def get_memories(self):

        self.cursor.execute(
            "SELECT * FROM memories"
        )

        return self.cursor.fetchall()