from memory.database import Database


class MemoryManager:

    def __init__(self):

        self.database = Database()
        self.database.initialize()

    def remember(self, content):

        self.database.save_memory(content)

    def recall(self):

        return self.database.get_memories()