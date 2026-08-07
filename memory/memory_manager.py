from memory.database import Database


class MemoryManager:

    def __init__(self):

        self.database = Database()
        self.database.initialize()

    def remember(self, memory):

        if not self.database.memory_exists(memory):

            self.database.save_memory(memory)

    def recall(self):

        return self.database.get_memories()