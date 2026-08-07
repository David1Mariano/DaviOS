class Memory:

    def __init__(self, content, memory_type, importance):

        self.content = content
        self.memory_type = memory_type
        self.importance = importance

    def __repr__(self):

        return f"Memory(content='{self.content}', memory_type='{self.memory_type}', importance={self.importance})"