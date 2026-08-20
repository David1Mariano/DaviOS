class Memory:
    def __init__(
        self,
        content,
        memory_type,
        importance,
        emotion,
        emotional_intensity,
        facts=None
    ):
        self.id = None
        self.content = content
        self.memory_type = memory_type
        self.importance = importance
        self.emotion = emotion
        self.emotional_intensity = emotional_intensity
        self.facts = facts if facts is not None else []

    def __repr__(self):
        return (
            f"Memory(content='{self.content}', memory_type='{self.memory_type}', "
            f"importance={self.importance}, emotion='{self.emotion}', "
            f"emotional_intensity={self.emotional_intensity}, "
            f"facts={self.facts})"
        )