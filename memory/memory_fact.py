from dataclasses import dataclass


@dataclass
class MemoryFact:
    target: str
    relation: str
    emotion: str
    emotional_intensity: float
    temporal_context: str = "current"
    negation: bool = False