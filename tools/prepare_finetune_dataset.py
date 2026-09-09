"""Preparacao de dataset de fine-tuning local.

Coleta exemplos de conversa + feedback explicito do usuario e gera un
dataset JSONL local (nao depende de servicos externos).

NOTA: Este pipeline APENAS prepara el dataset. No ejecuta fine-tuning
ni altera pesos del modelo. El entrenamiento real es una etapa separada,
fuera del alcance de conversacion en tiempo real.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("davios.finetune")


@dataclass
class Example:
    """Un ejemplo conversacional para fine-tuning."""

    user: str
    assistant: str
    feedback: str = "neutral"  # 'good' | 'bad' | 'neutral'
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ],
            "feedback": self.feedback,
            "timestamp": self.timestamp,
        }


class FinetuneDatasetCollector:
    """Guarda ejemplos que el usuario marca explicitamente como buenos/malos.

    Solo se registra con feedback del usuario: nunca recolecta conversacion
    arbitraria sin consentimiento del proceso de feedback.
    """

    def __init__(self, dataset_path: str = "data/finetune.jsonl"):
        self.path = Path(dataset_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        user: str,
        assistant: str,
        feedback: str = "neutral",
    ) -> Example:
        if feedback not in ("good", "bad", "neutral"):
            raise ValueError("Feedback debe ser 'good', 'bad' o 'neutral'.")
        example = Example(user=user, assistant=assistant, feedback=feedback)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
        logger.info("[FINETUNE] ejemplo registrado: feedback=%s", feedback)
        return example

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as fh:
            return sum(1 for _ in fh if _.strip())

    def export(self, output_path: Optional[str] = None) -> str:
        """Devuelve el dataset como texto JSONL (o lo copia a output_path)."""
        dest = Path(output_path) if output_path else self.path
        if dest != self.path:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.path.read_bytes())
        return str(dest)

    def stats(self) -> dict[str, int]:
        counts = {"good": 0, "bad": 0, "neutral": 0}
        if not self.path.exists():
            return counts
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fb = data.get("feedback", "neutral")
                if fb in counts:
                    counts[fb] += 1
        return counts