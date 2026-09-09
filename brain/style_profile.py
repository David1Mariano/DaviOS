"""StyleProfile do DaviOS.

Analisa as mensagens do usuario e extrai padroes de linguagem
(abreviacoes, gírias, formalidade, emojis, tamanho de mensagem, etc).
O perfil e persistido no SQLite e evolui com o uso.

O LLM recebe o perfil no prompt para espelhar o estilo do usuario.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("davios.style")


# Abreviacoes comuns em portugues informal e suas expansoes
KNOWN_ABBREVIATIONS = {
    "vc": "voce",
    "tb": "tambem",
    "tbm": "tambem",
    "tmb": "tambem",
    "pq": "porque",
    "q": "que",
    "n": "nao",
    "num": "nao",
    "kd": "cadê",
    "blz": "beleza",
    "vlw": "valeu",
    "flw": "falou",
    "mds": "meu deus",
    "msg": "mensagem",
    "mt": "muito",
    "mta": "muita",
    "mto": "muito",
    "bjs": "beijos",
    "abs": "abraco",
    "t+": "ate mais",
    "aki": "aqui",
    "aq": "aqui",
    "p": "para",
    "pro": "para o",
    "pra": "para a",
    "d": "de",
    "eh": "eh",
    "to": "estou",
    "tou": "estou",
    "ta": "esta",
    "tamo": "estamos",
    "ne": "ne",
    "rs": "risos",
    "kk": "risos",
    "haha": "risos",
    "hehe": "risos",
}


@dataclass
class StyleProfile:
    """Perfil de estilo de fala do usuario.

    Os pesos usam media movel com decaimento exponencial para dar
    mais importancia a mensagens recentes.
    """

    # Contadores normalizados (0.0 a 1.0)
    formality: float = 0.5  # 0.0 = muito informal, 1.0 = muito formal
    emoji_usage: float = 0.0  # frequencia de emojis
    avg_message_length: float = 50.0  # tamanho medio em caracteres
    abbreviation_ratio: float = 0.0  # proporcao de abreviacoes
    uses_punctuation: float = 0.8  # frequencia de pontuacao padrao
    uses_caps: float = 0.0  # frequencia de maiusculas

    # Dados brutos
    total_messages: int = 0
    common_abbreviations: dict[str, int] = field(default_factory=dict)
    common_words: dict[str, int] = field(default_factory=dict)
    common_emojis: dict[str, int] = field(default_factory=dict)

    # Confianca: cresce com mais dados (max 1.0)
    confidence: float = 0.0

    # Limite minimo de mensagens para aplicar estilo com conviccao
    MIN_MESSAGES_FOR_STYLE: int = 10

    def update(self, message: str) -> None:
        """Atualiza o perfil com uma nova mensagem do usuario.

        Usa media movel com peso maior para mensagens recentes.
        """
        self.total_messages += 1

        # Peso da mensagem recente (decaimento exponencial)
        recent_weight = 0.3
        old_weight = 1.0 - recent_weight

        # Atualiza tamanho medio da mensagem
        msg_len = len(message.strip())
        self.avg_message_length = (
            self.avg_message_length * old_weight + msg_len * recent_weight
        )

        # Analisa abreviacoes
        words = message.lower().split()
        abbrev_count = 0
        for word in words:
            clean = re.sub(r"[^\w]", "", word)
            if clean in KNOWN_ABBREVIATIONS:
                abbrev_count += 1
                self.common_abbreviations[clean] = (
                    self.common_abbreviations.get(clean, 0) + 1
                )
            elif len(clean) > 2:
                self.common_words[clean] = self.common_words.get(clean, 0) + 1

        # Proporcao de abreviacoes
        abbrev_ratio = abbrev_count / max(len(words), 1)
        self.abbreviation_ratio = (
            self.abbreviation_ratio * old_weight + abbrev_ratio * recent_weight
        )

        # Analisa emojis
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d\u23cf\u23e9\u231a\ufe0f\u3030"
            "]+",
            flags=re.UNICODE,
        )
        emojis_found = emoji_pattern.findall(message)
        for emoji in emojis_found:
            self.common_emojis[emoji] = self.common_emojis.get(emoji, 0) + 1

        emoji_freq = len(emojis_found) / max(len(words), 1)
        self.emoji_usage = (
            self.emoji_usage * old_weight + emoji_freq * recent_weight
        )

        # Analisa formalidade
        has_punctuation = bool(re.search(r"[.!?;,]", message))
        punct_score = 1.0 if has_punctuation else 0.3
        self.uses_punctuation = (
            self.uses_punctuation * old_weight + punct_score * recent_weight
        )

        # Uso de maiusculas (excluindo primeira letra)
        if len(message) > 1:
            caps_count = sum(1 for c in message[1:] if c.isupper())
            caps_ratio = caps_count / max(len(message) - 1, 1)
            self.uses_caps = self.uses_caps * old_weight + caps_ratio * recent_weight

        # Formalidade: combina pontuacao, ausencia de abreviacoes, tamanho
        formality_score = (
            0.4 * punct_score
            + 0.3 * (1.0 - self.abbreviation_ratio)
            + 0.3 * min(self.avg_message_length / 100.0, 1.0)
        )
        self.formality = self.formality * old_weight + formality_score * recent_weight

        # Confianca: cresce com mais mensagens
        self.confidence = min(
            self.total_messages / self.MIN_MESSAGES_FOR_STYLE, 1.0
        )

        logger.debug(
            "[STYLE] msg=%d formality=%.2f emoji=%.2f abbrev=%.2f conf=%.2f",
            self.total_messages,
            self.formality,
            self.emoji_usage,
            self.abbreviation_ratio,
            self.confidence,
        )

    def to_instruction(self) -> str:
        """Gera instrucao para o LLM baseada no perfil.

        Retorna string vazia se nao ha dados suficientes (confidence baixa).
        """
        if self.total_messages < 3:
            return ""

        parts: list[str] = []

        # Nivel de formalidade
        if self.formality < 0.3:
            parts.append("O usuario escreve de forma bem informal.")
        elif self.formality < 0.6:
            parts.append("O usuario escreve de forma neutra.")
        else:
            parts.append("O usuario escreve de forma formal.")

        # Abreviacoes
        if self.abbreviation_ratio > 0.15:
            top_abbrevs = sorted(
                self.common_abbreviations.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            abbrev_list = ", ".join(f"'{a}'" for a, _ in top_abbrevs)
            parts.append(
                f"O usuario usa abreviacoes com frequencia: {abbrev_list}."
            )

        # Emojis
        if self.emoji_usage > 0.1:
            top_emojis = sorted(
                self.common_emojis.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            emoji_list = " ".join(e for e, _ in top_emojis)
            parts.append(f"O usuario usa emojis: {emoji_list}.")

        # Tamanho das mensagens
        if self.avg_message_length < 20:
            parts.append("O usuario prefere mensagens curtas.")
        elif self.avg_message_length > 80:
            parts.append("O usuario escreve mensagens mais longas e detalhadas.")

        # Pontuacao
        if self.uses_punctuation < 0.4:
            parts.append("O usuario raramente usa pontuacao.")

        # Maiusculas
        if self.uses_caps > 0.3:
            parts.append("O usuario usa maiusculas com frequencia.")

        if not parts:
            return ""

        return "Estilo de fala do usuario: " + " ".join(parts)

    def should_apply_style(self) -> bool:
        """True se ha dados suficientes para aplicar o estilo com conviccao."""
        return self.total_messages >= self.MIN_MESSAGES_FOR_STYLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "formality": round(self.formality, 3),
            "emoji_usage": round(self.emoji_usage, 3),
            "avg_message_length": round(self.avg_message_length, 1),
            "abbreviation_ratio": round(self.abbreviation_ratio, 3),
            "uses_punctuation": round(self.uses_punctuation, 3),
            "uses_caps": round(self.uses_caps, 3),
            "total_messages": self.total_messages,
            "common_abbreviations": dict(self.common_abbreviations),
            "common_words": dict(
                sorted(self.common_words.items(), key=lambda x: x[1], reverse=True)[:20]
            ),
            "common_emojis": dict(self.common_emojis),
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StyleProfile:
        """Reconstroi perfil a partir de dicionario persistido."""
        profile = cls()
        profile.formality = data.get("formality", 0.5)
        profile.emoji_usage = data.get("emoji_usage", 0.0)
        profile.avg_message_length = data.get("avg_message_length", 50.0)
        profile.abbreviation_ratio = data.get("abbreviation_ratio", 0.0)
        profile.uses_punctuation = data.get("uses_punctuation", 0.8)
        profile.uses_caps = data.get("uses_caps", 0.0)
        profile.total_messages = data.get("total_messages", 0)
        profile.common_abbreviations = data.get("common_abbreviations", {})
        profile.common_words = data.get("common_words", {})
        profile.common_emojis = data.get("common_emojis", {})
        profile.confidence = data.get("confidence", 0.0)
        return profile


def apply_style_substitutions(
    text: str,
    profile: StyleProfile,
    max_substitutions: int = 3,
) -> str:
    """Aplica substituicoes leves baseadas no perfil do usuario.

    So substitui se o usuario usa a abreviacao com frequencia (dados reais).
    Nunca forca gírias que o usuario nunca usou.
    """
    if not profile.should_apply_style():
        return text

    # Monta dicionario de substituicoes baseado no que o usuario realmente usa
    substitutions: dict[str, str] = {}
    for abbrev, full in KNOWN_ABBREVIATIONS.items():
        if (
            abbrev in profile.common_abbreviations
            and profile.common_abbreviations[abbrev] >= 2
        ):
            substitutions[full] = abbrev

    if not substitutions:
        return text

    # Aplica substituicoes (limitando para nao exagerar)
    result = text
    count = 0
    for full, abbrev in substitutions.items():
        pattern = r"\b" + re.escape(full) + r"\b"
        result, n = re.subn(pattern, abbrev, result, flags=re.IGNORECASE)
        count += n
        if count >= max_substitutions:
            break

    return result

