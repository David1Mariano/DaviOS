from __future__ import annotations

import logging
import re

from memory.memory_fact import MemoryFact

logger = logging.getLogger("davios.memory.interpreter")


class MemoryInterpreter:

    PREFERENCE_WORDS = {
        "gosto", "goste", "gostar", "adoro", "amo", "curto", "odeio", "detesto",
        "favorita", "favoritas", "favorito", "favoritos", "prefiro",
    }
    TRAILING_DISCOURSE = {"na", "real", "verdade", "mesmo"}
    # Palavras que conectam o alvo à palavra de preferência em frases como
    # "pizza é uma das minhas comidas favoritas". Nunca fazem parte do alvo.
    PRE_TARGET_FILLERS = {
        "e", "é", "uma", "um", "uns", "umas", "das", "dos", "do", "da", "de",
        "a", "o", "as", "os", "minhas", "meus", "minha", "meu",
        "comidas", "coisas", "coisa", "que", "eu",
        "mais", "nao", "não", "algo", "agora", "hoje", "atualmente", "antes",
        "antigamente",
    }
    STOP_TARGET_WORDS = {
        "eu", "meu", "minha", "meus", "minhas", "voce", "muito", "mais",
        "que", "e", "é", "uma", "um", "o", "a", "os", "as", "de", "do", "da",
        "no", "na", "em", "com", "para", "por", "tambem", "agora", "hoje",
        "mas", "porem", "entao", "muito", "bem", "nao", "não",
    }

    def interpret(self, text, context, emotions):

        personal_markers = [
            "eu",
            "meu",
            "minha",
            "me",
            "comigo",
            "gosto",
            "adoro",
            "amo",
            "odeio",
            "não gosto",
        ]

        text_lower = text.lower()

        personal_relevance = 0.0

        for marker in personal_markers:
            if marker in text_lower:
                personal_relevance += 0.2

        personal_relevance = min(personal_relevance, 1.0)

        emotional_relevance = 0.0

        if emotions:
            strongest_emotion = max(
                emotions,
                key=lambda emotion: emotion["emotional_intensity"]
            )

            emotional_relevance = (
                strongest_emotion["emotional_intensity"] / 10
            )

        facts = []

        # Detect "meu nome e X" / "me chamo X" patterns
        name_fact = self._extract_name_fact(text_lower)
        if name_fact:
            facts.append(name_fact)
            personal_relevance = max(personal_relevance, 0.6)

        # Fatos estruturados declarativos (trabalho, estudo, residencia, posse)
        for statement_fact in self._extract_statement_facts(text_lower):
            facts.append(statement_fact)
            personal_relevance = max(personal_relevance, 0.6)

        for emotion_data in emotions or []:

            target = self.extract_emotion_target(
                emotion_data["text"],
                emotion_data["emotion"]
            )

            if not target or not self._is_valid_fact_target(target):
                # Alvos sem conteudo semantico (pronomes como "te"/"me",
                # interjeicoes como "kkk") NUNCA viram fatos — ex: "te amo"
                # nao pode gerar like(target="te").
                continue

            relation = self.determine_relation(
                emotion_data["emotion"],
                emotion_data["context"]
            )

            temporal_context = self.detect_temporal_context(
                emotion_data["text"]
            )

            negation = emotion_data["context"].get("negation_applies", False)
            uncertain = emotion_data["context"].get("uncertain", False)

            facts.append(
                MemoryFact(
                    target=target,
                    relation=relation,
                    emotion=emotion_data["emotion"],
                    emotional_intensity=emotion_data["emotional_intensity"],
                    temporal_context=temporal_context,
                    negation=negation,
                    confidence=0.65 if uncertain else 0.7,
                    subject="user",
                    source="user_statement",
                )
            )

        logger.debug("[MEMORY] interpreted facts=%d", len(facts))

        return {
            "memory_candidate": personal_relevance >= 0.4,
            "facts": facts,
            "personal_relevance": personal_relevance,
            "emotional_relevance": emotional_relevance,
        }

    def _extract_name_fact(self, text_lower: str):
        """Detecta padroes como 'meu nome e X' e cria um fato de identidade."""
        match = re.search(r"meu nome (?:e|é)\s+(\w+)", text_lower)
        if not match:
            match = re.search(r"me chamo\s+(\w+)", text_lower)
        if not match:
            return None
        name = match.group(1)
        return MemoryFact(
            target="nome",
            relation="identity",
            emotion="neutral",
            emotional_intensity=3,
            temporal_context="current",
            negation=False,
            confidence=0.8,
            subject="user",
            value=name,
            source="user_statement",
            fact_type="identity",
        )

    # ------------------------------------------------------------------
    # Fatos estruturados declarativos
    # ------------------------------------------------------------------

    # (regex do verbo, relacao) — ordem importa: mais especifico primeiro
    STATEMENT_PATTERNS = (
        (r"(?:estou|to)\s+trabalhando\s+(?:no|na|em|numa|num)\s+([a-z0-9][\w \-]*)", "working_on"),
        (r"trabalho\s+(?:no|na|em|numa|num)\s+([a-z0-9][\w \-]*)", "working_on"),
        (r"meu\s+(?:projeto|trabalho)\s+(?:se\s+chama|e|é)\s+([a-z0-9][\w \-]*)", "working_on"),
        (r"(?:estou|to)\s+fazendo\s+(?:um\s+)?(?:projeto|app|aplicativo|sistema)\s+(?:chamado|chamada|do|da)?\s*([a-z0-9][\w \-]*)", "working_on"),
        (r"estou\s+desenvolvendo\s+(?:o|a|um|uma)?\s*([a-z0-9][\w \-]*)", "working_on"),
        (r"estou\s+estudando\s+([a-z0-9][\w \-]*)", "studies"),
        (r"eu\s+estudo\s+([a-z0-9][\w \-]*)", "studies"),
        (r"moro\s+em\s+([a-z0-9][\w \-]*)", "lives_in"),
        (r"eu\s+tenho\s+(?:um|uma|um(a)?)?\s*([a-z0-9][\w \-]*)", "has"),
    )

    def _extract_statement_facts(self, text_lower: str) -> list[MemoryFact]:
        """Extrai fatos declarativos (working_on, studies, lives_in, has).

        Complementa a extracao por emocao: cobre frases sem palavra de
        preferencia, como 'Estou trabalhando no NetOptimizer'.
        """
        facts: list[MemoryFact] = []
        negation = bool(
            re.search(r"\b(nao|não|nunca|jamais)\b", text_lower)
        )
        for pattern, relation in self.STATEMENT_PATTERNS:
            match = re.search(pattern, text_lower)
            if not match:
                continue
            target = self._clean_target(match.group(1))
            if not target or not self._is_valid_fact_target(target):
                continue
            facts.append(
                MemoryFact(
                    target=target,
                    relation=relation,
                    emotion="neutral",
                    emotional_intensity=0.0,
                    temporal_context="current",
                    negation=negation,
                    confidence=0.75,
                    subject="user",
                    source="user_statement",
                    fact_type="statement",
                )
            )
        return facts

    # Alvos de fato precisam ter conteudo semantico. Pronomes e
    # interjeicoes nunca viram fatos (evita like(target="te") de "te amo").
    _FACT_TARGET_PRONOUNS = frozenset({
        "te", "me", "nos", "lhe", "voce", "vc", "isso", "isto", "aquilo",
        "ele", "ela", "eles", "elas", "mim", "ti", "si", "nada", "tudo",
        "algo", "alguem", "nao", "ai", "aqui", "la",
    })

    _FACT_TARGET_NOISE = frozenset({
        "kkk", "kkkk", "kkkkk", "kkkkkk", "rs", "rsrs", "haha", "hahaha",
        "lol", "kkkl", "muito", "mais", "bem", "agora", "hoje",
    })

    @classmethod
    def _is_valid_fact_target(cls, target) -> bool:
        """Alvo de fato exige >= 3 caracteres, nao-pronome e
        nao-interjeicao. Nunca rejeita preferencias reais (pizza, python)."""
        t = " ".join(str(target or "").split()).casefold()
        if len(t) < 3:
            return False
        if t in cls._FACT_TARGET_PRONOUNS or t in cls._FACT_TARGET_NOISE:
            return False
        return True

    def _clean_target(self, raw: str) -> str:
        """Remove conectivos finais e palavras vazias do alvo extraido."""
        words = raw.strip().split()
        cleaned: list[str] = []
        for word in words:
            if word in self.STOP_TARGET_WORDS:
                break
            cleaned.append(word)
        while cleaned and cleaned[-1] in self.TRAILING_DISCOURSE:
            cleaned.pop()
        target = " ".join(cleaned).strip(" .,!?")
        return target or ""

    def extract_emotion_target(self, text, emotion):
        import re

        words = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
        preference_index = next(
            (index for index, word in enumerate(words)
             if word in self.PREFERENCE_WORDS),
            None,
        )
        if preference_index is None:
            return None

        target_start = preference_index + 1
        if target_start < len(words) and words[target_start] in {"muito", "mais"}:
            target_start += 1
        if target_start < len(words) and words[target_start] in {"de", "do", "da"}:
            target_start += 1

        target_words = []
        for word in words[target_start:]:
            if word in self.TRAILING_DISCOURSE:
                break
            if word in {"agora", "atualmente", "hoje", "antes", "antigamente", "mais"}:
                continue
            target_words.append(word)

        if not target_words:
            # Alvo antes da palavra de preferência:
            # "pizza é uma das minhas comidas favoritas" -> pizza
            target_words = self._extract_leading_target(words, preference_index)

        return " ".join(target_words) or None

    def _extract_leading_target(self, words, preference_index):
        for word in reversed(words[:preference_index]):
            if word in self.PRE_TARGET_FILLERS:
                continue
            return [word]
        return []

    def determine_relation(self, emotion, context):

        if emotion == "dislike":
            return "dislike"

        if emotion == "happiness":
            return "like"

        return "unknown"

    def detect_temporal_context(self, text):

        text_lower = text.lower()

        if any(
            word in text_lower
            for word in ["agora", "atualmente", "hoje"]
        ):
            return "current"

        if any(
            word in text_lower
            for word in ["antes", "antigamente"]
        ):
            return "past"

        return "current"