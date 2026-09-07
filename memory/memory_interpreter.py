from memory.memory_fact import MemoryFact


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

        # Detect "meu nome e X" pattern
        name_fact = self._extract_name_fact(text_lower)
        if name_fact:
            facts.append(name_fact)
            personal_relevance = max(personal_relevance, 0.6)

        for emotion_data in emotions or []:

            target = self.extract_emotion_target(
                emotion_data["text"],
                emotion_data["emotion"]
            )

            if not target:
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

        print("=== INTERPRETED FACTS ===")
        for fact in facts:
            print(fact)

        return {
            "memory_candidate": personal_relevance >= 0.4,
            "facts": facts,
            "personal_relevance": personal_relevance,
            "emotional_relevance": emotional_relevance,
        }

    def _extract_name_fact(self, text_lower: str):
        """Detecta padroes como 'meu nome e X' e cria um fato de identidade."""
        import re
        match = re.search(r"meu nome e\s+(\w+)", text_lower)
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