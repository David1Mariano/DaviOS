from memory.memory_fact import MemoryFact


class MemoryInterpreter:

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

            negation = emotion_data["context"].get(
                "negation",
                False
            )

            facts.append(
                MemoryFact(
                    target=target,
                    relation=relation,
                    emotion=emotion_data["emotion"],
                    emotional_intensity=emotion_data["emotional_intensity"],
                    temporal_context=temporal_context,
                    negation=negation,
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

    def extract_emotion_target(self, text, emotion):

        words = (
            text.lower()
            .replace(",", "")
            .replace(".", "")
            .split()
        )

        emotion_words = {
            "gosto",
            "adoro",
            "amo",
            "odeio",
            "detesto",
        }

        context_words = {
            "eu",
            "não",
            "de",
            "do",
            "da",
            "agora",
            "atualmente",
            "hoje",
            "antes",
            "sempre",
            "nunca",
            "mais",
            "ainda",
        }

        filtered_words = [
            word
            for word in words
            if word not in emotion_words
            and word not in context_words
        ]

        if filtered_words:
            return " ".join(filtered_words)

        return None

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