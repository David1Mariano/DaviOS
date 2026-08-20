class EmotionAnalyzer:

    def analyze(self, text, context):

        text = text.lower()
        context = context or {}

        print(f"EMOTION TEXT: {text}")
        print(f"EMOTION CONTEXT: {context}")
        if context.get("negation") and context.get("negated_word") == "gosto":
            return {
                "emotion": "dislike",
                "emotional_intensity": 5,
            }

        if "adoro" in text:
            return {
                "emotion": "happiness",
                "emotional_intensity": 8,
            }

        if "amo" in text:
            return {
                "emotion": "happiness",
                "emotional_intensity": 7,
            }

        if "gosto" in text:
            return {
                "emotion": "happiness",
                "emotional_intensity": 5,
            }

        return {
            "emotion": "neutral",
            "emotional_intensity": 0,
        }