import logging

logger = logging.getLogger("davios.memory.emotion")


class EmotionAnalyzer:

    def analyze(self, text, context):

        text = text.lower()
        context = context or {}

        logger.debug("[EMOTION] text=%s context=%s", text, context)
        if any(word in text.split() for word in ("odeio", "detesto")):
            return {
                "emotion": "dislike",
                "emotional_intensity": 5,
            }

        if context.get("negation_applies"):
            return {
                "emotion": "dislike",
                "emotional_intensity": 5,
            }

        if any(word in text.split() for word in ("adoro", "curto")):
            return {
                "emotion": "happiness",
                "emotional_intensity": 8,
            }

        if any(word in text.split() for word in ("favorita", "favoritas", "favorito", "favoritos", "prefiro")):
            return {
                "emotion": "happiness",
                "emotional_intensity": 6,
            }

        if "amo" in text:
            return {
                "emotion": "happiness",
                "emotional_intensity": 7,
            }

        if any(word in text.split() for word in ("gosto", "goste", "gostar")):
            return {
                "emotion": "happiness",
                "emotional_intensity": 5,
            }

        return {
            "emotion": "neutral",
            "emotional_intensity": 0,
        }