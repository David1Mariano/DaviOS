class ContextAnalyzer:

    @staticmethod
    def _normalize(text):
        import unicodedata

        return "".join(
            character
            for character in unicodedata.normalize("NFD", text.lower())
            if unicodedata.category(character) != "Mn"
        )

    def analyze(self, text):

        text = text.lower()

        if "mas" in text:

            parts = text.split("mas", 1)

            return {
                "has_contrast": True,
                "parts": [
                    self.analyze_part(parts[0].strip()),
                    self.analyze_part(parts[1].strip())
                ]
            }

        return {
            "has_contrast": False,
            "parts": [
                self.analyze_part(text)
            ]
        }

    def analyze_part(self, text):

        words = text.split()
        normalized_words = [self._normalize(word) for word in words]
        preference_words = {
            "gosto", "goste", "gostar", "adoro", "amo", "curto", "detesto", "odeio"
        }

        negation = False
        negated_word = None
        negation_applies = False

        for index, word in enumerate(normalized_words):

            if word in {"nao", "nunca", "jamais"}:

                negation = True

                if index + 1 < len(normalized_words):
                    negated_word = normalized_words[index + 1]

                negation_applies = (
                    index + 1 < len(normalized_words)
                    and normalized_words[index + 1] in preference_words
                )

                break

        normalized_text = self._normalize(text)
        uncertain = any(
            marker in normalized_text
            for marker in (
                "eu acho que", "acho que", "talvez", "nao tenho certeza"
            )
        )
        temporal_words = {
            "agora", "mais", "atualmente", "hoje", "antes", "antigamente"
        }

        return {
            "text": text,
            "negation": negation,
            "negation_applies": negation_applies,
            "negated_word": negated_word,
            "uncertain": uncertain,
            "temporal_words": [
                word for word in normalized_words if word in temporal_words
            ],
        }