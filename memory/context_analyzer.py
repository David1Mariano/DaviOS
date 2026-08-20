class ContextAnalyzer:

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

        negation = False
        negated_word = None

        for index, word in enumerate(words):

            if word == "não":

                negation = True

                if index + 1 < len(words):
                    negated_word = words[index + 1]

                break

        return {
            "text": text,
            "negation": negation,
            "negated_word": negated_word
        }