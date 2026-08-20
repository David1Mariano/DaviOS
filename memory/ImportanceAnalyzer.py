from memory.memory import Memory

class ImportanceAnalyzer:
    """
    Responsável por calcular e ponderar o nível de relevância 
    de memórias no DaviOS (usando heurísticas até o Reasoning ser implementado).
    """

    def __init__(self, custom_weights: dict = None):
        self.weights = custom_weights or {
            "type_weight": 0.5,
            "keyword_weight": 0.5
        }

    def analyze(self, memory: Memory) -> float:
        """
        Analisa o objeto Memory e retorna um score de importância de 0.0 a 1.0.
        """
        score = 0.3  # Base padrão

        # 1. Analisa com base no tipo de memória
        if memory.memory_type == "preference":
            score += 0.4  # Preferências moldam a personalidade do DaviOS
        elif memory.memory_type == "episodic":
            score += 0.2

        # 2. Analisa palavras-chave fortes no conteúdo (heurística temporária)
        content_lower = memory.content.lower()
        strong_keywords = ["adoro", "odeio", "amo", "nunca", "sempre", "gosto"]
        
        if any(word in content_lower for word in strong_keywords):
            score += 0.3

        # Garante que o valor fique entre 0.0 e 1.0
        final_score = min(1.0, max(0.0, score))
        
        # Atualiza o campo de importância dentro da memória
        memory.importance = round(final_score, 2)
        return memory.importance