"""Testes do filtro de validade de targets do MemoryInterpreter.

Contexto: "te amo" gerava fato like(target="te") — o fato entrava no prompt
como "informacao do usuario" e o modelo afirmava coisas que o usuario nunca
disse ("Você disse q gosta de me fazer rire!"). Targets sem conteudo
semantico (pronomes, interjeicoes, uma palavra so) devem ser rejeitados,
mantendo intactas as preferencias reais.
"""

from memory.context_analyzer import ContextAnalyzer
from memory.emotion_analyzer import EmotionAnalyzer
from memory.memory_interpreter import MemoryInterpreter


def _interpret(text: str):
    interpreter = MemoryInterpreter()
    context = ContextAnalyzer().analyze(text)
    emotions = []
    for part in context.get("parts", []):
        emotion = EmotionAnalyzer().analyze(part["text"], part)
        emotions.append(
            {
                "text": part["text"],
                "emotion": emotion["emotion"],
                "emotional_intensity": emotion["emotional_intensity"],
                "context": part,
            }
        )
    return interpreter.interpret(text, context, emotions)


class TestCasualInputsProduceNoFacts:
    """Entradas casuais/afetivas nao devem gerar fatos com target vazio."""

    def test_te_amo_produces_no_fact(self):
        interpretation = _interpret("te amo")
        targets = [f.target for f in interpretation["facts"]]
        assert "te" not in targets
        assert interpretation["facts"] == []

    def test_te_adoro_produces_no_fact(self):
        interpretation = _interpret("te adoro")
        targets = [f.target for f in interpretation["facts"]]
        assert "te" not in targets

    def test_meu_deus_produces_no_fact(self):
        interpretation = _interpret("meu deus")
        assert interpretation["facts"] == []

    def test_kkk_produces_no_fact(self):
        interpretation = _interpret("kkk")
        assert interpretation["facts"] == []

    def test_kkkk_produces_no_fact(self):
        interpretation = _interpret("kkkkk")
        assert interpretation["facts"] == []

    def test_boa_produces_no_fact(self):
        interpretation = _interpret("boa")
        assert interpretation["facts"] == []

    def test_interjection_with_emoji_produces_no_fact(self):
        interpretation = _interpret("kkk 😄 boa!")
        assert interpretation["facts"] == []


class TestValidPreferencesStillWork:
    """Regressao: preferencias reais continuam gerando fatos validos."""

    def test_gosto_de_pizza(self):
        interpretation = _interpret("eu gosto de pizza")
        targets = [f.target.lower() for f in interpretation["facts"]]
        assert "pizza" in targets

    def test_gosto_de_programacao(self):
        interpretation = _interpret("eu gosto de programacao")
        targets = [f.target.lower() for f in interpretation["facts"]]
        assert "programacao" in targets

    def test_real_target_with_emoji_still_extracted(self):
        """Emoji depois do alvo real nao impede a extracao."""
        interpretation = _interpret("eu gosto de pizza 😄")
        targets = [f.target.lower() for f in interpretation["facts"]]
        assert "pizza" in targets
