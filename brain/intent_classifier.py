"""Classificador de intenção do ConversationEngine.

Baseado em regras leves (sem LLM): rápido, determinístico e offline.
Intenções: GREETING, FAREWELL, MEMORY_STATEMENT, MEMORY_QUERY,
GENERAL_QUESTION, OPINION, COMMAND, CONVERSATION, UNKNOWN.

Desde a etapa 6 também reconhece ORDEM explícita de troca de modelo
(MODEL_SWITCH) e a representa como `ModelSwitchRequest`. Este módulo NÃO
resolve o modelo e NÃO executa nada: conhecer o catálogo e trocar o runtime
é responsabilidade do ModelManager (`resolve_model` + `switch_active_model`).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Intent(str, Enum):
    GREETING = "greeting"
    FAREWELL = "farewell"
    MEMORY_STATEMENT = "memory_statement"
    MEMORY_QUERY = "memory_query"
    GENERAL_QUESTION = "general_question"
    OPINION = "opinion"
    COMMAND = "command"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"
    # --- Inteligencia conversacional (etapa 2) ---------------------------
    SOCIAL = "social"
    SOCIAL_RECIPROCAL = "social_reciprocal"
    FOLLOW_UP = "follow_up"
    FILE_REQUEST = "file_request"
    TIME_REQUEST = "time_request"
    # --- Gerenciamento de modelos (etapa 3) -----------------------------
    MODEL_STATUS = "model_status"
    MODEL_LIST = "model_list"
    MODEL_LIST_INSTALLED = "model_list_installed"
    MODEL_LIST_AVAILABLE = "model_list_available"
    MODEL_SWITCH = "model_switch"
    MODEL_DOWNLOAD = "model_download"
    MODEL_REQUIREMENTS = "model_requirements"
    MODEL_CONFIRM_DOWNLOAD = "model_confirm_download"
    MODEL_CANCEL = "model_cancel"


@dataclass(frozen=True)
class ModelSwitchRequest:
    """Pedido ESTRUTURADO de troca de modelo — intenção, não execução.

    Carrega apenas o que o usuário disse:

    - `action`: sempre ``"switch_model"`` nesta etapa. É o nome da ação
      pretendida, não uma chamada: nenhum objeto do runtime é tocado aqui;
    - `target`: o alvo em LINGUAGEM NATURAL, exatamente como o usuário o
      nomeou e já sem as palavras de preenchimento
      (``"troca para o Qwen 8B"`` -> ``"qwen 8b"``). Pode ser ``""`` quando a
      pessoa pediu uma troca sem dizer qual modelo ("quero um modelo grande");
    - `raw_text`: a mensagem original, preservada para log/resposta.

    O classificador NÃO resolve `target`: ele não conhece o catálogo e não
    sabe qual ID corresponde a "8b" — isso é `ModelManager.resolve_model()`
    contra o catálogo, e a troca real é `ModelManager.switch_active_model()`.
    """

    action: str = "switch_model"
    target: str = ""
    raw_text: str = ""

    def to_dict(self) -> dict[str, str]:
        """Versão serializável (útil para log e para os testes)."""
        return {
            "action": self.action,
            "target": self.target,
            "raw_text": self.raw_text,
        }


QUESTION_STARTERS = {
    "qual", "quais", "quem", "que", "o que", "oque", "como", "quando",
    "onde", "por que", "porque", "porque", "quanto", "quanta", "quantos",
    "quantas", "explique", "explica", "explicar", "me explique",
    "me ajuda", "me ajude", "ajuda", "ajude", "consegue", "pode",
}

MEMORY_QUERY_MARKERS = (
    "meu nome", "minha ", "meu ", "eu gosto", "eu nao gosto",
    "eu odeio", "eu adoro", "eu prefiro", "o que eu", "quem eu",
    "lembra", "lembre", "voce lembra", "ja falei",
)

OPINION_MARKERS = (
    "o que voce acha", "voce acha", "sua opiniao", "qual sua opiniao",
    "o que voce pensa", "voce concorda", "voce gosta de conversar",
    "o que voce sente",
)

COMMAND_MARKERS = (
    "abra ", "abrir ", "execute ", "executar ", "rode ", "rodar ",
    "apague ", "apagar ", "delete ", "instale ", "desligue ",
    "desligar ", "feche ", "fechar ",
)

MEMORY_STATEMENT_MARKERS = (
    "eu gosto", "eu nao gosto", "nao gosto mais", "eu odeio", "eu adoro",
    "eu amo", "eu detesto", "meu nome", "minha comida favorita",
    "meu favorito", "eu prefiro", "eu estudo", "estou estudando",
    "eu trabalho", "eu moro", "eu tenho",
)

# Perguntas gerais típicas de conhecimento
GENERAL_QUESTION_HINTS = (
    "o que e", "oque e", "o que sao", "por que o", "porque o",
    "como funciona", "explique", "qual a diferenca", "defina",
    "me ensine", "me ajude a", "significa",
)

# --- Model management markers (etapa 3) --------------------------------
MODEL_STATUS_MARKERS = (
    "qual modelo", "qual e o modelo", "que modelo", "modelo atual",
    "modelo que voce", "modelo que você", "modelo usado", "modelo em uso",
    "modelo ativo", "modelo que ta", "modelo que está",
)

MODEL_LIST_MARKERS = (
    "quais modelos", "que modelos", " modelos ", "modelos que",
    "modelos que eu", "lista de modelos", "modelos disponiveis",
    "modelos disponíveis", "modelos instlados", "modelos instalados",
    "modelos que tenho", "modelos no pc", "modelos no computador",
)

MODEL_SWITCH_MARKERS = (
    "troque", "mude", "altera", "alterar", "use", "utilize",
    "passa para", "passa pro", "trocar", "mudar",
)

MODEL_DOWNLOAD_MARKERS = (
    "baixe", "baixar", "download", "instale", "instalar",
    "preciso baixar", "quero baixar", "vou baixar",
    "preciso instalar", "quero instalar",
)

MODEL_REQUIREMENTS_MARKERS = (
    "quanto ocupa", "quanto gasta", "quantoake", "requisito",
    "requisitos", "precisa de", "preciso de", "ram necessaria",
    "ram necessária", "vram necessaria", "vram necessária",
    "compatível", "compativel", "roda no", "roda na",
    "consumo", "memoria necessaria", "memória necessária",
)

MODEL_CONFIRM_MARKERS = (
    "sim", "pode", "baixa", "pode baixar", "manda ver",
    "pode instalar", "baixar", "instalar", "confirmo",
    "concordo", "okay", "ok", "claro", "segue",
    "prosseguir", "continua",
)

MODEL_CANCEL_MARKERS = (
    "nao", "não", "cancela", "cancelar", "abort", "abortar",
    "parar", "pare", "não quero", "não quero baixar",
    "espera", "wait", "stop", "suspender", "pausa",
)

# Perguntas sobre modelos específicos
MODEL_SPECIFIC_MARKER = re.compile(
    r"(qwen|llama|mistral|gemma|solar|deepseek|dolphin|phi|samantha)"
    r"[\s_-]*"
    r"(\d+(?:\.\d+)?b|b)"
    r"[\s_-]*"
    r"(q\d[_k_]*[_\d]*|f16|f32|bf16|iq\d[_]*)?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Troca de modelo por linguagem natural (etapa 6)
# ---------------------------------------------------------------------------
# O extrator abaixo é DELIBERADAMENTE conservador: trocar de modelo altera
# estado do runtime, então só há intenção quando existe uma ORDEM explícita.
# Consulta ("qual modelo você está usando?"), dúvida ("acho que talvez...")
# e negação continuam sendo conversa — na dúvida, NÃO se executa.

# Verbos/expressões que expressam ORDEM de mudança de modelo.
MODEL_SWITCH_VERB_RE = re.compile(
    r"\b(?:"
    r"troca|trocar|troque|"
    r"muda|mudar|mude|"
    r"altera|alterar|altere|"
    r"usa|usar|use|"
    r"utiliza|utilizar|utilize|"
    r"passa|passar|passe|"
    r"quero\s+(?:usar|trocar|mudar|utilizar|colocar|ter)|"
    r"quero\s+(?:o|um|a|uma)?\s*modelo"
    r")\b"
)

# Sinais de consulta/dúvida/negação: nunca são ordem.
MODEL_SWITCH_HEDGE_RE = re.compile(
    r"\b(?:"
    r"acho|talvez|sera|poderia|podemos|posso|pode|consegue|consigo|"
    r"deveria|devo|gostaria|queria|quisera|recomend\w*|suger\w*|sugest\w*|"
    r"nao\s+quero|nao\s+preciso|nao\s+troca|nao\s+muda|nao\s+altera|"
    r"nao\s+sei|seria\s+bom|seria\s+melhor|voce\s+acha|vc\s+acha|"
    r"e\s+possivel|faz\s+sentido|vale\s+a\s+pena"
    r")\b"
)

# Palavras de preenchimento que NÃO fazem parte do alvo.
MODEL_SWITCH_FILLERS = frozenset({
    "por", "favor", "please", "para", "pro", "pra", "o", "a", "os", "as",
    "um", "uma", "uns", "umas", "meu", "minha", "seu", "sua", "esse", "essa",
    "este", "esta", "aquele", "aquela", "modelo", "model", "modelos", "llm",
    "local", "ativo", "atual", "de", "do", "da", "em", "no", "na", "com",
    "como", "sobre", "ai", "agora", "novo", "nova",
})

# Vocativos que podem abrir/fechar a frase.
MODEL_SWITCH_VOCATIVES = frozenset({"davi", "davios", "assistente"})

# Alvos que NÃO nomeiam modelo nenhum: geram pedido de esclarecimento em vez
# de escolha automática ("troca para um modelo grande" não escolhe nada).
# Aliases que o catálogo REALMENTE resolve (forte, leve, balanceado, pequeno,
# potente...) ficam de fora de propósito: quem decide isso é o catálogo.
MODEL_SWITCH_VAGUE_TARGETS = frozenset({
    "grande", "maior", "menor", "melhor", "melhores", "bom", "boa", "outro",
    "outra", "diferente", "rapido", "mais rapido", "ideal", "recomendado",
    "apropriado", "adequado", "qualquer", "certo", "top", "interessante",
    "moderno", "recente", "otimo", "excelente",
})

# Evidência mínima de que a frase fala de MODELO. Sem um tamanho ("8b",
# "0.6b"), uma quantização ("q4km", "q4_k_m") ou a palavra "modelo", verbos
# genéricos do dia a dia ("use o bom senso", "troca de assunto") não viram
# comando de troca.
MODEL_TARGET_HINT_RE = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?\s*b\b)"      # 8b, 0.6b, 14 b
    r"|(?:\bq\d[_a-z0-9]*\b)"           # q4km, q4_k_m, q8_0
    r"|(?:\bgguf\b)"
)
MODEL_WORD_RE = re.compile(r"\b(?:modelo|model|modelos|llm)\b")


# ---------------------------------------------------------------------------
# Inteligencia conversacional (etapa 2)
# ---------------------------------------------------------------------------

# Minusculas + sin acentos, para robustez ante "e você?" / "que horas são?".
def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    ).lower()


# ---------------------------------------------------------------------------
# Extração da ordem de troca de modelo (etapa 6)
# ---------------------------------------------------------------------------

def _strip_word_edges(text: str) -> str:
    """Remove pontuação/vocativo/preenchimento nas duas pontas da frase."""
    tokens = [
        token.strip(",.!?;:\"'")
        for token in (text or "").split()
    ]
    tokens = [token for token in tokens if token]
    edge = MODEL_SWITCH_FILLERS | MODEL_SWITCH_VOCATIVES
    while tokens and tokens[0] in edge:
        tokens.pop(0)
    while tokens and tokens[-1] in edge:
        tokens.pop()
    return " ".join(tokens)


def _starts_with_switch_verb(folded: str) -> bool:
    """True se a frase COMEÇA pelo verbo de troca (vocativo à frente é aceito)."""
    head = re.sub(r"^[^\w]+", "", folded or "")
    if MODEL_SWITCH_VERB_RE.match(head):
        return True
    for vocative in MODEL_SWITCH_VOCATIVES:
        if head.startswith(vocative):
            rest = re.sub(r"^[^\w]+", "", head[len(vocative):])
            if MODEL_SWITCH_VERB_RE.match(rest):
                return True
    return False


def extract_model_switch_request(text: str) -> Optional["ModelSwitchRequest"]:
    """Traduz uma ORDEM explícita de troca de modelo em ModelSwitchRequest.

    Devolve ``None`` para consultas ("qual modelo está usando?"), dúvidas
    ("acho que talvez eu devesse usar o X"), negações e frases sem referência
    a modelo — nesses casos a conversa segue normalmente. Nunca resolve o
    alvo no catálogo: devolve o texto do alvo para o ModelManager resolver.

    Regra de segurança: na dúvida, NÃO há intenção de troca.
    """
    raw = (text or "").strip()
    if not raw:
        return None
    folded = _fold(raw)

    # 1) Consulta explícita nunca é ordem ("qual modelo...?", "como trocar?").
    head = re.sub(r"^[^\w]+", "", folded)
    words = head.split()
    if words and words[0] in QUESTION_STARTERS:
        return None

    # 2) Sem verbo de troca não há ordem ("me fale sobre o Qwen 8B").
    match = MODEL_SWITCH_VERB_RE.search(folded)
    if match is None:
        return None

    # 3) Interrogação que não começa pelo verbo é pergunta, não ordem
    #    ("você pode trocar o modelo para o Qwen 8B?", "consigo trocar?").
    if "?" in raw and not _starts_with_switch_verb(folded):
        return None

    # 4) Dúvida / negação / recomendação: na dúvida NÃO se executa
    #    ("acho que talvez eu devesse usar o Qwen 8B").
    if MODEL_SWITCH_HEDGE_RE.search(folded):
        return None

    target = _strip_word_edges(folded[match.end():])

    # 5) Evidência mínima de que a frase fala de MODELO. Sem isso, verbos
    #    genéricos do dia a dia não viram comando ("troca de assunto").
    if not (
        MODEL_TARGET_HINT_RE.search(target) or MODEL_WORD_RE.search(folded)
    ):
        return None

    # 6) Alvo vago ("grande", "melhor") não nomeia modelo: devolve o pedido
    #    SEM alvo e a camada de resposta pede esclarecimento.
    if target in MODEL_SWITCH_VAGUE_TARGETS:
        target = ""

    return ModelSwitchRequest(action="switch_model", target=target, raw_text=raw)




# Conversa casual: reacciones breves que NO deben convertirse en ayuda.
SOCIAL_CASUAL_EXACT = frozenset({
    "legal", "boa", "boaa", "boah", "eai", "eaii", "jaja", "ah", "aha",
    "hm", "epa", "okey", "okay",
})
SOCIAL_LAUGHTER_RE = re.compile(r"^(?:k{2,}|x+d+)$")

# Pregunta reciprocada hacia el asistente ("e vc?", "como voce esta?").
SOCIAL_RECIPROCAL_PATTERNS = (
    "e voce", "e vc", "y tu", "e tu", "e ti",
    "voce esta bem", "vc esta bem", "voce ta bem", "vc ta bem",
    "como voce esta", "como vc esta", "como estas", "como va tu",
    "tudo bem e voce", "tudo bem e vc",
)

# Fragmentos de follow-up: frases cortas que refieren a lo dicho antes.
FOLLOW_UP_FRAGMENTS = frozenset({
    "por que", "porque", "pq", "p q", "como assim", "e dai", "edai",
    "e da", "qual", "quais", "como", "isso", "essa parte", "explica mais",
    "sigue", "continua", "ah", "aha", "hm",
})
FOLLOW_UP_EXPLICIT = (
    "o que voce quer saber", "oque voce quer saber",
    "que voce quer saber", "que quieres saber", "o que tu quieres saber",
)

# Pedidos de archivo.
FILE_REQUEST_CONTENT_RE = re.compile(
    r"\b(?:o que|que|lo que)\s+(?:tem|hay|tiene)\s+(?:no|en|em)\s+([^\s]+)"
)
FILE_REQUEST_VERB_RE = re.compile(
    r"\b(leia|ler|lee|abre|abrir|abra|muestra|mostra|mostre|ensename|"
    r"ensena|muestrame)\b"
)
FILE_WORDS = (
    "arquivo", "archivo", "file", ".py", ".txt", ".md", ".json", ".log",
    "contenido", "codigo", "codigo fuente",
)

# Pedido de hora actual.
TIME_REQUEST_PATTERNS = (
    "que horas", "q horas", "que hora", "as horas", "hora atual",
    "hora agora", "hora es",
)


def looks_like_follow_up_fragment(folded: str) -> bool:
    """True si el texto (ya normalizado) es un follow-up corto típico.

    NO depende del contexto: solo decide si la FRASE en sí es un fragmento
    de continuacion ("por que?", "como assim?", "e dai?"). La decision final
    con/sin historico la toma ``IntentClassifier._is_follow_up``.
    """
    text = (folded or "").strip()
    if not text:
        return False
    if any(frag in text for frag in FOLLOW_UP_EXPLICIT):
        return True
    if text in FOLLOW_UP_FRAGMENTS:
        return True
    if len(text.split()) <= 3 and text.startswith(
        ("por que", "porque", "como assim", "e dai", "edai")
    ):
        return True
    return False


class IntentClassifier:
    """Classificador de intenção por regras, extensível para LLM futuramente."""

    def classify(self, text: str, context=None) -> Intent:
        original = (text or "").strip()
        normalized = original.lower().rstrip("!.?,")
        folded = _fold(normalized)
        has_question_mark = "?" in original

        if self._is_farewell(normalized):
            return Intent.FAREWELL
        if self._is_social(folded):
            return Intent.SOCIAL
        if self._is_greeting(normalized):
            return Intent.GREETING
        if self._is_social_reciprocal(folded):
            return Intent.SOCIAL_RECIPROCAL
        if self._is_file_request(folded):
            return Intent.FILE_REQUEST
        if self._is_time_request(folded):
            return Intent.TIME_REQUEST
        if self._is_follow_up(folded, context):
            return Intent.FOLLOW_UP
        # Etapa 6: ORDEM explícita de troca de modelo. Vem antes de COMMAND
        # (que não cobre "troca"/"usa") e antes de GENERAL_QUESTION, mas só
        # quando há evidência suficiente — consultas e dúvidas seguem o fluxo
        # normal de conversa.
        if self.is_model_switch_request(original) is not None:
            return Intent.MODEL_SWITCH
        if self._matches_any(normalized, COMMAND_MARKERS):
            return Intent.COMMAND
        if self._matches_any(normalized, OPINION_MARKERS):
            return Intent.OPINION
        if self.is_memory_query(normalized, has_question_mark):
            return Intent.MEMORY_QUERY
        if self.is_memory_statement(normalized):
            return Intent.MEMORY_STATEMENT
        if self.is_general_question(normalized, original=original):
            return Intent.GENERAL_QUESTION
        return Intent.CONVERSATION

    # ------------------------------------------------------------------

    @staticmethod
    def _matches_any(text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)

    @staticmethod
    def _is_greeting(text: str) -> bool:
        words = text.split()
        if not words or len(words) > 4:
            return False
        if "tudo bem" in text or "tudo bom" in text:
            greetings = {"oi", "ola", "eae", "iai", "hello", "hi", "opa",
                         "bom", "boa", "dia", "tarde", "noite", "e", "ai",
                         "tudo", "bem"}
            return any(w in greetings for w in words)
        greetings = {
            "oi", "ola", "eae", "e", "ai", "iai", "hello", "hi", "opa",
            "bom", "boa", "dia", "tarde", "noite", "tudo", "bem",
        }
        return all(w in greetings for w in words)

    @staticmethod
    def _is_farewell(text: str) -> bool:
        farewells = {
            "sair", "tchau", "encerrar", "exit", "quit", "bye",
            "ate mais", "falow", "falou", "flw", "adeus",
        }
        return text in farewells

    # ------------------------------------------------------------------
    # Nuevas categorias conversacionales (etapa 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_social(folded: str) -> bool:
        if folded in SOCIAL_CASUAL_EXACT:
            return True
        return bool(SOCIAL_LAUGHTER_RE.match(folded))

    @staticmethod
    def _is_social_reciprocal(folded: str) -> bool:
        # Pregunta completa hacia el asistente ("como voce esta?").
        if any(
            pattern in folded
            for pattern in (
                "como voce esta", "como vc esta", "como estas",
                "como va tu", "voce esta bem", "voce ta bem",
                "vc esta bem", "vc ta bem", "tudo bem e voce",
                "tudo bem e vc",
            )
        ):
            return True
        # "e voce?" / "e vc?" como TOKENS independientes (no parte de
        # "que voce", "o que voce"...).
        tokens = folded.split()
        for i, token in enumerate(tokens[:-1]):
            if token == "e" and tokens[i + 1] in ("voce", "vc", "tu", "ti"):
                return True
            if token in ("y",) and tokens[i + 1] in ("tu", "ti"):
                return True
        return False

    @classmethod
    def _is_file_request(cls, folded: str) -> bool:
        m = FILE_REQUEST_CONTENT_RE.search(folded)
        if m:
            obj = m.group(1).rstrip("?!.,")
            if any(w in obj for w in FILE_WORDS) or re.search(r"\.\w{1,5}$", obj):
                return True
        if FILE_REQUEST_VERB_RE.search(folded):
            if any(w in folded for w in FILE_WORDS) or re.search(
                r"\.\w{1,5}\b", folded
            ):
                return True
        return False

    @staticmethod
    def _is_time_request(folded: str) -> bool:
        return any(p in folded for p in TIME_REQUEST_PATTERNS)

    def _is_follow_up(self, folded: str, context) -> bool:
        # Meta-pregunta explicita ("o que voce quer saber (sobre)?"): es
        # follow-up por naturaleza, con o sin historico.
        if any(frag in folded for frag in FOLLOW_UP_EXPLICIT):
            return True
        # Fragmento corto: solo es follow-up si hay conversacion reciente.
        if context is None:
            return False
        has_history = bool(getattr(context, "messages", None))
        if not has_history:
            return False
        return looks_like_follow_up_fragment(folded)

    def is_memory_query(self, text: str, has_question_mark: bool = False) -> bool:
        if not self._matches_any(text, MEMORY_QUERY_MARKERS):
            return False
        return has_question_mark or self._is_interrogative_start(text)

    def is_memory_statement(self, text: str) -> bool:
        return self._matches_any(text, MEMORY_STATEMENT_MARKERS)

    def is_model_status(self, text: str) -> bool:
        """Detecta perguntas sobre o modelo atual."""
        folded = _fold(text)
        if self._matches_any(text, MODEL_STATUS_MARKERS):
            return True
        if re.search(r"qual\s+(o\s+)?modelo\s+(que\s+)?(voce|você|vc|tu)\s+(esta|está|ta)\s*(usando|rodando)?", folded):
            return True
        return False

    def is_model_list(self, text: str) -> bool:
        """Detecta pedidos para listar modelos."""
        folded = _fold(text)
        if self._matches_any(text, MODEL_LIST_MARKERS):
            return True
        if re.search(r"quals?\s+(modelo|model)", folded):
            return True
        if re.search(r"lista\s+(de\s+)?modelo", folded):
            return True
        return False

    def is_model_list_installed(self, text: str) -> bool:
        """Detecta pedido específico para listar modelos instalados."""
        folded = _fold(text)
        return ("instalad" in folded and ("modelo" in folded or "model" in folded))

    def is_model_list_available(self, text: str) -> bool:
        """Detecta pedido para listar modelos disponíveis para instalar."""
        folded = _fold(text)
        return ("disponivél" in folded or "disponivel" in folded) and ("modelo" in folded or "model" in folded)

    def is_model_download(self, text: str) -> bool:
        """Detecta pedido para baixar/instalar modelo."""
        folded = _fold(text)
        if self._matches_any(text, MODEL_DOWNLOAD_MARKERS):
            return True
        # "quisiera tener", "quero ter", "preciso de"
        if re.search(r"(quero|preciso|gostaria de|quería)\s+(ter|baixar|instalar|download)", folded):
            return True
        return False

    def is_model_requirements(self, text: str) -> bool:
        """Detecta perguntas sobre requisitos de modelo."""
        folded = _fold(text)
        if self._matches_any(text, MODEL_REQUIREMENTS_MARKERS):
            return True
        # "esse modelo roda no meu pc?"
        if re.search(r"(esse|aquele)\s+(modelo|ele)\s+(roda|funciona|gira|da|anda)\s+(no|na|em|num)", folded):
            return True
        return False

    def is_model_confirm_download(self, text: str) -> bool:
        """Detecta confirmação para download/instalação."""
        folded = _fold(text)
        if self._matches_any(text, MODEL_CONFIRM_MARKERS):
            return True
        # "ok, baixa logo", "pode instalar"
        if re.search(r"(ok|claro|pode|segue|prosseguir)\b", folded):
            return True
        return False

    def is_model_switch(self, text: str) -> bool:
        """(Compatibilidade) True se a frase parece um pedido de troca.

        Mantido porque já existia na API pública do classificador. Ele responde
        apenas "parece uma troca?", de forma permissiva, e NÃO é o que a
        `classify()` usa: a decisão de intenção vem de
        `is_model_switch_request()`, que exige uma ORDEM explícita.
        """
        folded = _fold(text)
        if self._matches_any(text, MODEL_SWITCH_MARKERS):
            return True
        return extract_model_switch_request(text) is not None

    def is_model_switch_request(self, text: str) -> Optional[ModelSwitchRequest]:
        """Devolve o `ModelSwitchRequest` da frase, ou None se não houver ordem.

        Camada de INTENÇÃO apenas: o alvo continua em linguagem natural e é o
        `ModelManager` quem o resolve no catálogo (`resolve_model`) e executa
        (`switch_active_model`). Aqui não existe tabela de "8b -> id": seria
        conhecimento duplicado do catálogo.
        """
        return extract_model_switch_request(text)

    def is_model_cancel(self, text: str) -> bool:
        """Detecta pedido para cancelar operação."""
        folded = _fold(text)
        if self._matches_any(text, MODEL_CANCEL_MARKERS):
            return True
        return False

    def is_general_question(self, text: str, original: str = "") -> bool:
        if self._matches_any(text, GENERAL_QUESTION_HINTS):
            return True
        first_word = text.split()[0] if text else ""
        if first_word in QUESTION_STARTERS and not self._matches_any(
            text, MEMORY_QUERY_MARKERS
        ):
            return True
        return (
            "?" in original
            and not self._matches_any(text, MEMORY_STATEMENT_MARKERS)
        )

    @staticmethod
    def _is_interrogative_start(text: str) -> bool:
        first = text.split()[0] if text else ""
        return first in QUESTION_STARTERS or first in {"qual", "quem", "quais"}


def extract_topic_from_previous(context) -> str:
    """Extrai o tópico atual do contexto conversacional (para continuidade)."""
    topic = getattr(context, "current_topic", "") or ""
    return topic.strip()


def resolve_context_reference(text: str, context) -> Optional[str]:
    """Tenta resolver referências ('isso', 'disso'...) E continuaciones
    cortas de la conversación ('por que?', 'como assim?', 'o que voce
    quer saber sobre?').

    NUNCA reemplaza el mensaje original: lo preserva y añade contexto
    adicional para el modelo (conversación reciente, assunto actual).
    Devuelve None si no hay ninguna referencia resuelta.
    """
    lowered = text.lower()
    folded = _fold(lowered)
    messages = getattr(context, "messages", None) or []
    topic = extract_topic_from_previous(context)

    # 1) Referencias a objetos/sustantivos ("disso", "esse arquivo"...).
    if re.search(
        r"\b(isso|disso|disto|dessa|desse|aquele|aquilo|esse|essa)\b",
        folded,
    ):
        if not topic and messages:
            last_user = messages[-1].get("user", "") if messages else ""
            if last_user:
                topic = str(last_user)
        if topic:
            return "\n".join(
                [
                    f"Mensagem atual do usuario: {text}",
                    f"Assunto recente da conversa: {topic}",
                ]
            )
        return None

    # 2) Continuacion corta que depende del último mensaje.
    if looks_like_follow_up_fragment(folded) and messages:
        last_user = str(messages[-1].get("user", "") or "")
        last_resp = str(messages[-1].get("response", "") or "")
        parts = [
            f"Mensagem atual do usuario: {text}",
            "Esta mensaje e uma CONTINUACION do que se esta a falar.",
        ]
        if last_user:
            parts.append(f"Ultima mensagem do usuario: {last_user}")
        if last_resp:
            parts.append(f"Ultima resposta de DaviOS: {last_resp}")
        return "\n".join(parts)

    return None
