"""CognitiveCore do DaviOS.

Coordena o raciocínio antes da resposta:
    mensagem atual + contexto + memórias relevantes + intenção
        → PromptBuilder → LLMProvider → texto de resposta

O LLM NÃO controla memória nem sistema: ele apenas produz linguagem.
A memória continua sendo gravada/recuperada pelo MemoryManager; o
CognitiveCore apenas seleciona o que é relevante para o prompt.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any, Optional

from brain.intent_classifier import Intent, IntentClassifier
from brain.llm_provider import LLMProvider, LLMRequest, LLMUnavailableError
from brain.prompt_builder import PromptBuilder
from brain.tool_execution_flow import ToolExecutionOrchestrator
from brain.tool_registry import ToolRouter
from brain.tool_request_parser import sanitize_response_text
from config.davios_config import DaviosConfig
from personality.personality import DEFAULT_PERSONALITY

logger = logging.getLogger("davios.cognitive")

# Palavras muito comuns não servem como chave de memória
STOPWORDS = {
    "para", "com", "uma", "que", "por", "mais", "como", "mas", "isso",
    "este", "esta", "esse", "essa", "sobre", "quando", "muito", "pelo",
    "pela", "sem", "dos", "das", "nao", "sim", "voce", "meu", "minha",
    "tambem", "entao", "aqui", "onde", "todo", "toda", "ele", "ela",
}


class CognitiveCore:
    """Núcleo cognitivo: memória relevante + contexto → prompt → LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[DaviosConfig] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        memory_manager=None,
        tool_router: Optional[ToolRouter] = None,
    ):
        self.config = config or DaviosConfig.load()
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder or PromptBuilder(
            self.config, DEFAULT_PERSONALITY
        )
        self.memory_manager = memory_manager
        self.classifier = IntentClassifier()
        # C5: loop de ferramentas. Inerte quando tools_visible_to_llm=False
        # (padrão) ou sem router injetado — o fluxo é idêntico ao pré-C5.
        self.tool_flow = ToolExecutionOrchestrator(
            self.config, tool_router=tool_router
        )
        # Reaproveita o MESMO registry usado pelo ToolRouter (sem duplicar).
        # Com tool_router=None (ex: testes antigos), registry fica None e a
        # seção de ferramentas simplesmente não aparece — degrada com segurança.
        self.tools_registry = tool_router.registry if tool_router else None

    # ------------------------------------------------------------------

    def generate_response(
        self,
        user_message: str,
        *,
        context=None,
        intent: Optional[Intent] = None,
        new_facts: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Produz resposta via LLM. Lança LLMUnavailableError se indisponível.

        ``new_facts`` carrega fatos recém-extraídos/salvos nesta mesma
        mensagem (ex.: "Meu nome é Davi"), garantindo que o modelo use a
        informação acabada de fornecer.
        """
        # Aceita Intent (enum) ou string simples (ex.: decision.intent)
        if intent is None:
            resolved_intent = self.classifier.classify(user_message)
            intent_value = resolved_intent.value
        elif isinstance(intent, str):
            resolved_intent = intent
            intent_value = intent
        else:
            resolved_intent = intent
            intent_value = intent.value
        resolved_context = self._resolve_context_reference(user_message, context)
        memories = self.retrieve_relevant_memories(
            user_message, context, intent=resolved_intent
        )
        logger.info(
            "[MEMORY] retrieved=%d relevant=%d",
            len(memories), len(memories),
        )

        built = self.prompt_builder.build(
            user_message,
            context=context,
            memories=memories,
            intent=intent_value,
            resolved_context=resolved_context,
            new_facts=new_facts,
            style_profile=self._current_style_profile(),
            tools_registry=self.tools_registry,
        )
        logger.info(
            "[LLM] prompt_has_tools_section=%s prompt_len=%d",
            "FERRAMENTAS DISPONIVEIS" in built.prompt,
            len(built.prompt),
        )
        logger.info(
            "[CONTEXT] recent_messages=%d memories=%d new_facts=%d",
            len(getattr(context, "messages", []) or []),
            len(memories),
            len(new_facts or []),
        )

        request = LLMRequest(
            prompt=built.prompt,
            system=built.system,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            repeat_penalty=self.config.repeat_penalty,
            repeat_last_n=self.config.repeat_last_n,
        )
        logger.info("[LLM] generation_started provider=%s", self.llm_provider.name)
        response = self.llm_provider.generate(request)
        logger.info("[LLM] generation_completed backend=%s", response.backend)
        logger.info("[LLM] raw_response text=%r", response.text)

        # Protecao anti-repeticao: se a nova resposta for praticamente
        # identica a uma resposta recente, regenera uma vez com um
        # contexto enxuto (sem historico) para quebrar o atrator.
        recent_responses = [
            (m.get("response") or "")
            for m in (getattr(context, "messages", []) or [])
            if isinstance(m, dict)
        ]
        recent_responses = [r for r in recent_responses if r]
        if self._is_duplicate_response(response.text, recent_responses):
            logger.warning(
                "[LLM] resposta duplicada detectada; regenerando sem historico"
            )
            try:
                retry_built = self.prompt_builder.build(
                    user_message,
                    context=None,
                    memories=memories,
                    intent=intent_value,
                    resolved_context=None,
                    new_facts=new_facts,
                    style_profile=self._current_style_profile(),
                    tools_registry=self.tools_registry,
                )
                retry_request = LLMRequest(
                    prompt=retry_built.prompt,
                    system=retry_built.system,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    repeat_penalty=self.config.repeat_penalty,
                    repeat_last_n=self.config.repeat_last_n,
                )
                retry_response = self.llm_provider.generate(retry_request)
                logger.info(
                    "[LLM] retry_completed text=%r", retry_response.text
                )
                if not self._is_duplicate_response(
                    retry_response.text, recent_responses
                ):
                    response = retry_response
            except Exception:
                logger.warning(
                    "[LLM] retry apos duplicata falhou; mantendo original",
                    exc_info=True,
                )

        # Segunda camada: padron de ENCERRAMENTO repetido. Respostas
        # diferentes entre si podem terminar todas com o mesmo cierre
        # (ex: "... como posso te ajudar hoje?"). Detecta a frase final
        # repetida e regenera UNA vez MANTENDO o contexto (historico,
        # memorias, ferramentas, estilo) mas com instruccion explicita
        # de terminar de forma distinta.
        closing = self._repeated_closing(response.text, recent_responses)
        logger.info(
            "[CLOSING] extracao=%r | recentes=%r | repetido=%s",
            self._closing_sentence(response.text),
            [self._closing_sentence(r) for r in recent_responses[-3:]],
            bool(closing),
        )
        if closing:
            logger.warning(
                "[LLM] padron de encerramento repetido detectado: %r; "
                "tentando retry preservando o conteudo", closing
            )
            original_response = response
            retry_response = self._retry_avoiding_closing(
                built, closing, user_message,
            )
            accepted_retry = False
            if retry_response is not None:
                retry_closing = self._repeated_closing(
                    retry_response.text, recent_responses
                )
                retry_duplicated = self._is_duplicate_response(
                    retry_response.text, recent_responses
                )
                if not retry_closing and not retry_duplicated:
                    response = retry_response
                    accepted_retry = True
                    logger.info("[CLOSING] retry aceito (encerramento novo)")
                else:
                    # Retry ruim NUNCA substitui a resposta original e
                    # NUNCA entra no historico: recent_responses foi
                    # coletado antes, e o retry rejeitado e descartado —
                    # seu encerramento nao contamina o atrator.
                    logger.warning(
                        "[CLOSING] retry rejeitado (repetido=%s); "
                        "preservando a resposta original "
                        "semanticamente correta", retry_closing,
                    )
            if not accepted_retry:
                trimmed_text = self._trim_repeated_closing(
                    original_response.text, closing, recent_responses
                )
                if trimmed_text is not None:
                    try:
                        import dataclasses

                        response = dataclasses.replace(
                            original_response, text=trimmed_text
                        )
                    except Exception:
                        logger.warning(
                            "[CLOSING] aparo falhou ao reconstruir resposta; "
                            "original preservada", exc_info=True,
                        )
                    else:
                        logger.info(
                            "[CLOSING] aparo seguro aplicado: encerramento "
                            "repetido removido; conteudo preservado"
                        )
                else:
                    logger.warning(
                        "[CLOSING] aparo nao seguro; resposta original "
                        "preservada (com encerramento repetido)"
                    )

        # C5: fechamento do loop de ferramentas — apenas quando
        # tools_visible_to_llm=True E router injetado. Com a flag off o
        # fluxo é idêntico ao pré-C5 (uma única chamada ao LLM).
        if self.tool_flow.is_enabled():
            outcome = self.tool_flow.handle_first_response(
                response.text,
                original_prompt=built.prompt,
                original_system=built.system,
                run_llm=self._run_followup_llm,
            )
            # C6: sanitização final — remove blocos residuais que o Qwen
            # possa ter ecovado. Acontece DEPOIS do tool_flow para não
            # interferir no parser.
            final_text = sanitize_response_text(outcome.final_text)
            return {
                "text": final_text,
                "intent": intent_value,
                "memories_used": memories,
                "llm": response.to_dict(),
                "tool_flow": outcome.to_dict(),
            }

        # C6: mesmo no caminho sem ferramenta, sanitiza por segurança.
        final_text = sanitize_response_text(response.text)
        return {
            "text": final_text,
            "intent": intent_value,
            "memories_used": memories,
            "llm": response.to_dict(),
        }

    def _run_followup_llm(self, prompt: str, system: str) -> str:
        """Segunda chamada ao LLM dentro do ciclo de ferramentas (C5).

        Pode lançar LLMUnavailableError — o orquestrador degrada
        graciosamente nesse caso (não derruba a conversa).
        """
        request = LLMRequest(
            prompt=prompt,
            system=system,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            repeat_penalty=self.config.repeat_penalty,
            repeat_last_n=self.config.repeat_last_n,
        )
        response = self.llm_provider.generate(request)
        return response.text

    @staticmethod
    def _is_duplicate_response(text: str, recent: list[str]) -> bool:
        """Detecta resposta inteira (ou quase) igual a uma resposta recente.

        Camada de seguranca apenas: conversas naturais podem repetir
        palavras/frases, entao so dispara quando a resposta normalizada
        e identica a uma anterior ou difere em pouquissimos caracteres.
        Retorna False para textos curtos demais (evita falso-positivo).
        """
        import difflib

        normalized = re.sub(r"\s+", " ", (text or "").strip().casefold())
        # Igualdade EXATA vale mesmo para respostas curtas (>=10 chars):
        # o caso real era "Boa! Como posso te ajudar hoje? 😊" (33 chars)
        # repetido 4x. O guarda de 40 chars fica apenas para o ratio
        # aproximado, onde textos curto demais geram falso-positivo.
        if len(normalized) < 10:
            return False
        for prev in recent:
            prev_norm = re.sub(r"\s+", " ", (prev or "").strip().casefold())
            if not prev_norm:
                continue
            if normalized == prev_norm:
                return True
            if len(normalized) < 40:
                continue
            try:
                ratio = difflib.SequenceMatcher(
                    None, normalized, prev_norm
                ).ratio()
            except Exception:
                continue
            if ratio >= 0.92:
                return True
        return False

    # Emojis/simbolos: removidos antes da extracao do encerramento, pois o
    # Qwen costuma termina-las com emoji DEPOIS da pontuacao final — sem
    # isso, a ultima "oracao" extraida seria apenas o emoji (ex: "😊"),
    # com 1 char, e o detector jamais veria o encerramento real.
    _EMOJI_RE = re.compile(
        "["
        "\U0001F000-\U0001FAFF"  # pictógrafos e emojis supplementares
        "\U00002600-\U000027BF"  # misc symbols e dingbats (inclui ✨)
        "\uFE0F\u200D\u2B50"     # variation selector, ZWJ, estrela
        "]"
    )

    @staticmethod
    def _strip_accents(text: str) -> str:
        """Remove acentos (NFKD) para comparacao robusta entre variantes."""
        import unicodedata

        return "".join(
            ch for ch in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(ch)
        )

    @staticmethod
    def _closing_sentence(text: str, max_chars: int = 45) -> str:
        """Extrae a frase final de uma resposta (ultima oracion).

        So retorna a oracion final separada por pontuacion; respostas de
        una unica oracion devolven toda a resposta. Capada a max_chars.
        Emojis e acentos sao removidos antes (ver _EMOJI_RE): a comparacao
        do encerramento deve ser sobre o TEXTO, nao sobre o emoji final.
        """
        normalized = re.sub(r"\s+", " ", (text or "").strip()).strip()
        if not normalized:
            return ""
        normalized = re.sub(
            r"\s+", " ", CognitiveCore._EMOJI_RE.sub(" ", normalized)
        ).strip()
        normalized = CognitiveCore._strip_accents(normalized)
        if not normalized:
            return ""
        sentences = re.split(r"(?<=[.!?…])\s+", normalized)
        closing = sentences[-1].strip()
        if len(closing) > max_chars:
            closing = closing[-max_chars:]
            # Corta ao inicio da prox. palabra para nao expor fragmentos.
            idx = closing.find(" ")
            if 0 < idx <= 12:
                closing = closing[idx + 1:]
        return closing.casefold().strip()

    # Padrão estrutural de "oferta de ajuda" como encerramento — não é uma
    # frase fixa: cobre variações lexical, idioma misto e reformulacoes
    # ("como posso te ajudar?", "o que posso fazer por você?", "em que
    # posso ajudar?", "como posso te ajudar today?", "how can i help?").
    # Uma pergunta legítima (ex: "você prefere manhã ou tarde?") NÃO casa.
    _HELP_OFFER_RE = re.compile(
        r"\b(?:posso|pode[s]?|puedo|puedes?|podemos|can)\b"
        r"[^?.!]*\b(?:ajud|ayud|help|auxili|fazer|do|contribuir)\b"
        r"|\b(?:como|o que|que|em que|what)\b[^?.!]*\b(?:posso|pode[s]?"
        r"|puedo|puedes?|can i)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _is_help_offer_closing(cls, closing: str) -> bool:
        """O encerramento é uma pergunta genérica de oferta de ajuda?"""
        return bool(closing.endswith("?")) and bool(
            cls._HELP_OFFER_RE.search(closing)
        )

    @staticmethod
    def _word_overlap_ratio(a: str, b: str) -> float:
        """Jaccard sobre palavras (>2 chars): pega variação lexical que o
        ratio de caracteres perde (ex: 'te ajudar' vs 'te auxiliar')."""
        wa = {w for w in re.findall(r"\w+", a) if len(w) > 2}
        wb = {w for w in re.findall(r"\w+", b) if len(w) > 2}
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    @classmethod
    def _repeated_closing(
        cls, text: str, recent: list[str], min_chars: int = 12
    ) -> Optional[str]:
        """Devuelve o encerramento repetido se a frase final de 'text'
        coincide com a frase final doutra resposta recente.

        Tres criterios (qualquer um dispara):
            1. igual ou quase igual por caracteres (ratio >= 0.88);
            2. sobreposição de palavras (jaccard >= 0.5) — variação lexical
               ("ajudar" vs "auxiliar", "hoje" a mais/menos);
            3. ambos são ofertas genéricas de ajuda (padrão estrutural) —
               pega variações semânticas ("o que posso fazer por você?",
               "em que posso ajudar?", "how can i help?").

        Camada de seguranca generica — nao procura frases proibidas fixas.
        Textos curtos (< min_chars) ignorados: "beleza", "ok", "certo",
        "entendi" etc. poden aparecer naturalmente sen repetir padroes.
        Perguntas legitimas e específicas (ex: "você prefere manhã ou
        tarde?") NÃO casam com nenhum criterio e nunca sao bloqueadas.
        """
        closing = cls._closing_sentence(text)
        if len(closing) < min_chars:
            return None
        closing_is_offer = cls._is_help_offer_closing(closing)
        for prev in recent:
            prev_closing = cls._closing_sentence(prev)
            if len(prev_closing) < min_chars:
                continue
            if closing == prev_closing:
                return closing
            try:
                ratio = difflib.SequenceMatcher(
                    None, closing, prev_closing
                ).ratio()
            except Exception:
                continue
            if ratio >= 0.88:
                return closing
            if cls._word_overlap_ratio(closing, prev_closing) >= 0.5:
                return closing
            if closing_is_offer and cls._is_help_offer_closing(prev_closing):
                return closing
        return None

    def _trim_repeated_closing(
        self, text: str, closing: str, recent: list[str], min_len: int = 12
    ) -> Optional[str]:
        """Remove APENAS a ultima oracao (o encerramento repetido) quando
        isso e comprovadamente seguro: a resposta tem >= 2 oracoes, o resto
        continua substancial, nao cria nova duplicata nem novo encerramento
        repetido. Retorna None quando nao e seguro — o chamador preserva
        a resposta original intacta."""
        normalized = re.sub(r"\s+", " ", (text or "").strip()).strip()
        if not normalized:
            return None
        parts = re.split(r"(?<=[.!?…])\s+", normalized)
        if len(parts) < 2:
            return None
        trimmed = " ".join(parts[:-1]).strip()
        if len(trimmed) < min_len:
            return None
        if self._repeated_closing(trimmed, recent) is not None:
            return None
        if self._is_duplicate_response(trimmed, recent):
            return None
        return trimmed

    def _retry_avoiding_closing(
        self,
        built,
        closing: str,
        user_message: str,
    ) -> Optional[Any]:
        """Regenera UNA vez MANTENDO todo o contexto original (historico,
        memorias, ferramentas, personalidade, estilo), instruindo o modelo
        a PRESERVAR o conteudo/intencao ja formulado, responder a mensagem
        atual e apenas variar o encerramento (pode terminar sem pergunta).

        Retorna None se o retry falhar — o chamador nunca aceita um retry
        ruim e nunca entra em loop (limite fixo de 1 retry).
        """
        retry_prompt = (
            built.prompt
            + "\n\nNota: a resposta que voce acabou de gerar terminou com o "
            "encerramento \"%s\", ja usado em respostas recentes. Gere "
            "NOVAMENTE a resposta para a mensagem atual do usuario (\"%s\"), "
            "MANTENDO o mesmo conteudo e intencao que voce ja formulou, "
            "porem terminando de forma diferente: pode simplesmente terminar "
            "sem nenhuma pergunta. Nao repita esse encerramento e nao troque "
            "o conteudo por frases genericas tipo \"vamos falar sobre algo\"."
            % (closing, user_message)
        )
        request = LLMRequest(
            prompt=retry_prompt,
            system=built.system,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            repeat_penalty=self.config.repeat_penalty,
            repeat_last_n=self.config.repeat_last_n,
        )
        try:
            new_response = self.llm_provider.generate(request)
            logger.info(
                "[LLM] retry_encerramento_completed text=%r",
                new_response.text,
            )
            return new_response
        except Exception:
            logger.warning(
                "[LLM] retry por encerramento falhou", exc_info=True,
            )
            return None

    # ------------------------------------------------------------------
    # Recuperação de memórias relevantes (simples, extensível a RAG)
    # ------------------------------------------------------------------

    def _current_style_profile(self):
        """Recupera o StyleProfile do MemoryManager sem acoplar modulos.

        Retorna None quando o manager nao existe ou o perfil nao esta
        disponivel, para manter a geracao funcionando por regras.
        """
        manager = getattr(self, "memory_manager", None)
        if manager is None:
            return None
        try:
            return manager.style_profile
        except Exception:
            return None

    def retrieve_relevant_memories(self, text: str, context=None, memory_manager=None, intent=None) -> list[Any]:
        """Seleciona memórias relevantes para a mensagem atual.

        Estrategia (sem embeddings):
        1. Palavras-chave da mensagem + tópico atual da conversa;
        2. Fatos de identidade (ex.: nome do usuario);
        3. Fatos de projetos/estudos (working_on, studies);
        4. Preferências ativas *apenas* quando a query eh sobre a memoria
           do proprio usuario (MEMORY_QUERY) — evita injetar preferencias em
           conversas nao relacionadas.
        Nao envia o banco inteiro ao modelo.
        """
        from brain.intent_classifier import Intent

        manager = memory_manager
        if manager is None:
            manager = getattr(self, "memory_manager", None)
        if manager is None:
            return []

        keywords = self._extract_keywords(text)
        if context is not None:
            topic = getattr(context, "current_topic", "")
            if topic:
                keywords.append(topic.lower())

        relevant: list[Any] = []
        seen_targets: set[str] = set()
        is_memory_query = intent == Intent.MEMORY_QUERY

        def _cap() -> bool:
            return len(relevant) >= self.config.max_memories_in_prompt

        def _add(fact: Any) -> None:
            if not getattr(fact, "is_active", True):
                return
            key = (fact.target or "").casefold()
            if key in seen_targets:
                return
            seen_targets.add(key)
            relevant.append(fact)

        # 1. Correspondencia direta por palavra-chave (maior prioridade)
        for keyword in keywords:
            if _cap():
                break
            try:
                facts = manager.database.find_facts_by_target(
                    keyword, include_inactive=False
                )
            except Exception:
                facts = []
            for fact in facts:
                if _cap():
                    break
                _add(fact)

        # 2-3. Identidade e declarativos: sempre incluidos (pequeno conjunto,
        #    sempre relevantes para contextualizar o usuario).
        if not _cap():
            for family in ("identity", "working_on", "studies"):
                try:
                    facts = manager.database.find_active_preferences(relation_family=family)
                except Exception:
                    facts = []
                for fact in facts:
                    if _cap():
                        break
                    _add(fact)

        # 4. Preferencias sob consulta explicita sobre o usuario
        if is_memory_query and not _cap():
            try:
                preferences = manager.database.find_active_preferences()
            except Exception:
                preferences = []
            for fact in preferences:
                if _cap():
                    break
                _add(fact)
        return relevant

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        words = re.findall(r"[^\W_]+", (text or "").lower(), flags=re.UNICODE)
        keywords = [
            w for w in words if len(w) > 3 and w not in STOPWORDS
        ]
        # Preserva ordem, remove duplicatas
        seen: set[str] = set()
        ordered = []
        for word in keywords:
            if word not in seen:
                seen.add(word)
                ordered.append(word)
        return ordered[:6]

    @staticmethod
    def _resolve_context_reference(text: str, context) -> Optional[str]:
        from brain.intent_classifier import resolve_context_reference

        return resolve_context_reference(text, context)
