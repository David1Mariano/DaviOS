"""Conversation Engine do DaviOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from core.reasoning import ReasoningEngine, ReasoningResult
from memory.memory import Memory
from memory.memory_interpreter import MemoryInterpreter
from memory.memory_manager import MemoryManager

from brain.intent_classifier import Intent, IntentClassifier, extract_model_info_query
from brain.response_generator import ResponseGenerator


@dataclass
class ConversationContext:
    """Contexto conversacional."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    last_user_message: str = ""
    last_response: str = ""
    current_topic: str = ""
    last_intent: str = ""
    relevant_info: dict[str, Any] = field(default_factory=dict)

    def add_message(self, user_msg: str, response: str, intent: str) -> None:
        self.messages.append(
            {"user": user_msg, "response": response, "intent": intent}
        )
        self.last_user_message = user_msg
        self.last_response = response
        self.last_intent = intent
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_user_message": self.last_user_message,
            "last_response": self.last_response,
            "current_topic": self.current_topic,
            "last_intent": self.last_intent,
            "message_count": len(self.messages),
        }


@dataclass
class ConversationResult:
    """Resultado padronizado de uma interacao."""

    response: str
    intent: str = "conversation"
    memory_action: str = "none"
    memories_used: list[Any] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    should_exit: bool = False
    # Etapa 7: resultados ESTRUTURADOS de consultas READ-ONLY sobre modelos.
    # None nas demais interacoes — nenhuma mensagem comum carrega estado de
    # modelo. Campos distintos para estados semanticamente diferentes:
    #   model_status  -> consulta do MODELO ATUAL ("qual modelo voce usa?")
    #   model_list   -> listagem de modelos ("quais modelos instalados?")
    #   model_info   -> consulta sobre um modelo ESPECIFICO ("o qwen 8b existe?")
    model_status: Optional["ModelStatusResult"] = None
    model_list: Optional["ModelListResult"] = None
    model_info: Optional["ModelInfoResult"] = None
    # Etapa 6/7: resultado estruturado de troca de modelo (apenas quando uma
    # intenção MODEL_SWITCH foi detectada e executada). None em toda outra
    # interação — nenhuma mensagem comum carrega estado de modelo.
    model_switch: Optional[dict[str, Any]] = None



# ---------------------------------------------------------------------------
# Resultados estruturados de consultas READ-ONLY sobre modelos (etapa 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelStatusResult:
    """Resposta de consulta sobre o modelo ativo/carregado/persistido.

    Campos:
      active_model       -> modelo que o gerenciador\ModelManager considera ativo
                           (resolvido + selecionado), ou None quando nao ha modelo
                           ativo/logico.
      active_model_id    -> ID do modelo ativo (string), ou None.
      active_model_state -> estado logico do modelo ativo: active/switching/
                           unavailable (veja ModelManager._active_state).
      persisted_model_id -> modelo registrado na configuracao persistente, ou None.
      loaded_model       -> identidade REAL confirmada pelo provider no runtime,
                           ou None quando nao ha provider/confianca.
      provider_running   -> True/False/None (None = sem provider ou indisponivel
                           confirmado; cuidado: pode ser None quando o provider nao e
                           capaz de responder; nao e o mesmo que False).
      provider_backend   -> nome cosmetico do backend, ou None.
      matches_loaded    -> True/False/None: True so quando o MODELO ATIVO coincide
                           com o CARREGADO (comparacao por identidade real do
                           runtime, nunca por health_check). None quando nao e
                           possivel afirmar nem negar.
      error             -> None ou erro legivel quando o status nao e confiavel.
    """

    active_model: Optional[dict[str, Any]] = None
    active_model_id: Optional[str] = None
    active_model_state: Optional[str] = None
    persisted_model_id: Optional[str] = None
    loaded_model: Optional[str] = None
    provider_running: Optional[bool] = None
    provider_backend: Optional[str] = None
    matches_loaded: Optional[bool] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ModelListResult:
    """Resposta de listagem de modelos (installed | available | catalog).

    kind indica o que foi listado:
      installed  -> modelos realmente instalados no disco
      available  -> modelos catalogados mas NAO instalados (para instalacao)
      catalog    -> catalogo completo, distinguindo instalados/nao instalados

    Quando o gerenciador nao estiver disponivel, models e kind
    viram None e o motor gera uma resposta generica de indisponibilidade.
    """

    kind: Optional[str] = None  # installed | available | catalog
    models: Optional[list[dict[str, Any]]] = None
    installed_count: int = 0
    catalog_total: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class ModelInfoResult:
    """Resposta de consulta sobre um modelo ESPECIFICO.

    Carrega o que foi pedido e o que foi descoberto:
      question     -> o que se perguntou: installed | available | can_use |
                      loaded | compatible | state
      target       -> o alvo em linguagem natural ("qwen 8b"), como extraido
      resolved_id  -> ID catalogado correspondente, ou None
      resolved     -> ModelInfo completo (nome, caminho, tier, etc.), ou None
      installed    -> True/False/None: True quando esta instalado, False quando
                      catalogado mas nao instalado. None quando nao houve
                      correspondencia no catálogo.
      is_active    -> True/False/None: True quando este modelo e o ativo
      is_loaded    -> True/False/None: True quando o provider confirmou que
                      este modelo esta carregado
      error        -> None ou erro legivel
    """

    question: Optional[str] = None
    target: Optional[str] = None
    resolved_id: Optional[str] = None
    resolved: Optional[dict[str, Any]] = None
    installed: Optional[bool] = None
    is_active: Optional[bool] = None
    is_loaded: Optional[bool] = None
    error: Optional[str] = None


class ConversationEngine:
    """Orquestrador principal da conversacao."""

    EXIT_COMMANDS = {
        "sair", "tchau", "encerrar", "exit", "quit", "bye",
        "ate mais", "falow", "falou", "flw",
    }

    GREETING_COMMANDS = {
        "oi", "ola", "eae", "e ai", "iai", "hello", "hi",
        "opa", "bom dia", "boa tarde", "boa noite",
    }

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        response_generator: Optional[ResponseGenerator] = None,
        cognitive_core=None,
        model_manager=None,
    ):
        self.memory_manager = memory_manager or MemoryManager()
        self.reasoning = ReasoningEngine()
        self.response_generator = response_generator or ResponseGenerator()
        self.cognitive_core = cognitive_core
        # Etapa 6: o engine NAO troca modelo por conta propria. Ele entrega o
        # pedido estruturado ao ModelManager, que resolve no catalogo, valida,
        # persiste e coordena o provider — nada de llama.cpp aqui.
        self.model_manager = model_manager
        self.classifier = IntentClassifier()
        self.context = ConversationContext()
        # Garante que o nucleo cognitivo compartilhe o mesmo sistema de
        # memoria: sem isso o CognitiveCore nunca recupera fatos.
        if (
            self.cognitive_core is not None
            and getattr(self.cognitive_core, "memory_manager", None) is None
        ):
            self.cognitive_core.memory_manager = self.memory_manager

    def process(self, user_input: str) -> ConversationResult:
        """Processa uma mensagem do usuario."""
        text = (user_input or "").strip()
        if not text:
            return ConversationResult(
                response="Nao entendi. Pode repetir?",
                intent="empty_input",
            )

        normalized = text.lower().rstrip("!.?")

        # Atualiza o perfil de estilo com cada mensagem do usuario,
        # antes de qualquer roteamento. Falhas sao silenciosas.
        self._update_style_profile(text)

        if self._is_exit_command(normalized):
            return ConversationResult(
                response="Ate mais!",
                intent="exit",
                should_exit=True,
            )

        if self._is_greeting(normalized):
            response = self.response_generator.generate_greeting()
            self.context.add_message(text, response, "greeting")
            return ConversationResult(
                response=response,
                intent="greeting",
                context=self.context.to_dict(),
            )

        intent = self.classifier.classify(text, context=self.context)

        # Etapa 6: unica intencao que altera estado do sistema. O engine so
        # roteia: extracao/validacao/execucao ficam na camada de modelos.
        if intent == Intent.MODEL_SWITCH:
            return self._handle_model_switch(text)

        # Etapa 7: consultas READ-ONLY sobre modelos. Nenhuma delas altera
        # estado: apenas leem o ModelManager/provider e descrevem o que veem.
        if intent in (
            Intent.MODEL_STATUS,
            Intent.MODEL_LIST_INSTALLED,
            Intent.MODEL_LIST_AVAILABLE,
            Intent.MODEL_LIST,
            Intent.MODEL_INFO,
        ):
            return self._handle_model_query(intent, text)

        if intent == Intent.COMMAND:
            response = (
                "Essa funcao ainda nao esta disponible. Por enquanto eu "
                "converso, lembro de informacoes e respondo perguntas."
            )
            self.context.add_message(text, response, "command")
            return ConversationResult(
                response=response,
                intent="command",
                context=self.context.to_dict(),
            )

        if intent == Intent.MEMORY_QUERY:
            return self._handle_question(text)

        if intent in (
            Intent.GENERAL_QUESTION,
            Intent.OPINION,
            Intent.SOCIAL,
            Intent.SOCIAL_RECIPROCAL,
            Intent.FOLLOW_UP,
            Intent.FILE_REQUEST,
            Intent.TIME_REQUEST,
        ):
            return self._handle_llm_question(text, intent)

        if intent == Intent.MEMORY_STATEMENT:
            return self._handle_statement(text)

        # CONVERSATION/UNKNOWN: statement com possiveis fatos ou papo aberto
        return self._handle_statement(text)

    def reset_context(self) -> None:
        self.context = ConversationContext()

    def _is_exit_command(self, normalized: str) -> bool:
        return normalized in self.EXIT_COMMANDS

    def _is_greeting(self, normalized: str) -> bool:
        return normalized in self.GREETING_COMMANDS

    def _is_question(self, text_lower: str) -> bool:
        if text_lower.endswith("?"):
            return True
        starters = {
            "qual", "quem", "o que", "oque", "como", "onde",
            "quando", "por que", "porque", "quanto", "quanta",
            "quantos", "quantas", "quais",
        }
        first_word = text_lower.split()[0] if text_lower else ""
        return first_word in starters

    # ------------------------------------------------------------------ #
    # Troca de modelo por linguagem natural (etapa 6)
    # ------------------------------------------------------------------ #
    # O engine NAO conhece IDs, quantizacoes nem llama.cpp. Ele recebe a
    # intencao estruturada (Intent.MODEL_SWITCH + ModelSwitchRequest), entrega
    # o alvo em linguagem natural ao ModelManager e traduz o resultado:
    #
    #   mensagem -> intent -> ModelManager.resolve_model() -> catalogo
    #            -> ModelManager.switch_active_model() -> provider -> resposta
    #
    # Fonte unica da verdade: quem decide e executa e o ModelManager.

    def _handle_model_switch(self, text: str) -> ConversationResult:
        """Atende uma ordem explicita de troca de modelo.

        Estados possiveis (campo `status` do resultado estruturado):
        switched | need_target | not_found | not_installed | switch_failed |
        divergent | unavailable.
        """
        request = self.classifier.is_model_switch_request(text)
        target = (request.target if request is not None else "").strip()
        outcome: dict[str, Any] = {
            "success": False,
            "requested_model": target or None,
            "resolved_model": None,
            "previous_model": None,
            "error": None,
            "status": "unavailable",
        }

        manager = self.model_manager
        if manager is None:
            outcome["error"] = "gerenciador de modelos indisponivel nesta sessao"
            return self._finish_model_switch(
                "O gerenciamento de modelos nao esta ativo nesta sessao.",
                outcome,
                text,
            )

        if not target:
            # Pedido sem alvo ("quero um modelo grande"): nada e escolhido em
            # silencio. Pedimos que a pessoa nomeie o modelo.
            outcome["status"] = "need_target"
            return self._finish_model_switch(
                "Qual modelo voce quer que eu use? Diga o nome ou o tamanho "
                '(por exemplo, "Qwen 8B") ou um tier como "leve" ou "forte".',
                outcome,
                text,
            )

        # Resolucao delegada: sem casamento no catalogo, nada e inventado.
        resolved = manager.resolve_model(target)
        if resolved is None:
            outcome["status"] = "not_found"
            outcome["error"] = f"modelo nao encontrado no catalogo: {target!r}"
            return self._finish_model_switch(
                "Nao encontrei esse modelo no catalogo. Nada foi alterado.",
                outcome,
                text,
            )
        outcome["resolved_model"] = resolved.id

        if not resolved.is_installed():
            # Catalogado mas ausente no disco: NAO baixamos nesta etapa e NAO
            # trocamos para outro modelo por conta propria.
            outcome["status"] = "not_installed"
            outcome["error"] = (
                f"{resolved.id} esta no catalogo mas o arquivo nao existe; "
                "download nao faz parte desta etapa"
            )
            return self._finish_model_switch(
                f"Nao consegui trocar para {resolved.name} porque o modelo "
                "nao esta instalado. Nada foi baixado e o modelo atual "
                "continua o mesmo.",
                outcome,
                text,
            )

        previous = manager.get_active_model()
        outcome["previous_model"] = previous.id if previous is not None else None

        # Execucao real: validacao + provider + persistencia + rollback.
        switched = manager.switch_active_model(resolved)
        if switched is None:
            # Falha do gerenciador chega aqui intacta, com o motivo real.
            outcome["error"] = (
                getattr(manager, "last_error", "")
                or "troca nao realizada pelo gerenciador de modelos"
            )
            if self._model_runtime_diverged(manager):
                outcome["status"] = "divergent"
                response = (
                    "A troca encontrou um erro durante a recuperacao do "
                    "estado. O runtime e o registro podem estar divergentes."
                )
            else:
                outcome["status"] = "switch_failed"
                response = (
                    "A troca para o modelo solicitado falhou. O modelo "
                    "anterior foi preservado."
                )
            return self._finish_model_switch(response, outcome, text)

        outcome["success"] = True
        outcome["status"] = "switched"
        outcome["resolved_model"] = switched.id
        return self._finish_model_switch(
            f"Modelo alterado para {switched.name}.", outcome, text
        )

    @staticmethod
    def _model_runtime_diverged(manager) -> bool:
        """True so quando o status AFIRMA que registro e runtime divergem.

        Um `None` (provider sem identidade confiavel) nao e divergencia: nao
        afirmamos problema sem evidencia.
        """
        try:
            status = manager.get_status()
        except Exception:
            return False
        return status.get("active_model_matches_loaded") is False

    def _finish_model_switch(
        self, response: str, outcome: dict[str, Any], text: str
    ) -> ConversationResult:
        """Fecha a interacao registrando resposta, contexto e resultado."""
        self.context.add_message(text, response, "model_switch")
        return ConversationResult(
            response=response,
            intent="model_switch",
            memory_action="model_switch",
            context=self.context.to_dict(),
            model_switch=outcome,
        )


    # ------------------------------------------------------------------ #
    # Consultas READ-ONLY de modelo (etapa 7)
    # ------------------------------------------------------------------ #

    def _handle_model_query(self, intent: Intent, text: str) -> ConversationResult:
        """Roteia consultas READ-ONLY de modelo para o handler adequado."""
        if self.model_manager is None:
            self.context.add_message(text, "gerenciador indisponivel", "model_query")
            return ConversationResult(
                response="O gerenciamento de modelos nao esta ativo nesta sessao. Sem ele nao consigo verificar o estado dos modelos.",
                intent="model_query",
                memory_action="none",
                context=self.context.to_dict(),
            )

        handlers = {
            Intent.MODEL_STATUS: self._handle_model_status,
            Intent.MODEL_LIST_INSTALLED: self._handle_model_list_installed,
            Intent.MODEL_LIST_AVAILABLE: self._handle_model_list_available,
            Intent.MODEL_LIST: self._handle_model_list,
            Intent.MODEL_INFO: self._handle_model_info_query,
        }
        handler = handlers.get(intent)
        if handler is None:
            self.context.add_message(text, "gerenciador indisponivel", "model_query")
            return ConversationResult(
                response="O gerenciamento de modelos nao esta ativo nesta sessao. Sem ele nao posso verificar o estado dos modelos.",
                intent="model_query",
                memory_action="none",
                context=self.context.to_dict(),
            )
        return handler(text)

    def _handle_model_status(self, text: str) -> ConversationResult:
        """Responde a: "qual modelo voce esta usando?", "que modelo esta rodando?". """
        manager = self.model_manager
        try:
            status = manager.get_status()
        except Exception as exc:
            import logging

            logging.getLogger("davios.conversation").warning(
                "[MODEL] falha ao obter status: %s", exc, exc_info=True
            )
            self.context.add_message(text, "erro ao consultar status", "model_status")
            return ConversationResult(
                response="Nao consegui verificar o estado dos modelos agora. Tente novamente em instantes.",
                intent="model_status",
                memory_action="model_status_error",
                context=self.context.to_dict(),
                model_status=ModelStatusResult(error=str(exc)),
            )

        active_id = status.get("active_model_id")
        active_name = status.get("active_model")
        active = None
        if active_id:
            try:
                active = manager.resolve_model(active_id)
            except Exception:
                active = None
        active_state = getattr(manager, 'active_state', None)
        persisted_id = status.get("persisted_active_model_id")
        loaded = status.get("loaded_model")
        provider_running = status.get("provider_running")
        provider_backend = status.get("provider_backend")
        matches_loaded = status.get("active_model_matches_loaded")

        active_str = active_name or active_id or "nenhum"
        loaded_str = loaded if loaded else "nenhum confirmado"
        persisted_str = persisted_id if persisted_id else "nenhum registrado"

        if provider_running is False:
            provider_str = "indisponivel"
        elif provider_running is True:
            provider_str = provider_backend or "disponivel"
        else:
            provider_str = "desconhecido (sem provider)"

        if matches_loaded is True:
            sync_str = "sincronizado"
        elif matches_loaded is False:
            sync_str = "divergente"
        else:
            sync_str = "nao e possivel confirmar"

        parts = [f"Modelo ativo: {active_str}.", f"Modelo carregado: {loaded_str}.", f"Provider: {provider_str}."]
        if persisted_id and active_id and persisted_id != active_id:
            parts.append(f"Registrado: {persisted_str} (diferente do ativo).")
        parts.append(f"Estado: {sync_str}.")

        response = " ".join(parts)
        self.context.add_message(text, response, "model_status")
        return ConversationResult(
            response=response, intent="model_status", memory_action="model_status",
            context=self.context.to_dict(),
            model_status=ModelStatusResult(
                active_model=self._model_to_dict(active), active_model_id=active_id,
                active_model_state=active_state, persisted_model_id=persisted_id,
                loaded_model=loaded, provider_running=provider_running,
                provider_backend=provider_backend, matches_loaded=matches_loaded, error=None,
            ),
        )

    def _handle_model_list_installed(self, text: str) -> ConversationResult:
        """Responde a: "quais modelos estao instalados?". """
        manager = self.model_manager
        try:
            installed = manager.list_models(only_installed=True)
        except Exception as exc:
            import logging

            logging.getLogger("davios.conversation").warning(
                "[MODEL] falha ao listar modelos instalados: %s", exc, exc_info=True
            )
            self.context.add_message(text, "erro ao listar modelos instalados", "model_list_installed")
            return ConversationResult(
                response="Nao consegui listar os modelos instalados agora. Tente novamente em instantes.",
                intent="model_list_installed",
                memory_action="model_list_error",
                context=self.context.to_dict(),
                model_list=ModelListResult(kind="installed", models=None, error=str(exc)),
            )

        models = [self._model_to_dict(m) for m in installed]
        count = len(models)
        if count == 0:
            response = "Nenhum modelo instalado encontrado no momento."
        else:
            names = [m["name"] for m in models if m and m.get("name")]
            response = (f"Modelos instalados ({count}): " + ", ".join(names) + ".") if names else f"{count} modelo(s) instalado(s) encontrado(s)."

        self.context.add_message(text, response, "model_list_installed")
        return ConversationResult(
            response=response, intent="model_list_installed", memory_action="model_list",
            context=self.context.to_dict(),
            model_list=ModelListResult(
                kind="installed", models=models, installed_count=count,
                catalog_total=(len(manager.list_models(only_installed=False)) if manager else 0),
            ),
        )

    def _model_to_dict(self, model) -> Optional[dict[str, Any]]:
        if model is None:
            return None
        try:
            return model.to_dict()
        except AttributeError:
            return {'id': getattr(model, 'id', None), 'name': getattr(model, 'name', None), 'tier': getattr(model, 'tier', None)}

    def _handle_model_list_available(self, text: str) -> ConversationResult:
        """Responde a: "quais modelos estao disponiveis?", "que modelos posso usar?". """
        manager = self.model_manager
        try:
            available = manager.list_models(only_installed=False)
        except Exception as exc:
            import logging
            logging.getLogger('davios.conversation').warning('[MODEL] falha ao listar: %s', exc, exc_info=True)
            self.context.add_message(text, 'erro ao listar', 'model_list_available')
            return ConversationResult(response='Nao consegui listar modelos disponiveis agora.', intent='model_list_available', memory_action='model_list_error', context=self.context.to_dict(), model_list=ModelListResult(kind='available', models=None, error=str(exc)))
        models = [self._model_to_dict(m) for m in available]
        total = len(models)
        if total == 0:
            response = 'Nenhum modelo encontrado no catalogo.'
        else:
            installed = [m for m in models if m and m.get('installed')]
            not_installed = [m for m in models if m and not m.get('installed')]
            parts = []
            if installed:
                names = [m['name'] for m in installed if m.get('name')]
                parts.append(f'Instalados: {", ".join(names)}.' if names else 'Instalados: nenhum.')
            if not_installed:
                names = [m['name'] for m in not_installed if m.get('name')]
                if names:
                    parts.append(f'Disponiveis para uso: {", ".join(names)}.')
            if not parts:
                names = [m['name'] for m in models if m.get('name')]
                response = f'{total} modelo(s) no catalogo: ' + ', '.join(names) + '.'
            else:
                response = ' '.join(parts)
        self.context.add_message(text, response, 'model_list_available')
        ic = len([m for m in models if m and m.get('installed')])
        return ConversationResult(response=response, intent='model_list_available', memory_action='model_list', context=self.context.to_dict(), model_list=ModelListResult(kind='available', models=models, installed_count=ic, catalog_total=total))

    def _handle_model_list(self, text: str) -> ConversationResult:
        """Responde a listagens genericas: "quais modelos existem?". """
        return self._handle_model_list_available(text)

    def _handle_model_info_query(self, text: str) -> ConversationResult:
        """Responde a consultas sobre um modelo ESPECIFICO: "o qwen 8b esta instalado?", etc."""
        request = extract_model_info_query(text)
        if request is None:
            return ConversationResult(response='Nao consegui interpretar a consulta.', intent='model_info', memory_action='model_info_error', context=self.context.to_dict(), model_info=ModelInfoResult(question=None, target=None, error='consulta nao reconhecida'))
        manager = self.model_manager
        target = request.target or ''
        resolved = manager.resolve_model(target) if target else None
        rid = getattr(resolved, 'id', None) if resolved else None
        rd = self._model_to_dict(resolved) if resolved else None
        if resolved is None:
            r = f'Nao encontrei modelo chamado {target!r} no catalogo. Nada alterado.'
            self.context.add_message(text, r, 'model_info')
            return ConversationResult(response=r, intent='model_info', memory_action='model_info', context=self.context.to_dict(), model_info=ModelInfoResult(question=request.question, target=target, resolved_id=None, resolved=None, installed=None, is_active=None, is_loaded=None, error=f'modelo nao encontrado: {target!r}'))
        ii = resolved.is_installed() if hasattr(resolved, 'is_installed') else False
        active = manager.get_active_model()
        aid = getattr(active, 'id', None) if active else None
        ia = (aid == rid) if aid else False

        try:
            st = manager.get_status()
            ld = st.get('loaded_model')
            il = False
            if ld and rid:
                ln = str(ld).strip().lower()
                rn = rid.lower()
                il = (ln == rn or ln.endswith(f'/{rn}'))
        except Exception:
            il = None
        q = request.question
        if q == 'installed':
            r = f'Sim, o {resolved.name} ({rid}) esta instalado.' if ii else f'O {resolved.name} ({rid}) esta no catalogo mas nao esta instalado no momento.'
        elif q == 'available':
            r = f'Sim, o {resolved.name} ({rid}) esta disponivel e instalado.' if ii else f'Sim, o {resolved.name} ({rid}) esta no catalogo e disponivel para uso (nao instalado).'
        elif q == 'can_use':
            r = f'O {resolved.name} ({rid}) esta no catalogo mas nao esta instalado. Para usar, e necessario instalar o .gguf.' if not ii else f'Sim, o {resolved.name} ({rid}) esta instalado e disponivel para uso.'
        elif q == 'loaded':
            if il is True:
                r = f'Sim, o {resolved.name} ({rid}) esta carregado no runtime agora.'
            elif il is False:
                r = f'O {resolved.name} ({rid}) nao esta carregado no runtime agora.'
            else:
                r = f'Nao consigo confirmar se o {resolved.name} ({rid}) esta carregado no momento.'
        elif q == 'state':
            p = []
            if ii: p.append('instalado')
            if ia: p.append('ativo')
            if il: p.append('carregado')
            if not p: p.append('nao encontrado no runtime')
            r = f'O {resolved.name} ({rid}): ' + ', '.join(p) + '.'
        else:
            r = f'Sobre o {resolved.name} ({rid}): ' + ('instalado' if ii else 'nao instalado') + ', ' + ('ativo' if ia else 'nao ativo') + '.'
        self.context.add_message(text, r, 'model_info')
        return ConversationResult(response=r, intent='model_info', memory_action='model_info', context=self.context.to_dict(), model_info=ModelInfoResult(question=request.question, target=target, resolved_id=rid, resolved=rd, installed=ii, is_active=ia, is_loaded=il, error=None))
    def _handle_question(self, text: str) -> ConversationResult:
        if self.cognitive_core is not None and self._llm_ready():
            try:
                outcome = self.cognitive_core.generate_response(
                    text, context=self.context, intent=Intent.MEMORY_QUERY
                )
                response = self._apply_style(outcome["text"])
                self.context.add_message(text, response, "question")
                return ConversationResult(
                    response=response,
                    intent="question",
                    memory_action="llm_recall",
                    memories_used=outcome.get("memories_used", []),
                    context=self.context.to_dict(),
                )
            except Exception:
                import logging

                logging.getLogger("davios.conversation").warning(
                    "[LLM] falha ao responder pergunta; usando regras", exc_info=True
                )
        response, memories_used = self.response_generator.answer_question(
            text, self.memory_manager, self.context
        )
        self.context.add_message(text, response, "question")
        return ConversationResult(
            response=response,
            intent="question",
            memory_action="recall",
            memories_used=memories_used,
            context=self.context.to_dict(),
        )

    def _handle_llm_question(self, text: str, intent: Intent) -> ConversationResult:
        """Perguntas abertas/opiniao: LLM quando disponivel, regras caso contrario."""
        if self.cognitive_core is not None and self._llm_ready():
            try:
                outcome = self.cognitive_core.generate_response(
                    text, context=self.context, intent=intent
                )
                response = self._apply_style(outcome["text"])
                memories_used = outcome.get("memories_used", [])
                self.context.add_message(text, response, intent.value)
                return ConversationResult(
                    response=response,
                    intent=intent.value,
                    memory_action="llm_context",
                    memories_used=memories_used,
                    context=self.context.to_dict(),
                )
            except Exception as exc:
                import logging

                logging.getLogger("davios.conversation").warning(
                    "[LLM] falha ao gerar resposta: %s", exc
                )
                response = (
                    "Tive um problema ao consultar meu modelo local agora. "
                    "Posso tentar de novo em instantes?"
                )
                self.context.add_message(text, response, intent.value)
                return ConversationResult(
                    response=response,
                    intent=intent.value,
                    memory_action="llm_error",
                    context=self.context.to_dict(),
                )

        # Sem LLM: mantem o comportamento anterior de regras.
        # Com provider LOCAL real e sem modelo instalado, explica como instalar.
        from brain.providers.local_llm_provider import LLAMA_CPP_INSTRUCTIONS, LocalLLMProvider

        if isinstance(getattr(self.cognitive_core, "llm_provider", None), LocalLLMProvider):
            self.context.add_message(text, "modelo ausente", intent.value)
            return ConversationResult(
                response=LLAMA_CPP_INSTRUCTIONS,
                intent=intent.value,
                memory_action="llm_unavailable",
                context=self.context.to_dict(),
            )
        return self._handle_question(text)

    def _llm_ready(self) -> bool:
        try:
            return bool(self.cognitive_core.llm_provider.is_available())
        except Exception:
            return False

    def _apply_style(self, text: str) -> str:
        """Aplica substituicoes de estilo apos a geracao do LLM.

        Qualquer falha no manager nunca quebra a resposta.
        """
        try:
            return self.memory_manager.apply_style(text)
        except Exception:
            return text

    def _update_style_profile(self, text: str) -> None:
        """Atualiza e persiste o perfil de estilo com a mensagem do usuario.

        Falhas sao silenciosas para nao interromper a conversa.
        """
        try:
            self.memory_manager.update_style_profile(text)
        except Exception:
            pass


    def _handle_statement(self, text: str) -> ConversationResult:
        intent = self.classifier.classify(text, context=self.context)
        context = self.memory_manager.context_analyzer.analyze(text)
        emotions = self._build_emotions(context)
        interpretation = MemoryInterpreter().interpret(text, context, emotions)

        if not interpretation["facts"]:
            # Papo aberto / frase sem fato memoravel: LLM se disponivel
            if self.cognitive_core is not None and self._llm_ready():
                try:
                    outcome = self.cognitive_core.generate_response(
                        text, context=self.context, intent=intent
                    )
                    response = self._apply_style(outcome["text"])
                    self.context.add_message(text, response, intent.value)
                    return ConversationResult(
                        response=response,
                        intent=intent.value,
                        memory_action="llm_context",
                        memories_used=outcome.get("memories_used", []),
                        context=self.context.to_dict(),
                    )
                except Exception:
                    pass  # cai no fallback de regras abaixo
            response = self.response_generator.generate_fallback(text, self.context)
            self.context.add_message(text, response, "conversation")
            return ConversationResult(
                response=response,
                intent="conversation",
                memory_action="none",
                context=self.context.to_dict(),
            )

        # A memoria e salva ANTES de gerar a resposta: assim o modelo pode
        # usar a informacao recem-fornecida via bloco de fatos novos.
        memory = Memory(
            content=text,
            memory_type="preference" if interpretation.get("memory_candidate") else "episodic",
            importance=0.7,
            emotion="neutral",
            emotional_intensity=0.0,
            facts=interpretation["facts"],
        )

        match = self.memory_manager.match_memory_for_facts(memory.facts)
        decision = self.reasoning.evaluate_input(
            user_input=text,
            memory_context=context,
            emotions=emotions,
            relevant_memories=[match["matching_fact"]] if match["matching_fact"] else [],
            current_facts=memory.facts,
            existing_memory=match["existing_memory"],
            matching_fact=match["matching_fact"],
        )

        result = self.memory_manager.apply_memory_operation(
            reasoning_result=decision,
            new_memory=memory,
            existing_memory=match["existing_memory"],
        )

        # Resposta natural via LLM usando os fatos recem-salvos; o gerador
        # por regras permanece como fallback deterministico.
        if self.cognitive_core is not None and self._llm_ready():
            try:
                outcome = self.cognitive_core.generate_response(
                    text,
                    context=self.context,
                    intent=decision.intent,
                    new_facts=memory.facts,
                )
                response = self._apply_style(outcome["text"])
                self.context.add_message(text, response, decision.intent)
                topic = self._topic_from_facts(memory.facts)
                if topic:
                    self.context.current_topic = topic
                return ConversationResult(
                    response=response,
                    intent=decision.intent,
                    memory_action=result.get("action", "none"),
                    memories_used=outcome.get("memories_used", []),
                    context=self.context.to_dict(),
                )
            except Exception:
                import logging

                logging.getLogger("davios.conversation").warning(
                    "[LLM] falha apos salvar memoria; usando regras", exc_info=True
                )

        response = self.response_generator.generate_from_memory_result(
            result, decision, text, self.context
        )

        self.context.add_message(text, response, decision.intent)
        topic = self._topic_from_facts(memory.facts)
        if topic:
            self.context.current_topic = topic

        return ConversationResult(
            response=response,
            intent=decision.intent,
            memory_action=result.get("action", "none"),
            memories_used=[match["existing_memory"]] if match["existing_memory"] else [],
            context=self.context.to_dict(),
        )

    # Relacoes cujo alvo pode virar topico global da conversa. Fatos
    # episodicos/casuais NUNCA definem o topico — so working_on, studies,
    # preferencias reais, identidade etc.
    _TOPIC_RELATIONS = frozenset({
        "working_on", "studies", "like", "dislike", "love", "hate",
        "identity", "name", "possession", "lives", "work", "preference",
    })

    @classmethod
    def _topic_from_facts(cls, facts) -> str:
        """Primeiro alvo semanticamente adequado para current_topic
        ('' se nenhum fato for adequado)."""
        for fact in facts or []:
            relation = str(getattr(fact, "relation", "") or "").strip()
            target = str(getattr(fact, "target", "") or "").strip()
            if relation in cls._TOPIC_RELATIONS and target:
                return target
        return ""

    def _build_emotions(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        emotions = []
        for part in context.get("parts", []):
            emotion = self.memory_manager.emotion_analyzer.analyze(
                part["text"], part
            )
            emotions.append(
                {
                    "text": part["text"],
                    "emotion": emotion["emotion"],
                    "emotional_intensity": emotion["emotional_intensity"],
                    "context": part,
                }
            )
        return emotions
