"""PromptBuilder do DaviOS.

Monta o contexto enviado ao modelo de linguagem:
    instruções de sistema + personalidade
    + contexto conversacional
    + memórias relevantes
    + ferramentas disponíveis (opcional, C3)
    + mensagem atual

Os prompts vivem aqui (e na config), nunca espalhados pelo código.

NOTA (C3/C4b): a seção de ferramentas lista nome/descrição/argumentos e,
desde a C4b, inclui as INSTRUÇÕES do protocolo de solicitação
(``<<<TOOL_REQUEST>>>`` ... ``<<<END_TOOL_REQUEST>>>``, definidas em
brain/tool_request_parser.py — constantes reaproveitadas, nunca
reescritas aqui). Só entra no prompt quando
``config.tools_visible_to_llm`` está ligada E um ToolRegistry é passado.
Sem a flag (padrão), o prompt gerado é idêntico, byte a byte, ao de antes.
O texto do Qwen que eventualmente contiver o bloco AINDA NÃO é lido por
ninguém — a conexão parser → execução é C5+.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from config.davios_config import DaviosConfig
from brain.tool_registry import ToolRegistry
from brain.tool_request_parser import END_MARKER, START_MARKER
from personality.personality import Personality, DEFAULT_PERSONALITY


@dataclass
class BuiltPrompt:
    """Prompt estruturado pronto para o LLMProvider."""

    system: str
    prompt: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }


class PromptBuilder:
    """Constrói prompts a partir de contexto, memórias e personalidade."""

    def __init__(
        self,
        config: Optional[DaviosConfig] = None,
        personality: Optional[Personality] = None,
    ):
        self.config = config or DaviosConfig.load()
        self.personality = personality or DEFAULT_PERSONALITY

    def build(
        self,
        user_message: str,
        *,
        context=None,
        memories: Optional[list[Any]] = None,
        intent: str = "conversation",
        resolved_context: Optional[str] = None,
        new_facts: Optional[list[Any]] = None,
        style_profile: Optional[Any] = None,
        tools_registry: Optional[ToolRegistry] = None,
    ) -> BuiltPrompt:
        """Monta o prompt final com memórias relevantes e histórico recente.

        ``new_facts`` são fatos extraídos da própria mensagem atual (já
        salvos na memória) — entram em bloco próprio para que o modelo use
        a informação recém-fornecida.

        ``style_profile`` é o perfil de estilo de fala do usuário. Quando
        fornecido e há dados suficientes, sua descrição é incluida no texto
        de sistema para orientar o modelo a responder no mesmo registro.

        ``tools_registry`` é o ToolRegistry opcional (C3). A seção de
        ferramentas só entra no prompt se ``config.tools_visible_to_llm``
        estiver ligada E o registry for fornecido e não estiver vazio.
        Sem a flag (padrão), o prompt é idêntico, byte a byte, ao anterior.
        """
        memory_block = self._format_memories(memories or [])
        new_facts_block = self._format_new_facts(new_facts or [])
        history_block = self._format_history(context)
        topic = getattr(context, "current_topic", "") if context else ""
        style_text = self._format_style(style_profile)

        tools_block = ""
        if self.config.tools_visible_to_llm:
            tools_block = self.build_tools_section(tools_registry)

        sections: list[str] = []

        if memory_block:
            sections.append(
                f"Informacoes que voce sabe sobre o usuario:\n{memory_block}"
            )

        if new_facts_block:
            sections.append(
                "Informacoes que o usuario acabou de informar nesta mensagem:"
                f"\n{new_facts_block}"
            )

        if history_block:
            sections.append(f"Conversa recente:\n{history_block}")

        if topic:
            sections.append(f"Assunto atual da conversa: {topic}")

        if resolved_context:
            sections.append(
                f"Referencia a mensagem anterior:\n{resolved_context}"
            )

        if tools_block:
            sections.append(tools_block)

        sections.append(f"Mensagem do usuario: {user_message}")

        prompt = "\n\n".join(sections)

        metadata: dict[str, Any] = {
            "intent": intent,
            "memories_count": len(memories or []),
            "new_facts_count": len(new_facts or []),
            "history_messages": len(getattr(context, "messages", []) or []),
            "style_applied": bool(style_text),
        }

        if tools_block:
            metadata["tools_in_prompt"] = True
            metadata["tools_count"] = len(tools_registry.list_tools())

        return BuiltPrompt(
            system=self._system_text(style_text),
            prompt=prompt,
            metadata=metadata,
        )

    def _system_text(self, style_text: str = "") -> str:
        # system_prompt condicional: quando tools_visible_to_llm=True, usa
        # a versão SEM a frase de negação de capacidades (que conflitaria
        # com a seção de ferramentas). Com a flag off, usa o system_prompt
        # original (byte a byte idêntico ao comportamento anterior).
        if self.config.tools_visible_to_llm:
            config_system = (
                self.config.system_prompt_tools_enabled or ""
            ).strip()
        else:
            config_system = (self.config.system_prompt or "").strip()

        parts = [
            config_system,
            self.personality.to_system_text(),
            (style_text or "").strip(),
        ]

        return "\n\n".join(part for part in parts if part)

    def build_tools_section(self, registry: Optional[ToolRegistry]) -> str:
        """Gera a seção de ferramentas do prompt (C3 + instruções C4b).

        Conteúdo:
            1. Listagem informativa (nome, descrição, argumentos) — C3;
            2. Instruções do protocolo de solicitação — C4b, com os
               marcadores EXATOS importados de brain/tool_request_parser.py
               (fonte única de verdade; nunca reescritos aqui);
            3. Exemplo concreto usando uma ferramenta REAL do registry
               passado (nunca um nome inventado).

        As instruções refletem a decisão do C4: apenas a PRIMEIRA
        solicitação por resposta é considerada, e o bloco contém apenas
        o JSON (sem comentários/explicações dentro dele).

        - registry None ou vazio → "" (nem listagem, nem instruções);
        - registry quebrado nunca quebra a geração.
        """
        if registry is None:
            return ""

        try:
            tools = registry.list_tools()
        except Exception:
            return ""

        if not tools:
            return ""

        # --- 1. Listagem (C3) ---
        lines = ["### FERRAMENTAS DISPONIVEIS"]

        for tool in tools:
            name = getattr(tool, "name", "") or ""
            description = getattr(tool, "description", "") or ""
            arguments = getattr(tool, "arguments", None) or []

            if arguments:
                args_text = ", ".join(str(a) for a in arguments)
                lines.append(
                    f"- {name}: {description} "
                    f"(argumentos: {args_text})"
                )
            else:
                lines.append(
                    f"- {name}: {description} (sem argumentos)"
                )

        # --- 2. Instruções do protocolo (C4b) ---
        template_block = (
            f"{START_MARKER}\n"
            '{"tool": "<nome>", "arguments": {"<argumento>": "<valor>"}}\n'
            f"{END_MARKER}"
        )

        lines.append("")
        lines.append(
            "Use uma ferramenta SOMENTE quando a resposta exigir dados "
            "externos que voce nao possui (hora atual, conteudo de "
            "arquivo, pagina web). Conversa casual, cumprimentos, "
            "agradecimentos, reacoes como kkk, boa, serio?, opinoes, "
            "explicacoes, piadas e mensagens que nao exigem acesso a "
            "dados externos devem ser respondidas diretamente, SEM "
            "gerar TOOL_REQUEST."
        )

        lines.append(
            "Para usar uma ferramenta, inclua na resposta um bloco "
            "exatamente neste formato:"
        )

        lines.append(template_block)

        lines.append(
            "Regras: use somente ferramentas da lista acima; envie no "
            "maximo UM pedido por resposta (apenas o primeiro sera "
            "considerado); dentro do bloco escreva apenas o JSON, sem "
            "comentarios ou texto extra. Use ferramenta somente quando "
            "houver necessidade operacional clara (ex: que horas sao, "
            "leia main.py); nao use para conversa social. "
            "echo so deve ser usada quando o usuario pedir explicitamente "
            "para repetir/ecoar texto."
        )

        # --- 3. Exemplo com ferramenta real do registry ---
        example = self._tools_example_block(tools)

        if example:
            lines.append("Exemplo:")
            lines.append(example)

        # --- 4. Exemplos few-shot ---
        few_shot = self._few_shot_examples(tools)

        if few_shot:
            lines.append("Exemplos:")
            lines.append(few_shot)

        return "\n".join(lines)

    @staticmethod
    def _tools_example_block(tools: list[Any]) -> str:
        """Monta um exemplo usando uma ferramenta operacional real.

        A ordem de preferência é deliberada:

            1. time
            2. datetime
            3. read_file
            4. list_dir
            5. web_fetch
            6. outra ferramenta com argumentos
            7. primeira ferramenta disponível

        ``echo`` fica propositalmente fora da prioridade porque é uma
        ferramenta de repetição, não uma boa ferramenta para ensinar ao
        modelo o conceito geral de quando deve usar ferramentas.

        Nunca inventa nomes de ferramentas nem argumentos.
        """
        if not tools:
            return ""

        # Índice por nome para tornar a escolha determinística,
        # independentemente da ordem do ToolRegistry.
        tools_by_name: dict[str, Any] = {}

        for tool in tools:
            name = getattr(tool, "name", "") or ""
            if name and name not in tools_by_name:
                tools_by_name[name] = tool

        # Ferramentas que representam necessidades operacionais reais.
        preferred_names = (
            "time",
            "datetime",
            "read_file",
            "list_dir",
            "web_fetch",
        )

        chosen = None

        for name in preferred_names:
            if name in tools_by_name:
                chosen = tools_by_name[name]
                break

        # Fallback: alguma ferramenta com argumentos declarados.
        if chosen is None:
            for tool in tools:
                if getattr(tool, "arguments", None):
                    chosen = tool
                    break

        # Último fallback: primeira ferramenta real.
        if chosen is None:
            chosen = tools[0]

        chosen_name = getattr(chosen, "name", "") or ""
        chosen_args = getattr(chosen, "arguments", None) or []

        if not chosen_name:
            return ""

        if chosen_args:
            args_json = ", ".join(
                f'"{str(argument)}": "<valor>"'
                for argument in chosen_args
            )

            inner = (
                f'{{"tool": "{chosen_name}", '
                f'"arguments": {{{args_json}}}}}'
            )
        else:
            inner = f'{{"tool": "{chosen_name}"}}'

        return (
            f"{START_MARKER}\n"
            f"{inner}\n"
            f"{END_MARKER}"
        )

    @staticmethod
    def _few_shot_examples(tools: list[Any]) -> str:
        """Gera exemplos few-shot determinísticos.

        Os nomes de argumento em cada exemplo são derivados DINAMICAMENTE
        do proprio objeto Tool registrado (tool.arguments), nunca escritos
        fixos no codigo — isso evita que um exemplo ensine um nome de
        argumento errado se a ferramenta real usar outro nome.

        NAO incluímos um exemplo negativo com resposta fixa (ex: "valeu!"
        -> "Por nada!") porque modelos locais pequenos tendem a memorizar
        e repetir literalmente esse texto para qualquer mensagem seguinte,
        travando a conversa em loop assim que a resposta entra no
        historico. A orientacao de quando NAO usar ferramenta ja esta
        coberta pela instrucao em texto corrido, antes desta secao.

        Prioriza ferramentas curadas (time, read_file, datetime, echo,
        web_fetch) com valores de exemplo naturais; se nenhuma estiver no
        registry, usa um fallback generico com a primeira ferramenta
        disponivel, tambem com argumentos reais.

        Limita os exemplos positivos a 2.
        """
        if not tools:
            return ""

        tools_by_name: dict[str, Any] = {}
        for tool in tools:
            name = getattr(tool, "name", "") or ""
            if name and name not in tools_by_name:
                tools_by_name[name] = tool

        def _build_block(tool_name: str, example_value: Optional[str]) -> str:
            tool = tools_by_name[tool_name]
            arg_names = getattr(tool, "arguments", None) or []
            if arg_names and example_value is not None:
                key = str(arg_names[0])
                inner = (
                    '{"tool": "%s", "arguments": {"%s": "%s"}}'
                    % (tool_name, key, example_value)
                )
            else:
                inner = '{"tool": "%s"}' % tool_name
            return f"{START_MARKER}\n{inner}\n{END_MARKER}"

        curated = [
            ("time", "Que horas sao agora?", None),
            ("read_file", "O que tem no arquivo main.py?", "main.py"),
            ("datetime", "Que dia e hoje?", None),
            ("echo", "Repita exatamente: teste 123", "teste 123"),
            ("web_fetch", "O que tem na pagina http://exemplo.com?", "http://exemplo.com"),
        ]

        exemplos: list[str] = []

        for tool_name, prompt_text, example_value in curated:
            if len(exemplos) >= 2:
                break
            if tool_name not in tools_by_name:
                continue
            if tool_name == "web_fetch":
                args = getattr(tools_by_name[tool_name], "arguments", None) or []
                if "url" not in [str(a) for a in args]:
                    continue
            bloco = _build_block(tool_name, example_value)
            exemplos.append(
                'Pergunta do usuario: "%s"\nResposta correta:\n%s'
                % (prompt_text, bloco)
            )

        # Fallback generico: nenhuma ferramenta curada no registry.
        if not exemplos:
            first = tools[0]
            name = getattr(first, "name", "") or ""
            if name:
                arg_names = getattr(first, "arguments", None) or []
                if arg_names:
                    key = str(arg_names[0])
                    inner = (
                        '{"tool": "%s", "arguments": {"%s": "<valor>"}}'
                        % (name, key)
                    )
                else:
                    inner = '{"tool": "%s"}' % name
                bloco = f"{START_MARKER}\n{inner}\n{END_MARKER}"
                exemplos.append(
                    'Pergunta do usuario relacionada a %s.\nResposta correta:\n%s'
                    % (name, bloco)
                )

        return "\n\n".join(exemplos)

    @staticmethod
    def _format_style(style_profile: Optional[Any]) -> str:
        """Extrai a descricao de estilo do perfil, se houver dados suficientes.

        Retorna "" quando nao ha perfil ou o proprio perfil decide que ainda
        nao ha dados suficientes. Um perfil quebrado nunca quebra a geracao.
        """
        if style_profile is None:
            return ""

        try:
            if hasattr(style_profile, "to_instruction"):
                text = style_profile.to_instruction()
            else:
                text = style_profile.to_prompt_description()
        except Exception:
            return ""

        return text.strip() if isinstance(text, str) else ""

    def _format_fact_line(self, fact: Any) -> Optional[str]:
        """Formata um MemoryFact como linha legivel para o modelo."""
        target = getattr(fact, "target", None)

        if target is None and isinstance(fact, dict):
            target = fact.get("target")

        relation = getattr(fact, "relation", None)

        if relation is None and isinstance(fact, dict):
            relation = fact.get("relation")

        negation = bool(getattr(fact, "negation", False))

        value = getattr(fact, "value", None)

        if value is None and isinstance(fact, dict):
            value = fact.get("value")

        if target == "nome" and value:
            return f"- Nome do usuario: {value}"

        if not target or not relation:
            return None

        if relation in ("like", "love"):
            state = "nao gosta" if negation else "gosta"
            return f"- O usuario {state} de {target}"

        if relation in ("dislike", "hate"):
            state = "nao gosta" if negation else "nao gosta"
            return f"- O usuario {state} de {target}"

        if relation == "identity":
            return f"- {target}: {value or 'desconhecido'}"

        if relation == "working_on":
            state = (
                "nao esta trabalhando em"
                if negation
                else "esta trabalhando em"
            )
            return f"- O usuario {state} {target}"

        if relation == "studies":
            state = (
                "nao esta estudando"
                if negation
                else "esta estudando"
            )
            return f"- O usuario {state} {target}"

        if relation == "lives_in":
            return f"- O usuario mora em {target}"

        if relation == "has":
            return f"- O usuario tem {target}"

        return f"- {target}: {relation}"

    def _format_memories(self, memories: list[Any]) -> str:
        lines: list[str] = []

        for fact in memories:
            line = self._format_fact_line(fact)

            if line:
                lines.append(line)

        return "\n".join(
            lines[: self.config.max_memories_in_prompt]
        )

    def _format_new_facts(self, facts: list[Any]) -> str:
        lines: list[str] = []

        for fact in facts:
            line = self._format_fact_line(fact)

            if line:
                lines.append(line)

        return "\n".join(lines)

    def _format_history(self, context) -> str:
        if not context:
            return ""

        messages = getattr(context, "messages", []) or []
        recent = messages[-self.config.max_recent_messages:]

        lines = [
            "Conversa anterior - so contexto, nao exemplo: nao copie nem imite "
            "o formato ou encerramento das respostas anteriores."
        ]

        for item in recent:
            user = item.get("user", "")
            response = item.get("response", "")

            lines.append(f"Usuario: {user}")
            lines.append(f"DaviOS: {response}")

        return "\n".join(lines)