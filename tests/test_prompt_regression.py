"""Teste de regressão byte-a-byte: prompt com flags novas default (False) 
deve ser idêntico ao prompt sem as chaves novas no config.

IMPORTANTE: O PromptBuilder lista TODAS as tools do registry quando 
tools_visible_to_llm=True, independentemente das flags de categoria.
As flags (actions_enabled, web_enabled, phone_read_enabled, phone_act_enabled)
controlam EXECUÇÃO (no ToolExecutionOrchestrator), não VISIBILIDADE no prompt.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.prompt_builder import PromptBuilder
from brain.tool_registry import Tool, ToolRegistry
from config.davios_config import DaviosConfig
from personality.personality import DEFAULT_PERSONALITY


def test_prompt_byte_a_byte_com_flags_phone_default_false():
    """Com phone_read_enabled=False e phone_act_enabled=False (default),
    o prompt deve ser IDÊNTICO ao prompt gerado com um config que não tem
    essas chaves (pois o PromptBuilder não filtra por categoria no prompt).
    
    As flags de categoria só afetam EXECUÇÃO, não o prompt.
    """
    
    # Config COM as chaves novas (default False)
    config_com_chaves = DaviosConfig(
        tools_visible_to_llm=True,
        actions_enabled=True,
        web_enabled=False,
        phone_read_enabled=False,
        phone_act_enabled=False,
        phone_base_url="",
        phone_timeout=10.0,
    )
    
    # Config SEM as chaves novas (simula config antigo - mesmo defaults)
    config_sem_chaves = DaviosConfig(
        tools_visible_to_llm=True,
        actions_enabled=True,
        web_enabled=False,
        # phone_read_enabled, phone_act_enabled, phone_base_url, phone_timeout
        # não passados - usam defaults do dataclass (False, False, "", 10.0)
    )
    
    # Registry com tools actions + web + phone
    reg = ToolRegistry()
    reg.register(Tool(name="datetime", description="Data/hora", arguments=[], category="actions"))
    reg.register(Tool(name="echo", description="Eco", arguments=["args"], category="actions"))
    reg.register(Tool(name="web_fetch", description="Busca web", arguments=["url"], category="web"))
    reg.register(Tool(name="phone_battery", description="Bateria", arguments=[], category="phone_read"))
    reg.register(Tool(name="phone_speak", description="Falar", arguments=["text"], category="phone_act"))
    
    builder = PromptBuilder(config_com_chaves, DEFAULT_PERSONALITY)
    built = builder.build(
        user_message="teste",
        context=None,
        memories=[],
        intent="CHAT",
        style_profile=None,
        tools_registry=reg,
    )
    
    builder2 = PromptBuilder(config_sem_chaves, DEFAULT_PERSONALITY)
    built2 = builder2.build(
        user_message="teste",
        context=None,
        memories=[],
        intent="CHAT",
        style_profile=None,
        tools_registry=reg,
    )
    
    # Prompt e system devem ser byte-a-byte idênticos
    assert built.prompt == built2.prompt, "Prompt difere com flags phone_* default False"
    assert built.system == built2.system, "System difere com flags phone_* default False"
    # Metadados também devem ser iguais (mesmo número de tools)
    assert built.metadata["tools_count"] == built2.metadata["tools_count"]


def test_prompt_lista_todas_tools_quando_tools_visible_true():
    """Quando tools_visible_to_llm=True, TODAS as tools do registry aparecem
    no prompt, independentemente das flags de categoria (que só bloqueiam execução)."""
    
    config = DaviosConfig(
        tools_visible_to_llm=True,
        actions_enabled=False,      # actions desligado
        web_enabled=False,          # web desligado
        phone_read_enabled=False,   # phone_read desligado
        phone_act_enabled=False,    # phone_act desligado
        phone_base_url="",
        phone_timeout=10.0,
    )
    
    reg = ToolRegistry()
    reg.register(Tool(name="datetime", description="Data/hora", arguments=[], category="actions"))
    reg.register(Tool(name="web_fetch", description="Busca web", arguments=["url"], category="web"))
    reg.register(Tool(name="phone_battery", description="Bateria", arguments=[], category="phone_read"))
    reg.register(Tool(name="phone_speak", description="Falar", arguments=["text"], category="phone_act"))
    
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    built = builder.build(
        user_message="teste",
        context=None,
        memories=[],
        intent="CHAT",
        style_profile=None,
        tools_registry=reg,
    )
    
    prompt = built.prompt
    
    # TODAS as tools aparecem no prompt (visibilidade != execução)
    assert "datetime" in prompt
    assert "web_fetch" in prompt
    assert "phone_battery" in prompt
    assert "phone_speak" in prompt
    
    # Metadados contam todas
    assert built.metadata["tools_count"] == 4
    assert built.metadata["tools_in_prompt"] is True


def test_prompt_sem_tools_section_quando_tools_visible_false():
    """Quando tools_visible_to_llm=False, NENHUMA tool aparece no prompt
    (mesmo com flags de categoria ligadas). Comportamento byte-a-byte idêntico
    ao pré-C3."""
    
    config = DaviosConfig(
        tools_visible_to_llm=False,  # FLAG PRINCIPAL DESLIGADA
        actions_enabled=True,
        web_enabled=True,
        phone_read_enabled=True,
        phone_act_enabled=True,
        phone_base_url="http://fake",
        phone_timeout=10.0,
    )
    
    reg = ToolRegistry()
    reg.register(Tool(name="datetime", description="Data/hora", arguments=[], category="actions"))
    reg.register(Tool(name="phone_battery", description="Bateria", arguments=[], category="phone_read"))
    reg.register(Tool(name="phone_speak", description="Falar", arguments=["text"], category="phone_act"))
    
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    built = builder.build(
        user_message="teste",
        context=None,
        memories=[],
        intent="CHAT",
        style_profile=None,
        tools_registry=reg,
    )
    
    prompt = built.prompt
    
    # NENHUMA seção de ferramentas
    assert "FERRAMENTAS DISPONIVEIS" not in prompt
    assert "datetime" not in prompt
    assert "phone_battery" not in prompt
    assert "phone_speak" not in prompt
    assert "TOOL_REQUEST" not in prompt
    
    # Metadados
    assert "tools_in_prompt" not in built.metadata or built.metadata.get("tools_in_prompt") is False
    # System prompt deve ser o original (com frase de negação)
    assert "nao pode executar comandos" in built.system