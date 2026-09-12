import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brain.prompt_builder import PromptBuilder
from brain.tool_adapters import register_default_tools
from brain.tool_registry import ToolRegistry
from brain.tool_request_parser import parse_tool_request, extract_all_requests
from config.davios_config import DaviosConfig
from personality.personality import DEFAULT_PERSONALITY

def test_read_file_few_shot_example_in_prompt():
    """Few-shot example de read_file presente no prompt quando flag ligada."""
    config = DaviosConfig()
    config.tools_visible_to_llm = True
    reg = ToolRegistry()
    register_default_tools(reg)
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    built = builder.build("Oi", tools_registry=reg)
    assert "read_file" in built.prompt
    assert "O que tem no arquivo" in built.prompt


def test_read_file_few_shot_recognized_by_parser():
    """Exemplo few-shot de read_file reconhecido pelo parser."""
    config = DaviosConfig()
    config.tools_visible_to_llm = True
    reg = ToolRegistry()
    register_default_tools(reg)
    builder = PromptBuilder(config, DEFAULT_PERSONALITY)
    built = builder.build("Oi", tools_registry=reg)
    blocks = extract_all_requests(built.prompt)
    read_file_blocks = [b for b in blocks if b.tool_name == "read_file"]
    assert len(read_file_blocks) >= 1
    assert read_file_blocks[0].found is True
    assert read_file_blocks[0].error == ""
    assert read_file_blocks[0].arguments == {"args": "main.py"}
