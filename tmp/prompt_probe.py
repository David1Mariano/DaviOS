import sys, os
sys.path.insert(0, os.getcwd())
from config.davios_config import DaviosConfig
from brain.prompt_builder import PromptBuilder
from personality.personality import DEFAULT_PERSONALITY
from brain.tool_registry import ToolRegistry
from brain.tool_adapters import register_default_tools
from brain.action_manager import ActionManager

class Ctx:
    messages = [
        {"role": "user", "content": "eai"},
        {"role": "assistant", "content": "Eae! Como posso te ajudar hoje? \ud83d\ude0a"},
        {"role": "user", "content": "vc me faz rir kkk"},
        {"role": "assistant", "content": "Que bom! Adoro te fazer rir kkk"},
    ]
    current_topic = "rir"

cfg = DaviosConfig()
reg = ToolRegistry()
register_default_tools(reg, action_manager=ActionManager(enabled=True), web_access=None)
pb = PromptBuilder(cfg, DEFAULT_PERSONALITY)
for msg in ["quanto de memoria ram voce ta usando agora?", "te amo", "fala pra mim que horas sao"]:
    built = pb.build(msg, context=Ctx(), memories=["- gosta de: rir (fonte: frase do usuario)"], intent="conversation", tools_registry=reg)
    print('='*70)
    print('INPUT:', msg)
    print('--- SYSTEM ---')
    print(built.system)
    print('--- PROMPT ---')
    print(built.prompt)
