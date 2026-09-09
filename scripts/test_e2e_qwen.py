"""Teste de integracao ponta-a-ponta com Qwen3-4B."""
import tempfile
import os
import sys
import io
import logging

# Forca UTF-8 no stdout para evitar erro com emojis do modelo
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.WARNING)

# Adiciona o diretorio raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.davios_config import DaviosConfig
from core.hardware_detector import HardwareDetector
from brain.model_manager import ModelManager
from brain.providers.local_llama_cpp_provider import LocalLlamaCppProvider
from brain.cognitive_core import CognitiveCore
from brain.prompt_builder import PromptBuilder
from brain.conversation_engine import ConversationEngine
from memory.memory_manager import MemoryManager

fd, path = tempfile.mkstemp(suffix=".db")
os.close(fd)
os.remove(path)

c = DaviosConfig.load()
h = HardwareDetector().detect()
mm = ModelManager(c)
s = mm.select_model(h)

provider = LocalLlamaCppProvider(config=c, model_manager=mm, selection=s)
provider.initialize()

manager = MemoryManager(db_path=path)
core = CognitiveCore(
    llm_provider=provider, config=c,
    prompt_builder=PromptBuilder(c), memory_manager=manager,
)
engine = ConversationEngine(memory_manager=manager, cognitive_core=core)

print("=== FALA 1 (armazenar) ===")
for msg in ["Meu nome e Davi", "Eu gosto de Python", "Estou trabalhando no NetOptimizer"]:
    r = engine.process(msg)
    print(f"Q: {msg}")
    print(f"A: {r.response}")
    print()

provider.unload()
manager.database.close()

print("=== FALA 2 (recuperar - nova instancia, mesma memoria) ===")
provider2 = LocalLlamaCppProvider(config=c, model_manager=mm, selection=s)
provider2.initialize()
manager2 = MemoryManager(db_path=path)
core2 = CognitiveCore(
    llm_provider=provider2, config=c,
    prompt_builder=PromptBuilder(c), memory_manager=manager2,
)
engine2 = ConversationEngine(memory_manager=manager2, cognitive_core=core2)
for msg in ["Qual e meu nome?", "Que linguagem eu gosto?", "Em qual projeto estou trabalhando?"]:
    r = engine2.process(msg)
    print(f"Q: {msg}")
    print(f"A: {r.response}")
    print()

provider2.unload()
manager2.database.close()
os.remove(path)

print("=== FIM ===")