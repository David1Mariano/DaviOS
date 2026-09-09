"""Providers de LLM do DaviOS."""

from brain.providers.local_llama_cpp_provider import (
    LocalLlamaCppProvider,
    LLAMA_CPP_INSTRUCTIONS as LLAMA_CPP_STANDALONE_INSTRUCTIONS,
)
from brain.providers.local_llm_provider import (
    LocalLLMProvider,
    LLAMA_CPP_INSTRUCTIONS,
)

__all__ = [
    "LocalLlamaCppProvider",
    "LocalLLMProvider",
    "LLAMA_CPP_INSTRUCTIONS",
    "LLAMA_CPP_STANDALONE_INSTRUCTIONS",
]
