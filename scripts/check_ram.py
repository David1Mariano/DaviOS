"""Verifica RAM disponivel."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil

v = psutil.virtual_memory()
print(f"RAM Total: {v.total / 1024**3:.1f} GB")
print(f"RAM Disponivel: {v.available / 1024**3:.1f} GB")
print(f"RAM Usada: {v.used / 1024**3:.1f} GB")
print(f"Percentual usado: {v.percent}%")

# Verificar se Qwen3-4B cabe (precisa ~3.3 GB)
qwen3_size_gb = 2.4
needed = qwen3_size_gb * 1.4
print()
print(f"Qwen3-4B precisa de ~{needed:.1f} GB de RAM disponivel")
print(f"Cabe na RAM: {v.available / 1024**3 >= needed}")