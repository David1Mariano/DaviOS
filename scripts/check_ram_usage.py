"""Verifica processos que mais consomem RAM."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import psutil

print("=" * 60)
print("Processos que mais consomem RAM")
print("=" * 60)

processes = []
for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
    try:
        info = proc.info
        ram_mb = info['memory_info'].rss / 1024 / 1024
        processes.append((info['pid'], info['name'], ram_mb))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

# Sort by RAM usage
processes.sort(key=lambda x: x[2], reverse=True)

print(f"{'PID':<10} {'RAM (MB)':<12} {'Processo'}")
print("-" * 60)
for pid, name, ram_mb in processes[:15]:
    print(f"{pid:<10} {ram_mb:<12.1f} {name}")

print()
print(f"Total de processos: {len(processes)}")

# Total RAM
v = psutil.virtual_memory()
print(f"RAM Total: {v.total / 1024**3:.1f} GB")
print(f"RAM Disponivel: {v.available / 1024**3:.1f} GB")
print(f"RAM Usada: {v.used / 1024**3:.1f} GB")