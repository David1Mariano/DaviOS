"""Verifica se a GPU AMD Radeon RX 6600 esta sendo usada pelo llama-server."""

from __future__ import annotations

import sys
import time
import urllib.request
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_gpu_in_use() -> bool:
    """Verifica se a GPU esta sendo usada pelo llama-server."""
    # Verifica processos llama-server.exe
    import psutil
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'llama-server' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'] or [])
                print(f"Processo encontrado: PID={proc.info['pid']}")
                print(f"  CMD: {cmdline[:200]}...")
                
                # Verifica se tem --ngl (GPU layers)
                if '--ngl' in cmdline or '-ngl' in cmdline:
                    print("  -> GPU layers configuradas!")
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def check_gpu_usage() -> None:
    """Verifica o uso de GPU via performance counters."""
    import psutil
    
    # No Windows, podemos verificar o uso de GPU via WMI ou performance counters
    # Vamos verificar se ha processos usando a GPU
    print("\nVerificando uso de GPU...")
    
    # Verifica processos que podem estar usando GPU
    gpu_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            name = proc.info['name'].lower()
            if any(x in name for x in ['llama', 'vulkan', 'amd', 'radeon']):
                gpu_processes.append(proc)
                mem = proc.info['memory_info']
                print(f"  {proc.info['name']} (PID={proc.info['pid']}): RSS={mem.rss / 1024**2:.1f} MB")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if not gpu_processes:
        print("  Nenhum processo relacionado a GPU encontrado.")


def main() -> int:
    print("=" * 60)
    print("Verificacao de GPU AMD Radeon RX 6600")
    print("=" * 60)
    print()
    
    # Verifica se o llama-server esta rodando
    try:
        req = urllib.request.Request("http://127.0.0.1:8080/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"llama-server: ONLINE (status: {resp.read().decode()})")
    except Exception as e:
        print(f"llama-server: OFFLINE ({e})")
        return 1
    
    print()
    gpu_configured = check_gpu_in_use()
    check_gpu_usage()
    
    print()
    if gpu_configured:
        print("RESULTADO: GPU configurada (Vulkan/AMD)")
    else:
        print("RESULTADO: CPU apenas (sem GPU layers)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())