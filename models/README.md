# Modelos Locais

O DaviOS usa modelos **GGUF** executados localmente via **llama.cpp standalone**
(`bin/llama.cpp/llama-server.exe`) — sem depender de `llama-cpp-python`.
Nenhum modelo é enviado ao Git — apenas este README.

## Estrutura

```
models/
├── light/         # <= 1.5 GB  (PCs fracos)
├── balanced/      # <= 4.5 GB  (PCs intermediarios) — modelo padrão
└── performance/   # <= 8.0 GB  (PCs fortes, >= 16 GB RAM e/ou >= 6 GB VRAM)
```

O `ModelManager` escolhe automaticamente a pasta do perfil detectado
(`LIGHT`/`BALANCED`/`PERFORMANCE`) e o maior modelo compatível dentro dela.

## Modelo recomendado (primeira versão)

**Qwen3-4B-GGUF — quantização Q4_K_M**

| Item | Valor |
|---|---|
| Arquivo | `Qwen3-4B-Q4_K_M.gguf` |
| Tamanho | ~2.5 GB |
| Fonte oficial | `Qwen/Qwen3-4B-GGUF` (Hugging Face) |
| Alternativa | `ggml-org/Qwen3-4B-GGUF` |
| Destino | `models/balanced/Qwen3-4B-Q4_K_M.gguf` |

Escolha documentada: o Qwen3-4B é suficientemente capaz para conversação
natural e assistência em programação, roda em CPU (Ryzen 5 5600) com RAM
confortável e pode usar GPU AMD via Vulkan. Não usamos o 0.5B como cérebro
definitivo.

## Instalação

1. Instale o backend (llama.cpp standalone, opcional se você já baixou
   os binários manualmente):

   ```powershell
   .\.venv\Scripts\python.exe scripts\setup_llama.py
   ```

2. Baixe o modelo (download retomável, com `.part` e lock):

   ```powershell
   .\.venv\Scripts\python.exe scripts\download_model.py
   ```

   Se o download automático falhar por conexão instável, baixe
   `Qwen3-4B-Q4_K_M.gguf` manualmente de
   `https://huggingface.co/Qwen/Qwen3-4B-GGUF` e coloque em
   `models/balanced/`. Um `.part` existente será retomado na próxima
   execução.

3. Reinicie o DaviOS:

   ```powershell
   python main.py
   ```

## GPU (opcional)

A GPU não é obrigatória. Sem GPU, tudo roda em CPU.

- **AMD (ex.: Radeon RX 6600):** use o build **Vulkan** do llama.cpp
  (`setup_llama.py` baixa a versão `win-vulkan-x64` por padrão) e ative
  `use_gpu: true` + `gpu_layers` em `config/davios.json` (ou
  `DAVIOS_USE_GPU=1`).
- Qualquer falha de offload faz fallback automático para CPU.

## Offline

Depois que `bin/llama.cpp/` e o modelo `.gguf` estão instalados, o DaviOS
funciona **completamente offline**: sem Internet, sem DNS, sem API, sem
assinatura. O `llama-server.exe` roda em `127.0.0.1` (IPC local, não é
Internet).
