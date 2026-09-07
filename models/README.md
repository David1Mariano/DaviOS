# Modelos Locais

O DaviOS usa modelos **GGUF** executados localmente (llama-cpp-python).
Nenhum modelo é enviado ao Git — apenas este README.

## Estrutura

```
models/
├── light/         # <= 1.5 GB  (PCs fracos)
├── balanced/      # <= 4.5 GB  (PCs intermediarios)
└── performance/   # <= 8.0 GB  (PCs fortes, >= 16 GB RAM e/ou >= 6 GB VRAM)
```

O `ModelManager` escolhe automaticamente a pasta do perfil detectado
(`LIGHT`/`BALANCED`/`PERFORMANCE`) e o maior modelo compatível dentro dela.

## Como instalar um modelo (recomendado para a maioria dos PCs)

1. Instale o backend (opcional, uma vez):

   ```powershell
   .\.venv\Scripts\pip.exe install llama-cpp-python
   ```

2. Baixe **um** modelo GGUF instruído, por exemplo:

   | Perfil | Sugestão | Tamanho aprox. | Roda em CPU? |
   |---|---|---|---|
   | light  | `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` | ~0.4 GB | sim |
   | balanced | `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` | ~1.0 GB | sim |
   | performance | `Qwen2.5-3B-Instruct-Q4_K_M.gguf` | ~2.0 GB | sim |

   Fontes: Hugging Face (ex.: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`,
   `TheBloke/...`). O download é manual — o DaviOS **nunca baixa
   modelos automaticamente**.

3. Copie o arquivo `.gguf` para a pasta do perfil correspondente
   (`models/balanced/` na dúvida).

4. Reinicie o DaviOS. Com `DAVIOS_DEBUG=1` o boot mostra o modelo
   detectado.

## GPU (opcional)

A GPU não é obrigatória. Sem GPU, tudo roda em CPU.

- **AMD (ex.: Radeon RX 6600):** builds de `llama-cpp-python` com
  **Vulkan** ou **ROCm/HIP** suportam offload. Instale a variante
  apropriada do `llama-cpp-python` e ative `use_gpu: true` +
  `gpu_layers` no `config/davios.json` (ou `DAVIOS_USE_GPU=1`).
- **NVIDIA:** build com CUDA (`CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`).
- Qualquer falha de offload faz fallback automático para CPU.
