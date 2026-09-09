# DaviOS - Guia de Instalação Offline

## Estado Atual

O código está 100% funcional e testado (127 testes passando).

**Falta apenas:**
1. Binário do llama.cpp (34 MB)
2. Modelo Qwen3-4B Q4_K_M (2.5 GB)

## Download Manual (Internet Instável)

Se o download automático falhar, baixe manualmente:

### 1. llama.cpp (Backend)

**Build recomendado (Vulkan para AMD RX 6600):**
```
https://github.com/ggml-org/llama.cpp/releases/download/b10853/llama-b10853-bin-win-vulkan-x64.zip
```

**Alternatão (CPU-only, fallback):**
```
https://github.com/ggml-org/llama.cpp/releases/download/b10853/llama-b10853-bin-win-cpu-x64.zip
```

**Instalação:**
1. Extraia o zip em: `bin\llama.cpp\`
2. Deve conter `llama-server.exe` e `llama-cli.exe`

### 2. Modelo Qwen3-4B Q4_K_M

**Fonte oficial (HuggingFace):**
```
https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
```

**Alternativa (ggml-org):**
```
https://huggingface.co/ggml-org/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf
```

**Instalação:**
1. Coloque o arquivo em: `models\balanced\Qwen3-4B-Q4_K_M.gguf`
2. Tamanho esperado: ~2.5 GB

## Download Automático

Se a Internet estiver funcionando, execute:

```bash
# Instalar llama.cpp
python scripts/setup_llama.py

# Baixar modelo
python scripts/download_model.py
```

Os scripts suportam **retomada**: se o download interromper, execute novamente para continuar de onde parou.

## Execução Offline

Após instalar os componentes:

```bash
python main.py
```

O DaviOS funciona **sem Internet** após a instalação.

## Estrutura de Arquivos

```
DaviOS/
├── bin/
│   └── llama.cpp/          # Binário do llama.cpp
│       ├── llama-server.exe
│       └── llama-cli.exe
├── models/
│   └── balanced/
│       └── Qwen3-4B-Q4_K_M.gguf   # Modelo (2.5 GB)
├── main.py                  # Executar assim
└── ...
```

## Testes

```bash
pytest tests -q
```

127 testes passando.

## Configuração

O DaviOS detecta automaticamente:
- CPU (Ryzen 5 5600)
- GPU (RX 6600 com Vulkan)
- RAM (16 GB)

E configura o melhor backend automaticamente.
