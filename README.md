 # DaviOS 🧠

> Assistente pessoal de IA local, com memória persistente, personalidade e arquitetura preparada para agentes e ferramentas.

## 📌 Sobre o projeto

O **DaviOS** é um projeto pessoal de assistente de inteligência artificial desenvolvido em Python.

A visão do projeto é transformar o DaviOS em um assistente principal capaz de:

- conversar naturalmente;
- lembrar informações importantes sobre o usuário;
- compreender contexto;
- raciocinar sobre informações armazenadas;
- utilizar um modelo de IA local;
- funcionar sem assinatura;
- funcionar sem depender de Wi-Fi durante a execução;
- auxiliar no desenvolvimento de projetos;
- futuramente utilizar ferramentas locais de forma controlada.

O projeto está sendo construído em camadas para que o núcleo continue funcionando mesmo quando componentes avançados, como um modelo local, não estejam disponíveis.

---

## 🎯 Objetivo

A arquitetura futura do DaviOS segue aproximadamente este fluxo:

```text
                    ┌──────────────┐
                    │    Usuário   │
                    └──────┬───────┘
                           │
                           ▼
                ┌────────────────────┐
                │ ConversationEngine │
                └─────────┬──────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  CognitiveCore  │
                 └───────┬─────────┘
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
         Memória      Contexto    Personalidade
             │           │           │
             └───────────┼───────────┘
                         ▼
                  ┌──────────────┐
                  │ LLM Provider │
                  └──────┬───────┘
                         │
                         ▼
                 Modelo local/offline
```

A longo prazo:

```text
DaviOS
 ├── Conversação
 ├── Memória
 ├── Raciocínio
 ├── Personalidade
 ├── Modelo local
 ├── Projetos
 ├── Arquivos
 ├── Git
 ├── Terminal controlado
 ├── Testes
 └── Ferramentas autorizadas
```

---

## 🏗️ Arquitetura atual

O projeto é dividido em módulos com responsabilidades específicas.

```text
DaviOS/
│
├── brain/
│   ├── conversation_engine.py
│   ├── cognitive_core.py
│   ├── intent_classifier.py
│   ├── llm_provider.py
│   ├── model_manager.py
│   ├── prompt_builder.py
│   ├── response_generator.py
│   └── providers/
│       └── local_llama_cpp_provider.py
│
├── core/
│   ├── hardware_detector.py
│   └── ...
│
├── memory/
│   ├── database.py
│   ├── memory_interpreter.py
│   └── ...
│
├── personality/
│   └── personality.py
│
├── config/
│   ├── davios_config.py
│   ├── davios.json
│   └── model_profiles.json
│
├── models/
│   ├── light/
│   ├── balanced/
│   ├── performance/
│   └── README.md
│
├── tests/
│
├── docs/
│   └── ARCHITECTURE.md
│
├── logs/
├── assets/
├── main.py
├── requirements.txt
└── README.md
```

### Responsabilidades principais

| Módulo | Responsabilidade |
|---|---|
| `brain/` | Inteligência, conversação, contexto e integração com modelos |
| `memory/` | Memória persistente e fatos do usuário |
| `core/` | Serviços fundamentais e detecção do ambiente |
| `personality/` | Personalidade e comportamento do DaviOS |
| `config/` | Configurações e perfis de execução |
| `models/` | Modelos locais, separados por perfil de hardware |
| `tests/` | Testes automatizados |
| `docs/` | Documentação técnica |

---

## 🧠 Sistema de memória

A memória do DaviOS utiliza SQLite e trabalha com fatos estruturados.

Um fato pode conter informações como:

```text
target
relation
emotion
emotional_intensity
temporal_context
negation
subject
value
confidence
importance
status
source
fact_type
created_at
updated_at
last_accessed_at
access_count
evidence
metadata
```

Exemplo:

```text
Usuário:
"eu não gosto mais de pizza"

DaviOS:
target = pizza
relation = dislike
negation = true
temporal_context = current
status = active
```

O sistema também possui tratamento para:

- repetição de informações;
- reforço de memórias existentes;
- contradições;
- atualização de fatos;
- consolidação de memórias antigas;
- identidade semântica;
- histórico;
- evidências;
- prevenção de duplicação de fatos ativos.

A ideia é que o DaviOS não simplesmente acumule frases em um banco de dados. Ele deve construir uma memória estruturada e atualizável.

---

## 💬 Conversação

O `ConversationEngine` funciona como a porta de entrada da conversa.

Ele diferencia situações como:

```text
GREETING
FAREWELL
COMMAND
MEMORY_QUERY
MEMORY_STATEMENT
GENERAL_QUESTION
OPINION
CONVERSATION
```

O fluxo pode direcionar cada tipo de interação para o componente adequado.

Por exemplo:

```text
"Meu nome é Davi"
        ↓
MemoryInterpreter
        ↓
MemoryManager
        ↓
Banco de memória
```

Enquanto:

```text
"O que é Python?"
        ↓
ConversationEngine
        ↓
CognitiveCore
        ↓
LLMProvider
        ↓
Modelo local
```

---

## 🤖 Modelo de IA local

O DaviOS foi projetado para não depender de um único backend.

A interface:

```text
LLMProvider
```

permite trocar o mecanismo de inferência sem reescrever o restante do sistema.

A estratégia atual para execução local utiliza:

```text
DaviOS
   ↓
LocalLlamaCppProvider
   ↓
llama-server.exe
   ↓
HTTP 127.0.0.1
   ↓
modelo GGUF
```

A comunicação com `127.0.0.1` ocorre localmente na própria máquina e não representa uma dependência de Internet.

### Fallback

Quando um modelo local não estiver disponível, o DaviOS deve continuar funcionando através do mecanismo de regras existente.

A ausência do modelo não deve inutilizar o sistema inteiro.

---

## 🖥️ Hardware

O projeto foi pensado para ser portátil entre máquinas diferentes.

O hardware é detectado dinamicamente e o sistema possui perfis de modelo:

```text
models/
├── light/
├── balanced/
└── performance/
```

A máquina principal de desenvolvimento possui atualmente:

```text
CPU: Ryzen 5 5600
RAM: 16 GB
GPU: Radeon RX 6600 8 GB
Sistema: Windows 11 64-bit
```

Essas características servem como referência de desenvolvimento, mas não devem ser tratadas como requisitos fixos do projeto.

---

## 🔌 Offline First

Uma das metas centrais do DaviOS é:

> **Depois de instalado e configurado, o assistente deve conseguir funcionar sem Wi-Fi e sem assinatura.**

Isso significa separar:

### Instalação

Pode exigir Internet para:

- baixar dependências;
- baixar o backend;
- baixar modelos;
- atualizar componentes.

### Execução

Deve poder funcionar localmente:

```text
Internet OFF
     ↓
DaviOS
     ↓
Memória local
     ↓
Modelo local
     ↓
Resposta
```

Nenhum componente de conversação deve depender obrigatoriamente de uma API externa.

---

## 🧪 Testes

O projeto possui uma suíte automatizada para preservar o comportamento existente durante a evolução.

Execute:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Antes de fazer alterações estruturais importantes, os testes devem ser executados.

Novas funcionalidades devem preferencialmente receber testes próprios.

### Princípio

> Evoluir o DaviOS sem quebrar o que já funciona.

---

## 🚀 Como executar

Ative o ambiente virtual:

```powershell
.venv\Scripts\activate
```

Execute:

```powershell
python main.py
```

Caso o modelo local ainda não esteja instalado, o DaviOS deve continuar utilizando o modo de fallback disponível.

---

## 🧰 Desenvolvimento

Dependências Python são mantidas em:

```text
requirements.txt
```

O ambiente virtual recomendado é:

```text
.venv/
```

O ambiente virtual não deve ser versionado no Git.

Modelos GGUF também não devem ser enviados para o repositório, devido ao tamanho dos arquivos.

---

## 🔐 Princípios de segurança

O DaviOS futuramente poderá utilizar ferramentas locais, mas acesso ao sistema operacional deve ser implementado de forma controlada.

A arquitetura deve evitar:

- execução arbitrária de comandos sem autorização;
- alterações destrutivas sem confirmação;
- exclusão de arquivos sem controle;
- acesso desnecessário a dados pessoais;
- execução silenciosa de ações potencialmente perigosas.

A evolução planejada é utilizar ferramentas explícitas e permissões específicas.

Exemplo:

```text
DaviOS
   ↓
Tool Registry
   ↓
Ferramenta autorizada
   ↓
Ação
```

Em vez de:

```text
IA
 ↓
terminal irrestrito
```

---

## 🗺️ Roadmap

### Fase 1 • Núcleo
- [x] Estrutura modular
- [x] Banco SQLite
- [x] Sistema de memória
- [x] Consolidação de memórias
- [x] ConversationEngine
- [x] Intent classification
- [x] CognitiveCore
- [x] PromptBuilder
- [x] Abstração `LLMProvider`
- [x] ModelManager
- [x] Detecção de hardware
- [x] Fallback sem modelo

### Fase 2 • Cérebro local
- [ ] Finalizar `LocalLlamaCppProvider`
- [ ] Instalar backend standalone llama.cpp
- [ ] Instalar primeiro modelo GGUF
- [ ] Integrar inferência local
- [ ] Testar CPU
- [ ] Testar aceleração GPU quando disponível
- [ ] Benchmark de velocidade
- [ ] Ajustar contexto e parâmetros

### Fase 3 • Memória avançada
- [ ] Embeddings
- [ ] Busca semântica
- [ ] RAG
- [ ] Memória episódica
- [ ] Memória de longo prazo
- [ ] Resumo automático de conversas
- [ ] Gerenciamento de contexto

### Fase 4 • Agente
- [ ] Tool Registry
- [ ] Leitura controlada de arquivos
- [ ] Acesso a projetos
- [ ] Git
- [ ] Execução controlada de testes
- [ ] Terminal com permissões
- [ ] Planejamento de tarefas
- [ ] Verificação de resultados

### Fase 5 • Assistente principal
- [ ] Sessões persistentes
- [ ] Contexto entre sessões
- [ ] Personalidade adaptativa
- [ ] Interface mais completa
- [ ] Voz
- [ ] Automação local
- [ ] Monitoramento e logs
- [ ] Sistema de permissões
- [ ] Recuperação de erros

---

## 📁 Filosofia de desenvolvimento

O DaviOS deve crescer como um sistema modular.

### Evitar

```text
Uma IA gigante fazendo tudo.
```

### Preferir

```text
┌──────────────────────┐
│ ConversationEngine   │
├──────────────────────┤
│ CognitiveCore        │
├──────────────────────┤
│ Memory               │
├──────────────────────┤
│ Personality          │
├──────────────────────┤
│ LLMProvider          │
├──────────────────────┤
│ Tools                │
└──────────────────────┘
```

Cada camada possui uma responsabilidade clara.

Isso facilita testes, manutenção, troca de modelos e evolução do projeto.

---

## 📜 Status atual

O DaviOS já possui uma base funcional de:

- memória persistente;
- interpretação de fatos;
- atualização de memórias;
- conversação determinística;
- contexto;
- personalidade;
- abstração de modelos;
- detecção de hardware;
- arquitetura para LLM local;
- fallback sem modelo;
- suíte de testes automatizados.

O próximo grande marco é conectar o cérebro generativo local ao sistema através do `LocalLlamaCppProvider`.

---

## 👨‍💻 Desenvolvimento

Projeto pessoal desenvolvido por **Davi** com foco em aprendizado, experimentação e construção de um assistente de IA local.

O objetivo não é apenas criar um chatbot.

O objetivo é construir, passo a passo, um **assistente pessoal de IA que realmente consiga trabalhar junto com seu criador**.

---

## ⭐ Visão

```text
Hoje:

Você → DaviOS → memória + regras

↓

Próximo marco:

Você → DaviOS → memória + contexto + modelo local

↓

Futuro:

Você → DaviOS → raciocínio + memória + ferramentas + projetos

↓

Objetivo:

Seu assistente principal.
Local.
Privado.
Sem assinatura.
Independente de Wi-Fi.
Evoluindo junto com você.
```
