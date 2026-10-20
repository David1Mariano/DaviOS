# Roteador híbrido de provedores (OpenAI-compatible)

O DaviOS pode usar um provedor de LLM remoto (qualquer endpoint compatível
com a API OpenAI `/v1/chat/completions`) como camada opcional "online first",
com fallback garantido ao modelo local. Nenhum provedor é hard-coded e nenhum
segredo é gravado no código.

## Arquitetura

```
ConversationEngine -> CognitiveCore -> HybridLLMProvider (implementa LLMProvider)
                                  |- router:  OnlineRouter  (online, via urllib)
                                  `-- local:   LocalLLMProvider (Qwen local, llama-cpp)

Regras:
1. Nenhum provedor é hard-coded: endpoints/credenciais vêm de config/davios.json
   (campo remote_providers) OU de variáveis de ambiente DAVIOS_PROVIDER_N_*.
2. offline_mode=true OU network_policy="offline" → rede bloqueada, só local.
3. network_policy="auto" (padrão do router) → tenta o remoto primeiro; ao falhar,
   cai no local. O local é a garantia de resposta.
4. Cada chamada respeita timeout_seconds (default 8s, rigoroso) para que o loop
   sincrono nunca fique bloqueado.
5. OnlineRouter NÃO herda LLMProvider e não toca no modelo local. HybridLLMProvider
   é o único nó que conhece as duas, mantendo CognitiveCore/ConversationEngine neutros.
6. Transient retry: connection / timeout / HTTP 429,5xx → RouterTransientError
   (gerencia retry_count + backoff e roda próximo provedor).
   HTTP 404 / bad-request → RouterPermanentError (não retry, decide fallback).
```

## Configuração (3 formas, nenhuma contém segredo no código)

### 1. Variáveis de ambiente (recomendado — nunca commita segredos)

```powershell
# 1) Desbloqueia a rede (offline-first por definicao): ONLINE exige offline_mode=0
$env:DAVIOS_OFFLINE_MODE    = "0"                  # 1=offline-only (default), 0=online permitido
$env:DAVIOS_NETWORK_POLICY  = "auto"               # auto (online-first + fallback) | online | offline

# 2) Provedor 0 (obrigatorio para online)
$env:DAVIOS_PROVIDER_0_BASE_URL = "https://api.openai.com"
$env:DAVIOS_PROVIDER_0_API_KEY  = "sk-..."          # nunca commitar no repo
$env:DAVIOS_PROVIDER_0_MODEL    = "gpt-4o-mini"

# 3) Opcional: segundo provedor (balanceio/roteamento)
$env:DAVIOS_PROVIDER_1_BASE_URL = "https://openrouter.ai/api"
$env:DAVIOS_PROVIDER_1_API_KEY  = "sk-..."
$env:DAVIOS_PROVIDER_1_MODEL    = "openai/gpt-4o-mini"
$env:DAVIOS_PROVIDER_1_ENABLED  = "1"

# 4) Política do router
$env:DAVIOS_ROUTER_ENABLED     = "1"
# DAVIOS_ROUTER_MODE          -> reservado (não altera o comportamento)
# $env:DAVIOS_ROUTER_MODE = "auto"
$env:DAVIOS_ROUTER_TIMEOUT    = "60"      # timeout global usado como default por provedor
$env:DAVIOS_ROUTER_RETRY_COUNT = "2"
# (o backoff entre tentativas é interno/exponencial ~ 0.5s * 2^n, teto 8s;
#  não é configurável via DAVIOS_ROUTER_BACKOFF_SECONDS — o campo existe porém
#  não é lido pelo código atualmente.)
```

> **Como ler a política:** a decisão de "online first + fallback local" é feita por
> `DAVIOS_OFFLINE_MODE` + `DAVIOS_NETWORK_POLICY` (que o `build_engine` e o
> `HybridLLMProvider` leem de fato). O campo JSON `router.mode`
> (`DAVIOS_ROUTER_MODE`) é **reservado** — não altera o comportamento atual. O
> modo implementado é sempre `auto` (tenta o remoto; ao falhar, cai no local).

### 2. Arquivo `config/davios.json`

Edite `remote_providers` e `router`. Recomenda-se deixar `remote_providers: []`
no repositório e apontar a `api_key` via env, para que nenhum segredo fique
versionado. Exemplo apenas para referência (não commitar a key):

```json
{
  "remote_providers": [
    {
      "name": "openai",
      "base_url": "https://api.openai.com",
      "api_key": "",
      "model": "gpt-4o-mini",
      "timeout_seconds": 8.0,
      "enabled": true
    }
  ],
  "router": { "enabled": true, "mode": "auto", "timeout_seconds": 8.0, "retry_count": 2 }
}
```

### 3. Sem configuração

Sem `base_url` nem `DAVIOS_PROVIDER_N_*`, `get_enabled_providers()` retorna `[]` e
`OnlineRouter.generate()` levanta `LLMUnavailableError` com mensagem clara. O
`HybridLLMProvider` pega essa exceção e usa o **local** — portanto, com credenciais
ausentes o DaviOS continua funcionando em OFFLINE sem erros desnecessários.

## Como ativar (passo a passo)

1. Desbloqueie a rede: `offline_mode: false` e `network_policy: "auto"` em
   `config/davios.json` **ou**, preferencialmente, via env (a env tem prioridade
   sobre o JSON):
   ```powershell
   $env:DAVIOS_OFFLINE_MODE = "0"
   $env:DAVIOS_NETWORK_POLICY = "auto"
   ```
   Sem isso (default: `offline_mode: true` / `network_policy: "offline"`), o
   `build_engine` nem instancia o router — o assistente roda 100% offline/local.
2. Configure as env `DAVIOS_PROVIDER_0_BASE_URL` / `..._API_KEY` / `..._MODEL`.
3. Opcional: garanta que um modelo local GGUF esteja disponível (fallback).
4. `python main.py` — o log mostrará `[ROUTER] modo auto: online primeiro, fallback local ativo.`

## Privacidade de memória

O conteúdo enviado ao provedor remoto é **exatamente** o que o `PromptBuilder`
produzir para `LLMRequest.system` + `LLMRequest.prompt`. O `OnlineRouter`
reencapsula isso em `messages = [{"role": "system", ...}, {"role": "user", ...}]`
e o POSTa para `/v1/chat/completions` (payload construído em
`OnlineRouter._build_payload`).

Recorte ativo (configurado em `config/davios.json`):
- `max_recent_messages` (histórico conversacional, default 6)
- `max_memories_in_prompt` (memórias relevantes, default 5)

> **Aviso:** tudo o que estiver em `request.system` / `request.prompt` **é
> transmitido verbatim** ao provedor online — incluindo trechos de memórias
> recortadas e trechos do histórico. Portanto trate o provedor online como
> externo: não coloque dados sensíveis no `system_prompt` se não quiser que
> cheguem ao remoto. Memórias pessoais completas **não** são enviadas por
> padrão, apenas o contexto recortado acima.

---

## Testes: simulados × reais

### Simulados (sem rede externa, sem secrets) — suíte padrão

`tests/test_online_router.py` não toca a internet: o caminho "remoto" é um
**servidor HTTP fictício em thread local** (`FakeOpenAIServer`) que imita
`/v1/chat/completions`. Cobertura:

| Classe | Testes | Cobertura |
|---|---|---|
| `OnlineRouterOfflineTest` | 3 | gating: offline_mode / policy=offline / auto |
| `OnlineRouterRemoteFailureTest` | 1 | falha em todos os remotes → `LLMUnavailableError` (fallback local é feito pelo Hybrid) |
| `OnlineRouterRemoteSuccessTest` | 2 | happy path: parse de resposta OpenAI via HTTP fake |
| `HybridOfflineTest` | 3 | offline sempre usa local |
| `HybridAutoTest` | 3 | auto: remote quebrado / sem remotes → fallback local |
| `HybridUnloadTest` | 2 | unload delega ao provider local |

```bash
python -m pytest tests/test_online_router.py -q
# 14 passed
```

### Reais (com API de verdade — opcional, exige credenciais)

Não há teste automatizado contra API real no repositório (não podemos commitar
uma key). Para validar manualmente contra um provedor real:

```bash
# 1. Desbloqueia a rede (offline-first por definicao) e configura o provedor 0
export DAVIOS_OFFLINE_MODE=0
export DAVIOS_NETWORK_POLICY=auto
export DAVIOS_ROUTER_ENABLED=1
export DAVIOS_PROVIDER_0_BASE_URL=https://api.openai.com
export DAVIOS_PROVIDER_0_API_KEY=sk-xxx      # sua key (nunca commite)
export DAVIOS_PROVIDER_0_MODEL=gpt-4o-mini
export DAVIOS_ROUTER_TIMEOUT=8

# 2. Smoke test contra a API real (online-first + fallback local).
#    Script pronto em tmp/_smoke_online_real.py (sem segredos no codigo):
python tmp/_smoke_online_real.py

# 3. Suíte completa (usa só HTTP fake + local, sem tocar a internet)
python -m pytest tests/ -q
```

Critérios de aceitação num teste real:
- `response.provider == "online"` → a resposta veio do remoto.
- Com `DAVIOS_OFFLINE_MODE=1` ou `DAVIOS_NETWORK_POLICY=offline` →
  `network_allowed == False` e a rota online é **pulada** (nenhuma chamada de
  rede); a resposta vem do local/rules.
- Key inválida (HTTP 401) vira `RouterPermanentError`; o `HybridLLMProvider` pega
  a exceção e cai no **local** — sem travar o assistente.
- Timeout / conexão recusada vira `RouterTransientError` (retry); ao esgotar as
  tentativas vira `LLMUnavailableError` → fallback local.