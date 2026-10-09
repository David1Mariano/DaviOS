# Investigacao P3 — Trava na suíte completa de testes

**Status:** encerrada
**Branch:** `main` (local `HEAD` 2 commits à frente de `origin/main`)
**Pytest:** 9.1.1
**Data de elaboração:** 2026-06-10

> Documenta: contexto do problema; alterações P1/P2 relacionadas; comandos de
> teste e resultados confirmados; distinção entre fatos e hipóteses; limitações
> do diagnóstico; estado atual da investigação; recomendação para revisão e
> eventual commit. Apenas leitura: nenhuma correção foi implementada nesta
> missão e nenhum teste da suíte completa foi reexecutado.

## 1. Contexto

Antes das correções de **P2**, a suíte completa travaram perto de
`tests/test_partb_agents.py::TestAgentServer::test_start_and_stop`, com
interrupção do console (`KeyboardInterrupt`) acoplada a testes que criavam locks
de download (método antigo `os.kill(pid, 0)`, que envia `CTRL_C_EVENT` a um
console compartilhado). Este relatório registra o resultado da investigação
controlada de P3, que seguiu rigorosamente as restrições: nenhum comando de
stage/commit/push, nenhum sinal a processos externos, apenas testes direcionados,
uma única reprodução instrumentada da suíte completa e revisão somente-leitura
de P1/P2.

## 2. Alterações P1/P2 relacionadas

| Arquivo | Responsabilidade | Efeito resumido |
|---|---|---|
| `utils/model_downloader.py` | P2 | Troca o probe `os.kill(pid, 0)` no Windows por `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` via `WinDLL(kernel32, use_last_error=True)` + `ctypes.get_last_error()`; `ERROR_ACCESS_DENIED` ⇒ vivo; `ERROR_INVALID_PARAMETER` (87) ⇒ morto; qualquer outro código/exceção ⇒ vivo (conservador). Tipos `argtypes`/`restype` explícitos para 32/64 bits. |
| `tests/test_offline_backend.py` | P2 | 4 novos testes direcionados: detecção de vida (`_process_is_alive`), `ERROR_ACCESS_DENIED` ⇒ vivo, códigos Win32 mockados, lock com conteúdo inválido, semântica `acquire`/`release`. |
| `tests/test_context_representation.py` | P1 (não alterado nesta missão) | 32 testes de representação de contexto/corrupção de importação; inalterado por P2. |

> Nota de preservação: nenhum dos arquivos acima foi modificado nesta missão.
> O diff já havia sido aplicado em P2 (sem conflito; nenhum `git stash`,
> `checkout`, `reset` ou `restore` foi executado).

## 3. Comandos executados

### 3.1 Verificação inicial (antes de qualquer teste)

```powershell
cd c:\Users\Davi\DaviOS
git rev-parse --abbrev-ref HEAD        # main
git rev-parse HEAD                       # 79635af
git rev-parse origin/main                # ed487f9
git rev-list --left-right --count HEAD...origin/main
# resultado: 2 0  (2 commits à frente)
git --no-pager diff --cached --stat      # vazio: zero staged
git --no-pager diff --stat               # 7 arquivos modificados
git status --short                       # estado (§8)
```

### 3.2 Passo 1 — testes direcionados, processos separados

```powershell
# (a) downloader
python -m pytest tests\test_offline_backend.py tests\test_model_downloader.py -q -p no:cacheprovider
# resultado: 150 passed, 1 xfailed em 5.20s

# (b) agentes (processo próprio, finito)
python -m pytest tests\test_partb_agents.py -q --full-trace -p no:cacheprovider
# resultado: 16 passed em 0.13s
```

### 3.3 Passo 2 — reprodução instrumentada da suíte completa (única execução)

```powershell
python -m pytest --full-trace -o faulthandler_timeout=60 `
    -o faulthandler_exit_on_timeout=true -p no:cacheprovider -q
```

> `faulthandler-timeout` CLI foi removido no pytest 9.1.1; equivalente aplicado
> via `-o` e sem edição de `pytest.ini`. O corte de 280s foi atingido com
> **sucesso** (`TIMEOUT_280S` não disparou).

## 4. Resultados confirmados

- **Última etapa alcançada:** suíte completa concluída com normalidade —
  `856 passed, 1 xfailed in 23.78s`.
- **Coletados:** 857 testes (inclui `TestAgentServer::test_start_and_stop`).
- **Sem interceptações:** nenhum `TIMEOUT_280S`, nenhum `KeyboardInterrupt`,
  nenhuma stack do faulthandler (nenhum teste excedeu 60s), nenhum sinal de
  interrupção detectável.
- **Órfãos:** não restaram processos pytest aguardando.
- **Regressão preservada:** `test_offline_backend.py` + `test_context_representation.py` = 32 passed.

## 5. Fatos vs. hipóteses

### Fatos confirmados

- A suíte instrumentada terminou normalmente após P2.
- O método anterior de verificação de processos (`os.kill(pid, 0)` no Windows)
  foi investigado e substituído na P2 por `OpenProcess` + `get_last_error()`.
- Os testes direcionados informados passaram.
- A execução P3 não deixou processos pytest pendurados.
- A hipótese de deadlock (`Thread.start() → _started.wait()`) não foi observada
  na execução instrumentada.

### Hipótese ainda não comprovada

- O relato anterior de travamento em `test_start_and_stop` **pode** estar
  relacionado ao evento de interrupção ocorrido anteriormente, mas **não há
  evidência suficiente para provar a causalidade**.
- Não se reactivou o comportamento antigo para reproduzir (proibido pela
  revisão: interrompesse o console compartilhado e não era necessário).

## 6. Limitações do diagnóstico

1. Uma única reprodução instrumentada foi executada; variáveis ambiente, carga
   de CPU e estado do disco não foram variados.
2. O componente `agent/segundo-agente` (worktree de voz do Agente 2) ficou
   excluído de toda a execução e das edições; nenhum arquivo dele foi acessado.
3. As suítes de Qdrant/llama-server/highlight não são cobertas por esta
   execução — o que não exclui defeitos de exercício desses backends.
4. O relatório de "hang" histórico não está arquivado no repositório como
   log; ele só é reproduzido como referência textual no `git log`.
5. O `git diff` inicial ficou truncado por pager do terminal; o estado foi
   consolidado via `git --no-pager diff` e `git status --short`.

## 7. Estado atual da investigação

P3 concluída: sem travamento, sem falha e sem timeout na suíte completa,
incluindo `test_start_and_stop`. A causa confirmada como **fato** é que a
execução limpa após P2 entrega `856 passed, 1 xfailed` em 23,78s, sem
interceptações. A hipótese de que o relato anterior estava ligado ao
`KeyboardInterrupt` anterior permanece como **hipótese** (consistente, mas sem
prova).

## 8. Recomendações para revisão e eventual commit

- Submeter P2 (`utils/model_downloader.py`, `tests/test_offline_backend.py`) e
  P1 (`tests/test_context_representation.py`) em commits próprios, sem
  sobrepor-se a nenhum outro trabalho existente.
- O `docs/` não recebeu alterações nesta missão; o novo arquivo
  `docs/P3_TEST_INVESTIGATION.md` é proposto como local de registro histórico
  (criação autorizada nesta revisão).
- Revisar previamente a divergência local vs `origin/main` (2 commits à frente)
  e decidir, antes de qualquer stage, se a branch `main` deve ser rebaseada.
- Não iniciar suíte completa a menos que seja solicitada ou autorizada.
- Manter a política P2 de erro conservador (apenas
  `ERROR_INVALID_PARAMETER` → morto); `ERROR_ACCESS_DENIED` e erros inesperados
  ⇒ vivo.


