## Instruções de Teste - DaviOS Memory Deduplication Fix

### Pré-requisitos
```powershell
# Instalar pytest (se não estiver instalado)
python -m pip install pytest
```

---

## Opção 1: Teste Rápido (Recomendado)
Executa o teste final com a sequência completa:

```powershell
cd c:\Users\Davi\DaviOS
python test_final_sequence.py
```

**Resultado esperado:**
- ✅ Memory count: 1
- ✅ Relation: like (estado final)
- ✅ Histórico: like -> dislike -> like
- ✅ Pizza permanece em 1 memória durante todo o fluxo

---

## Opção 2: Testes Unitários Completos
Executa os 6 testes obrigatórios com pytest:

```powershell
cd c:\Users\Davi\DaviOS

# Executar todos os 6 testes
python -m pytest tests/test_memory_duplication_fix.py::TestMemoryDeduplication -v

# Ou executar testes específicos:

# Teste 1: Primeira declaração
python -m pytest tests/test_memory_duplication_fix.py::TestMemoryDeduplication::test_01_primeira_declaracao -v

# Teste 2: Mesma informação (reinforce)
python -m pytest tests/test_memory_duplication_fix.py::TestMemoryDeduplication::test_02_mesma_informacao_novamente -v

# Teste 3: Mudança de preferência (update)
python -m pytest tests/test_memory_duplication_fix.py::TestMemoryDeduplication::test_03_mudanca_de_preferencia -v

# Teste 4: Mudança novamente
python -m pytest tests/test_memory_duplication_fix.py::TestMemoryDeduplication::test_04_mudanca_novamente -v

# Teste 5: Frase incerta
python -m pytest tests/test_memory_duplication_fix.py::TestMemoryDeduplication::test_05_frase_incerta -v

# Teste 6: Frases semanticamente iguais
python -m pytest tests/test_memory_duplication_fix.py::TestMemoryDeduplication::test_06_frases_semanticamente_iguais -v
```

**Resultado esperado:** 6 PASSED ✅

---

## Opção 3: Teste com o Sistema Completo
Se deseja testar com o `main.py` integrado (mais elaborado):

### Passo 1: Primeira declaração
```powershell
cd c:\Users\Davi\DaviOS
# Edite main.py e coloque:
# user_input = "eu gosto de pizza"
python main.py
```
Esperado: Memory ID = 1, relation = like

### Passo 2: Mudança de preferência
```powershell
# Edite main.py e coloque:
# user_input = "eu não gosto mais de pizza"
python main.py
```
Esperado: Memory ID = 1 (MESMA!), relation = dislike

### Passo 3: Mudança novamente
```powershell
# Edite main.py e coloque:
# user_input = "agora eu gosto de pizza"
python main.py
```
Esperado: Memory ID = 1 (MESMA!), relation = like

---

## Validação Visual

O sucesso é confirmado quando você vê nos logs:

### Teste 1:
```
[OPERATION] action=create facts=1
Memory count: 1
Memory ID: 1
```

### Teste 2:
```
[OPERATION] action=reinforce memory_id=1 fact_id=1
[DATABASE] action=reinforce memory_id=1 fact_id=1
Memory count: 1
Memory ID: 1  ← MESMA!
```

### Teste 3:
```
[OPERATION] action=update memory_id=1 facts=1
[DATABASE] action=update memory_id=1 fact_id=1
Memory count: 1
Memory ID: 1  ← MESMA!
```

---

## Limpeza de Bases de Dados de Teste

Se precisar remover as bases criadas durante os testes:

```powershell
cd c:\Users\Davi\DaviOS

# Remover banco do teste final
Remove-Item -Force test_final.db -ErrorAction SilentlyContinue

# Remover banco principal se deseja recomeçar
Remove-Item -Force memory.db -ErrorAction SilentlyContinue
Remove-Item -Force davios.db -ErrorAction SilentlyContinue
```

---

## Verificação de Histórico

Para verificar que o histórico foi preservado corretamente:

```powershell
python -c "
from memory.memory_manager import MemoryManager
m = MemoryManager()
mems = m.database.find_memories()
if mems:
    facts = mems[0].facts
    if facts:
        history = m.database.get_fact_history(facts[0].id)
        print('Histórico:')
        for rev in history:
            print(f\"  {rev.get('relation')} - {rev.get('revision_type')}\")
"
```

---

## Logs Detalhados

Para ver logs mais detalhados durante execução:

```powershell
# No arquivo test_final_sequence.py, descomente/adicione:
import logging
logging.basicConfig(level=logging.DEBUG)

python test_final_sequence.py
```

---

## Checklist de Validação ✅

- [ ] Teste 1 passou (create memória nova)
- [ ] Teste 2 passou (reinforce mesma memória)
- [ ] Teste 3 passou (update relação like→dislike)
- [ ] Teste 4 passou (update relação dislike→like)
- [ ] Teste 5 passou (update com confidence reduzida)
- [ ] Teste 6 passou (frases variadas = mesma memória)
- [ ] Histórico preserva todas as mudanças
- [ ] Memory count = 1 após todos os testes
- [ ] Fact ID permanece = 1 após todos os testes
- [ ] Sem erros de lógica de negócio

---

## Troubleshooting

### Erro: "PermissionError: O arquivo já está sendo usado"
**Causa:** Windows mantém arquivo SQLite aberto
**Solução:** Aguarde alguns segundos ou reinicie PowerShell

### Erro: "pytest: command not found"
**Causa:** pytest não instalado ou não no PATH
**Solução:** 
```powershell
python -m pip install pytest --upgrade
```

### Erro: "No module named 'memory'"
**Causa:** Diretório errado
**Solução:** Certifique-se que está em `c:\Users\Davi\DaviOS`
```powershell
cd c:\Users\Davi\DaviOS
```

---

## Próximos Passos

Após validar que a correção funciona:

1. Integrar testes ao CI/CD
2. Executar testes de regressão no código existente
3. Evoluir o sistema para interação com recursos do computador (com permissões)
4. Refinar análise de contexto e raciocínio

---

**Status**: ✅ Pronto para teste em PowerShell
**Última atualização**: 2026-08-22
