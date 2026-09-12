"""Sonda temporaria de diagnostico (FORA do projeto, apagavel).

Replica EXATAMENTE _closing_sentence() e o ratio de _repeated_closing()
de brain/cognitive_core.py para medir os scores reais dos pares
observados na sessao com repeticao.
"""
import difflib
import re

# --- copia fiel de CognitiveCore._closing_sentence ---
def closing_sentence(text: str, max_chars: int = 45) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip()).strip()
    if not normalized:
        return ""
    sentences = re.split(r"(?<=[.!?…])\s+", normalized)
    closing = sentences[-1].strip()
    if len(closing) > max_chars:
        closing = closing[-max_chars:]
        idx = closing.find(" ")
        if 0 < idx <= 12:
            closing = closing[idx + 1:]
    return closing.casefold().strip()


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


# Respostas observadas na sessao real (transcritas do relato):
responses = [
    "Como posso te ajudar? 😊",                                # "salve"
    "Ah, entendo! Parece que já está com a manhã em marcha. Como posso te ajudar hoje? 😊",
    "São 10h30, então sim, está ainda no início da manhã. Como posso te ajudar agora?",
    "Ah, entendo! Vamos continuar a conversa então. Como posso te ayudar? 😊",
    "kkk, já vi. Como posso ajudar você hoje?",
    "Ah, entendo! Vou deixá-lo em paz e começar a falar um pouco mais sobre mim mesmo 😊 Como posso te ajudar today? 🤗",
    "Boa! Como posso te ajudar hoje? 😊",                      # x4 ("boa")
    "Boa! Como posso te ajudar hoje? 😊",
    "Boa! Como posso te ajudar hoje? 😊",
    "Boa! Como posso te ajudar hoje? 😊",
    "Ah, entendi! Vou me ajustar pra não repetir tanto. Como posso te ajudar hoje? 😊",
]

print("=== CLOSING extraido de cada resposta ===")
closings = []
for r in responses:
    c = closing_sentence(r)
    closings.append(c)
    print(f"  [{len(c):3d} chars] {c!r}")

print()
print("=== RATIO de cada closing vs 'como posso te ajudar hoje? 😊' (limiar camada 2 = 0.88) ===")
ref = closings[6]
for r, c in zip(responses, closings):
    print(f"  {ratio(c, ref):.4f}  <- {c!r}")

print()
print("=== Pares especificos pedidos ===")
pairs = [
    ("Como posso te ajudar hoje? 😊", "Como posso te ajudar? 😊"),
    ("Como posso te ajudar hoje? 😊", "Como posso ajudar você hoje?"),
    ("Como posso te ajudar hoje? 😊", "Como posso te ajudar today? 🤗"),
    ("Como posso te ajudar hoje? 😊", "Em que posso ajudar?"),
    ("Como posso te ajudar hoje? 😊", "O que posso fazer por você?"),
    ("Como posso te ajudar hoje? 😊", "Como posso contribuir?"),
    ("Como posso te ajudar hoje? 😊", "How can I help?"),
    ("Como posso te ajudar hoje?", "Como posso te ajudar?"),
    ("Como posso te ajudar hoje?", "Como posso te auxiliar?"),
]
for a, b in pairs:
    ca, cb = closing_sentence(a), closing_sentence(b)
    print(f"  {ratio(ca, cb):.4f}  {ca!r}  vs  {cb!r}")

print()
print("=== Camada 1 (resposta INTEIRA, limiar >= 0.92, min 40 chars) ===")
for r in ["Boa! Como posso te ajudar hoje? 😊",
          "Como posso te ajudar? 😊",
          "São 10h30. Como posso te ajudar?"]:
    n = re.sub(r"\s+", " ", r.strip().casefold())
    print(f"  len={len(n):3d}  {'IGNORADA (<40)' if len(n) < 40 else 'comparada'}  {r!r}")
