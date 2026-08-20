from reasoning.reasoning_engine import ReasoningEngine


engine = ReasoningEngine()


tests = [
    "pesquisar notícias sobre tecnologia",
    "eu gosto de pizza",
    "eu odeio acordar cedo",
    "abra o terminal",
    "hoje eu tive um dia estranho",
]


for text in tests:

    result = engine.evaluate_input(text)

    print("\nINPUT:", text)
    print("INTENT:", result.intent)
    print("ACTION:", result.action)
    print("CONFIDENCE:", result.confidence)
    print("TARGET:", result.target_module)
    print("REASONING:", result.reasoning_log)