import ast
files=['brain/intent_classifier.py','brain/prompt_builder.py','brain/conversation_engine.py','brain/cognitive_core.py','brain/tool_request_parser.py','brain/tool_execution_flow.py','memory/memory_interpreter.py','personality/personality.py','tests/test_intent_classifier.py','tests/test_personality.py']
for p in files:
    try:
        ast.parse(open(p,encoding='utf-8').read()); print('OK  ', p)
    except SyntaxError as e:
        print('FAIL', p, e.lineno, e.msg)
