import ast, pathlib
files = ['brain/intent_classifier.py','brain/prompt_builder.py','brain/conversation_engine.py','brain/cognitive_core.py','personality/personality.py','tests/test_intent_classifier.py','tests/test_personality.py','config/davios_config.py']
for p in files:
    src = pathlib.Path(p).read_text(encoding='utf-8-sig')
    try:
        ast.parse(src)
        print('OK  ', p)
    except SyntaxError as e:
        print('FAIL', p, 'line', e.lineno, '->', e.msg)
        lines = src.split(chr(10))
        for i in range(max(0,e.lineno-4), min(len(lines), e.lineno+2)):
            print('   %d| %s' % (i+1, lines[i][:110]))
