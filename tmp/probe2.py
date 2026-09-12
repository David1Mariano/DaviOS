import sys, os
sys.path.insert(0, os.getcwd())
from brain.tool_request_parser import parse_tool_request, START_MARKER, END_MARKER
print('START_MARKER =', repr(START_MARKER))
print('END_MARKER   =', repr(END_MARKER))
text = '>>>TOOL_REQUEST>>>\n{"tool": "time"}\n<<<END_TOOL_REQUEST>>>'
r = parse_tool_request(text)
print('invertido -> found=%r tool=%r error=%r' % (r.found, r.tool_name, r.error))
text3 = '<<<TOOL_REQUEST>>>\n{"tool": "time"}\n<<<END_TOOL_REQUEST>>>'
r3 = parse_tool_request(text3)
print('correto   -> found=%r tool=%r error=%r' % (r3.found, r3.tool_name, r3.error))
from brain.response_sanitizer import sanitize_response_text
for t in (text, 'ok >>>TOOL_REQUEST>>>\n{"tool": "memory_usage"}\n<<<END_TOOL_REQUEST>>>.'):
    print('sanitize:', repr(sanitize_response_text(t)))
