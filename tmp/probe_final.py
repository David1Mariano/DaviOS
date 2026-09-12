import sys, os
sys.path.insert(0, os.getcwd())
from brain.tool_request_parser import parse_tool_request, sanitize_response_text
r = parse_tool_request('>>>TOOL_REQUEST>>>\n{"tool": "time"}\n<<<END_TOOL_REQUEST>>>')
print('INVERTED:', r.found, r.tool_name, r.error)
r2 = parse_tool_request('<<<TOOL_REQUEST>>>\n{"tool": "time"}\n<<<END_TOOL_REQUEST>>>')
print('NORMAL:', r2.found, r2.tool_name)
print('SAN_INV:', repr(sanitize_response_text('antes >>>TOOL_REQUEST>>>\n{"tool": "time"}\n<<<END_TOOL_REQUEST>>> depois')))
print('SAN_MALFORMED:', repr(sanitize_response_text('oi >>>TOOL_REQUEST>>> lixo sem fim')))
