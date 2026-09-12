import pathlib
p = pathlib.Path('brain/tool_request_parser.py')
c = p.read_text(encoding='utf-8')

# Add DaviOS: prefix removal after block removal and normalization
old = """    # 4. Fallback se sobrou nada de útil.
    if not cleaned:
        return _SANITIZE_FALLBACK

    return cleaned"""

new = """    # 4. Remove prefixo 'DaviOS:' do início (caso o Qwen o gere).
    #    Só remove no início, preserva menções no meio do texto.
    cleaned = re.sub(r\"^DaviOS\\s*:\\s*\", \"\", cleaned, count=1)

    # 5. Fallback se sobrou nada de útil.
    if not cleaned:
        return _SANITIZE_FALLBACK

    return cleaned"""

c = c.replace(old, new)
p.write_text(c, encoding='utf-8')
print('ok')
