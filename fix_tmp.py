import ast
import pathlib

t = pathlib.Path("utils/model_downloader.py").read_text(encoding="utf-8")
try:
    ast.parse(t)
    print("AST OK -", len(t.splitlines()), "lines")
except SyntaxError as e:
    print("SYNTAX ERROR:", e.msg, "line", e.lineno, "offset", e.offset)
    ls = t.splitlines()
    lo = max(0, (e.lineno or 1) - 4)
    hi = min(len(ls), (e.lineno or 1) + 4)
    for i in range(lo, hi):
        print(i + 1, "|", repr(ls[i]))