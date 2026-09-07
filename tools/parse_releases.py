"""Parse llama.cpp releases JSON to find Windows Vulkan builds."""

import json
import os
import sys
from pathlib import Path

tmp_file = Path(os.environ["TEMP"]) / "llama_releases.json"
print(f"Looking at: {tmp_file}")

if not tmp_file.exists():
    print("ERROR: file not found")
    sys.exit(1)

with open(tmp_file, "r", errors="replace") as f:
    data = json.load(f)

print(f"Type: {type(data)}, Length: {len(data) if isinstance(data, list) else 'N/A'}")

if isinstance(data, list):
    for r in data[:5]:
        tag = r.get("tag_name", "?")
        pr = r.get("prerelease", False)
        assets = r.get("assets", [])
        print(f"\n{tag} (prerelease={pr}) -> {len(assets)} assets")
        for a in assets:
            n = a.get("name", "")
            s = a.get("size", 0) // (1024 * 1024)
            if "vulkan" in n.lower() or "win" in n.lower():
                print(f"  [{s}MB] {n}")
                print(f"  URL: {a['browser_download_url']}")
else:
    print(json.dumps(data, indent=2)[:2000])
