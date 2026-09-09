import urllib.request
import json

# Get release info with pagination
url = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5'
req = urllib.request.Request(url, headers={'User-Agent': 'DaviOS/1.0'})
with urllib.request.urlopen(req, timeout=30) as r:
    releases = json.loads(r.read())

print(f"Found {len(releases)} releases")
for rel in releases[:1]:
    print(f"\nRelease: {rel['tag_name']} ({rel['published_at']})")
    print(f"  Assets: {len(rel['assets'])}")
    for a in rel['assets']:
        name = a['name'].lower()
        if 'win' in name or 'vulkan' in name or 'x64' in name:
            print(f"  [WIN] {a['name']:55s} {a['size']:>12} bytes  {a['browser_download_url']}")
        else:
            print(f"        {a['name']:55s} {a['size']:>12} bytes")
