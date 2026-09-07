"""Verifica releases do llama.cpp para encontrar builds Windows x64."""

import json
import sys
from pathlib import Path

import urllib.request


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "DaviOS"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    # Tenta ggerganov primeiro, depois ggml-org
    for repo in ["ggerganov/llama.cpp", "ggml-org/llama.cpp"]:
        try:
            releases = fetch(f"https://api.github.com/repos/{repo}/releases?per_page=10")
        except Exception as e:
            print(f"ERROR fetching {repo}: {e}")
            continue

        print(f"\n=== {repo} ===")
        for r in releases:
            tag = r.get("tag_name", "")
            is_prerelease = r.get("prerelease", False)
            win_assets = [
                a
                for a in r.get("assets", [])
                if "win" in a.get("name", "").lower()
                or "amd64" in a.get("name", "").lower()
            ]
            print(f"  {tag} (prerelease={is_prerelease}) -> {len(win_assets)} win assets")
            for a in win_assets:
                size_mb = a.get("size", 0) // (1024 * 1024)
                print(f"    [{size_mb}MB] {a['name']}")
                print(f"         {a['browser_download_url']}")

        # Também verifica releases latest
        try:
            latest = fetch(f"https://api.github.com/repos/{repo}/releases/latest")
            tag = latest.get("tag_name", "")
            print(f"  Latest: {tag}")
            all_assets = latest.get("assets", [])
            print(f"  Total assets: {len(all_assets)}")
            for a in all_assets:
                size_mb = a.get("size", 0) // (1024 * 1024)
                print(f"    [{size_mb}MB] {a['name']}")
                print(f"         {a['browser_download_url']}")
        except Exception as e:
            print(f"  ERROR fetching latest: {e}")


if __name__ == "__main__":
    main()
