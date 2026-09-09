#!/usr/bin/env python3
"""Reescribe brain/action_manager.py de forma limpia."""
import pathlib

content = pathlib.Path("tmp/action_manager_clean.txt").read_text(encoding="utf-8")
pathlib.Path("brain/action_manager.py").write_text(content, encoding="utf-8")
print("action_manager.py reescrito OK")