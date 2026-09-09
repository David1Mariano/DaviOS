#!/usr/bin/env python3
"""Fix apply_style to use lazy import - idempotent version."""
import re

with open('memory/memory_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already fixed
if 'from brain.style_profile import apply_style_substitutions' in content:
    print('ALREADY FIXED - apply_style has lazy import')
else:
    # Find the apply_style method and add lazy import before the return statement
    idx = content.find('return apply_style_substitutions(text, self.style_profile)')
    if idx >= 0:
        line_start = content.rfind('\n', 0, idx) + 1
        insert = '        from brain.style_profile import apply_style_substitutions\n\n'
        content = content[:line_start] + insert + content[line_start:]
        with open('memory/memory_manager.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print('INSERT SUCCESS')
    else:
        print('ERROR: Could not find apply_style_substitutions call')

# Verify
with open('memory/memory_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()
idx = content.find('def apply_style')
print('\nCurrent apply_style method:')
print(content[idx:idx+200])
