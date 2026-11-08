"""
scratch/verify_drafts.py
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

with open(REPO_ROOT / 'docs/report_draft.md', encoding='utf-8') as f:
    text = f.read()

print('=== DRAFT AUDIT ===')
print('Length:', len(text))
print('[TODO: me] count:', text.count('[TODO: me]'))

headlines = [
    'statistically indistinguishable from climatology',
    'shifts from daily readings toward calendar features as the horizon grows',
    'no demonstrated advantage over the trivial winter rule',
    'pending',
    'unverified'
]

for h in headlines:
    found = h.lower() in text.lower()
    print(f"Headline '{h}': {'FOUND' if found else 'MISSING'}")

print('\n=== SLIDES AUDIT ===')
with open(REPO_ROOT / 'docs/slides_outline.md', encoding='utf-8') as f:
    slides_text = f.read()
print('Slides length:', len(slides_text))
print('Slide headings count:', slides_text.count('### Slide'))
print('[TODO: me] count in slides:', slides_text.count('[TODO: me]'))

print('\n=== NUMBERS USED AUDIT ===')
df_nums = pd.read_csv(REPO_ROOT / 'docs/numbers_used.csv')
print('Tracked numbers count:', len(df_nums))
print('Sample rows:')
print(df_nums.head(10).to_string())
