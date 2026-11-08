import re
import os
import glob
import json

target_files = [
    "PROJECT_NOTES.md",
    "README.md",
    *glob.glob("outputs/*.md"),
    "notebooks/01_eda.ipynb"
]

cause_words = ["because", "due to", "caused", "driven by", "reflects", "explains", "dust", "stubble", "lockdown", "monsoon", "crop burning"]
exempt_phrases = ["hypothesis", "possible explanation", "not tested", "cause unknown", "unverified"]

findings = []

for filepath in target_files:
    if not os.path.exists(filepath):
        continue
    
    if filepath.endswith(".ipynb"):
        with open(filepath, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for c_idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "markdown":
                text = "".join(cell.get("source", []))
                for line_idx, line in enumerate(text.splitlines(), 1):
                    for cw in cause_words:
                        if re.search(r'\b' + re.escape(cw) + r'\b', line, re.I):
                            # Check if line or adjacent text has exempt phrases
                            has_exempt = any(ep in line.lower() for ep in exempt_phrases)
                            findings.append({
                                "file": filepath,
                                "loc": f"cell {c_idx+1}, line {line_idx}",
                                "line": line.strip(),
                                "trigger": cw,
                                "has_exempt": has_exempt
                            })
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                for cw in cause_words:
                    if re.search(r'\b' + re.escape(cw) + r'\b', line, re.I):
                        has_exempt = any(ep in line.lower() for ep in exempt_phrases)
                        findings.append({
                            "file": filepath,
                            "loc": f"line {line_idx}",
                            "line": line.strip(),
                            "trigger": cw,
                            "has_exempt": has_exempt
                        })

print(f"Total lines matched: {len(findings)}")
unexempt = [f for f in findings if not f["has_exempt"]]
print(f"Total unexempt potential causal statements: {len(unexempt)}")
for u in unexempt[:30]:
    print(f"[{u['file']} {u['loc']}] Trigger: '{u['trigger']}' -> {u['line'][:120]}")

print("=== FULL UNEXEMPT LIST ===")
for i, u in enumerate(unexempt, 1):
    print(f"{i}. [{u['file']} {u['loc']}] Trigger: '{u['trigger']}'")
    print(f"   Line: {u['line']}\n")
