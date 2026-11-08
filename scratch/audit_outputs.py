import re
import os

files_to_check = ["README.md", "PROJECT_NOTES.md"]

referenced_outputs = set()

for fn in files_to_check:
    with open(fn, "r", encoding="utf-8") as f:
        text = f.read()
    # Find patterns like outputs/filename.ext or outputs/...
    matches = re.findall(r'outputs/([a-zA-Z0-9_\-\.]+)', text)
    for m in matches:
        # Ignore wildcards or extensions without filename
        if "*" in m or m.endswith("..."):
            continue
        referenced_outputs.add(m)

print(f"Total unique referenced files in outputs/: {len(referenced_outputs)}")

missing_files = []
existing_files = []

for rel in sorted(list(referenced_outputs)):
    full_path = os.path.join("outputs", rel)
    if os.path.exists(full_path):
        existing_files.append(rel)
    else:
        missing_files.append(rel)

print(f"Existing outputs files referenced: {len(existing_files)}")
print(f"Missing outputs files referenced: {len(missing_files)}")
for mf in missing_files:
    print("  MISSING:", mf)
