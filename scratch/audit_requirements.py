import ast
import os
import json
from pathlib import Path
import sys

# Standard library modules in Python 3.13
stdlib = sys.stdlib_module_names

requirements_path = Path("requirements.txt")
req_lines = [l.strip().split("==")[0].lower() for l in requirements_path.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]

all_imports = set()

# 1. Check all python files
for root, _, files in os.walk("."):
    if any(x in root for x in [".git", ".pytest_cache", "scratch", ".venv", ".venv_test", "v"]):
        continue
    for f in files:
        if f.endswith(".py"):
            p = Path(root) / f
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            all_imports.add((n.name.split(".")[0], str(p)))
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            all_imports.add((node.module.split(".")[0], str(p)))
            except Exception as e:
                print(f"Error parsing {p}: {e}")

# 2. Check notebook
nb_path = Path("notebooks/01_eda.ipynb")
if nb_path.exists():
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            code = "".join(cell.get("source", []))
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            all_imports.add((n.name.split(".")[0], f"notebooks/01_eda.ipynb cell {idx+1}"))
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            all_imports.add((node.module.split(".")[0], f"notebooks/01_eda.ipynb cell {idx+1}"))
            except Exception as e:
                # ignore cell magics like %matplotlib inline
                pass

# Local project modules
local_modules = {"src", "config", "compute_aqi", "features", "evaluate", "split", "train", "final_evaluate", "error_analysis", "explain", "predict", "load_raw", "daily_aggregation", "preprocess", "make_readme_tables"}

# Mapping package import names to distribution names
pkg_map = {
    "sklearn": "scikit-learn",
    "dateutil": "python-dateutil",
}

missing_reqs = []
checked_pkgs = set()

for mod, src in sorted(list(all_imports)):
    if mod in stdlib or mod in local_modules:
        continue
    dist_name = pkg_map.get(mod, mod).lower()
    checked_pkgs.add(dist_name)
    if dist_name not in req_lines:
        missing_reqs.append((mod, dist_name, src))

print("All external distribution packages imported:", sorted(list(checked_pkgs)))
print("Packages in requirements.txt:", sorted(req_lines))
print("Missing from requirements.txt:", missing_reqs)
