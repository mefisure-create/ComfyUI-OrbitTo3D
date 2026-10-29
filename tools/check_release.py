#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []
warnings = []

# Compile every shipped Python source.
for p in ROOT.rglob("*.py"):
    if "__pycache__" in p.parts:
        continue
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception as e:
        errors.append(f"py_compile {p.relative_to(ROOT)}: {e}")

# Public API contract.
init = (ROOT / "__init__.py").read_text()
expected = {
    "Orbit3DLoadModel",
    "Orbit3DMultiViewConfig",
    "Orbit3DSparseGenerator",
    "Orbit3DShapeGenerator",
    "Orbit3DShapeCascadeGenerator",
    "Orbit3DVGGTViewSelector",
    "Orbit3DRealBasicVSR",
    "Orbit3DMoGeGeometryListToBatch",
    "Orbit3DMoGeGeometryBatchToList",
    "Orbit3DAngleGrid",
    "Orbit3DMoGeInference",
}
tree = ast.parse(init)
mapping = None
for n in tree.body:
    if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS" for t in n.targets
    ):
        if isinstance(n.value, ast.Dict):
            mapping = {ast.literal_eval(k): None for k in n.value.keys}
if mapping is None:
    errors.append("NODE_CLASS_MAPPINGS not found")
elif set(mapping) != expected:
    errors.append(f"public node IDs differ: {sorted(set(mapping) ^ expected)}")

# Extended MoGe node must inherit native execution rather than copy it.
moge_path = ROOT / "orbit3d_ext" / "moge_nodes.py"
moge_tree = ast.parse(moge_path.read_text())
moge_classes = [
    n for n in moge_tree.body if isinstance(n, ast.ClassDef) and n.name == "Orbit3DMoGeInference"
]
if len(moge_classes) != 1:
    errors.append("Orbit3DMoGeInference class missing or duplicated")
else:
    moge_cls = moge_classes[0]
    if not any(isinstance(b, ast.Name) and b.id == "_ComfyMoGeInference" for b in moge_cls.bases):
        errors.append("Orbit3DMoGeInference no longer inherits native MoGeInference")
    if any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "execute"
        for n in moge_cls.body
    ):
        errors.append("Orbit3DMoGeInference must not override native execute()")

# Public Orbit wrapper must not reintroduce removed legacy sockets.
public = (ROOT / "orbit_nodes.py").read_text()
for banned in ("IMAGE_COND", "MOGE_CAM_CONFIG"):
    if banned in public:
        errors.append(f"legacy socket token present in orbit_nodes.py: {banned}")

# Requirements and pyproject direct dependency lists must remain aligned.
req = [
    x.strip()
    for x in (ROOT / "requirements.txt").read_text().splitlines()
    if x.strip() and not x.lstrip().startswith("#")
]
pyproject = (ROOT / "pyproject.toml").read_text()
for dep in req:
    if f'"{dep}"' not in pyproject:
        errors.append(f"{dep} in requirements.txt but not pyproject dependencies")
if "opencv-python" in req or '"opencv-python"' in pyproject:
    errors.append("opencv-python should not be a retained direct dependency")

# Required documentation layout.
required_docs = [
    ROOT / "README.md",
    ROOT / "docs" / "INSTALLATION.md",
    ROOT / "docs" / "FEATURES.md",
    ROOT / "docs" / "REGRESSION.md",
    ROOT / "UPSTREAM.md",
    ROOT / "RELEASE_CHECKLIST.md",
    ROOT / "THIRD_PARTY_NOTICES.md",
    ROOT / "ORBIT_BASELINE.md",
]
for p in required_docs:
    if not p.exists():
        errors.append(f"required release documentation missing: {p.relative_to(ROOT)}")

readme = (ROOT / "README.md").read_text()
if len(readme.splitlines()) > 120:
    errors.append("README front page is no longer concise (>120 lines)")
for link in ("docs/INSTALLATION.md", "docs/FEATURES.md", "docs/REGRESSION.md", "UPSTREAM.md"):
    if link not in readme:
        errors.append(f"README missing documentation link: {link}")
if "No patch to `comfy_extras/nodes_moge.py` is required" not in readme:
    errors.append("README lost stock-ComfyUI MoGe/no-core-patch guidance")

install_doc = (ROOT / "docs" / "INSTALLATION.md").read_text()
for token in ("--cache-none", "flash_attn", "natten", "Orbit3DMoGeInference"):
    if token not in install_doc:
        errors.append(f"docs/INSTALLATION.md missing required runtime guidance: {token}")

upstream_doc = (ROOT / "UPSTREAM.md").read_text()
for token in ("tools/check_upstream.py", "tools/update_upstream_manifest.py", "upstream_symbols.json"):
    if token not in upstream_doc:
        errors.append(f"UPSTREAM.md missing synchronization mechanism: {token}")
if not (ROOT / "tools" / "update_upstream_manifest.py").exists():
    errors.append("upstream rebaseline tool missing")
if not (ROOT / ".github" / "workflows" / "upstream-contract.yml").exists():
    errors.append("scheduled upstream contract workflow missing")

# Upstream local hash contract.
data = json.loads((ROOT / "upstream_symbols.json").read_text())

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

for item in data["files"]:
    p = ROOT / item["path"]
    if not p.exists():
        errors.append(f'manifest path missing: {item["path"]}')
    elif item["path"] != "upstream_symbols.json" and sha(p) != item["local_sha256"]:
        errors.append(f'local file drift: {item["path"]}')

# Every shipped release file (except the manifest itself) must be represented.
# In a Git working tree, the release payload is the set of tracked files. This
# deliberately excludes local repository state and build/test artefacts such as
# .git/, .ruff_cache/, node.zip, etc. In an unpacked source archive (no .git),
# fall back to the actual files present so archive contamination is still caught.
manifest_paths = {item["path"] for item in data["files"]}

def release_payload_paths() -> set[str]:
    if (ROOT / ".git").exists():
        try:
            result = subprocess.run(
                ["git", "-C", str(ROOT), "ls-files", "-z"],
                check=True,
                capture_output=True,
            )
            return {
                path
                for path in result.stdout.decode("utf-8").split("\0")
                if path and path != "upstream_symbols.json"
            }
        except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as e:
            warnings.append(f"git ls-files unavailable; falling back to filesystem scan: {e}")

    return {
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.name != "upstream_symbols.json"
    }

actual_paths = release_payload_paths()
missing_from_manifest = sorted(actual_paths - manifest_paths)
stale_manifest = sorted(manifest_paths - actual_paths)
if missing_from_manifest:
    errors.append(f"release files missing from manifest: {missing_from_manifest}")
if stale_manifest:
    errors.append(f"manifest references non-shipped files: {stale_manifest}")

# Registry metadata may remain placeholder for an RC, but must be resolved before publication.
if 'PublisherId = "local-orbit-to-3d"' in pyproject:
    errors.append("Comfy Registry PublisherId is still the RC placeholder")
if 'PublisherId = "fadj"' not in pyproject:
    errors.append("Comfy Registry PublisherId does not match the release publisher: fadj")
if 'name = "orbit-to-3d"' not in pyproject:
    errors.append("Comfy Registry package ID does not match the release package: orbit-to-3d")
for required_url in (
    'Repository = "https://github.com/mefisure-create/ComfyUI-OrbitTo3D"',
    'Documentation = "https://github.com/mefisure-create/ComfyUI-OrbitTo3D/tree/main/docs"',
    '"Bug Tracker" = "https://github.com/mefisure-create/ComfyUI-OrbitTo3D/issues"',
):
    if required_url not in pyproject:
        errors.append(f"pyproject.toml missing release URL: {required_url}")

if errors:
    print("RELEASE CHECK FAILED")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("Release checks passed.")
print(f"Public Orbit nodes: {len(expected)}")
print(f'Manifest files: {len(data["files"])}; tracked symbols: {len(data["tracked_symbols"])}')
for w in warnings:
    print("WARNING:", w)
