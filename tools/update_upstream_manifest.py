#!/usr/bin/env python3
"""Advance Orbit's reviewed ComfyUI-Trellis2 baseline safely.

This tool does not port or merge upstream code. It only updates provenance hashes
*after* a check_upstream report has been reviewed and the full regression passed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "upstream_symbols.json"


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{p.stderr.strip()}")
    return p.stdout


def git_file(repo: Path, ref: str, path: str) -> str | None:
    p = subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{path}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.stdout if p.returncode == 0 else None


def sha_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ast_sha(node: ast.AST) -> str:
    payload = ast.dump(node, annotate_fields=True, include_attributes=False).encode()
    return hashlib.sha256(payload).hexdigest()


def find_symbol(source: str, dotted: str) -> str | None:
    tree = ast.parse(source)
    cur = tree.body
    node = None
    for i, part in enumerate(dotted.split(".")):
        node = next(
            (
                n
                for n in cur
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.name == part
            ),
            None,
        )
        if node is None:
            return None
        if i + 1 < len(dotted.split(".")):
            if not isinstance(node, ast.ClassDef):
                return None
            cur = node.body
    return ast_sha(node)


def atomic_json_write(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-repo", required=True, type=Path)
    ap.add_argument("--reviewed-report", required=True, type=Path,
                    help="JSON report produced by tools/check_upstream.py")
    ap.add_argument("--new-base", required=True,
                    help="exact reviewed upstream commit/ref to pin")
    ap.add_argument("--accept-review", action="store_true",
                    help="confirm every finding in the report was reviewed")
    ap.add_argument("--regression-passed", action="store_true",
                    help="confirm the full Orbit regression passed after any ports")
    args = ap.parse_args()

    if not args.accept_review or not args.regression_passed:
        raise SystemExit(
            "Refusing to advance baseline: both --accept-review and "
            "--regression-passed are required."
        )

    repo = args.upstream_repo.expanduser().resolve()
    if not (repo / ".git").exists():
        raise SystemExit(f"not a git clone: {repo}")

    data = json.loads(MANIFEST.read_text())
    report = json.loads(args.reviewed_report.read_text())
    old_base = data["upstream"]["base_commit"]
    new_base = run_git(repo, "rev-parse", args.new_base).strip()

    if report.get("base") != old_base:
        raise SystemExit(
            f"review report base {report.get('base')} does not match manifest base {old_base}"
        )
    if report.get("target_commit") != new_base:
        raise SystemExit(
            f"review report target {report.get('target_commit')} does not match new base {new_base}"
        )

    ancestry = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", old_base, new_base]
    )
    if ancestry.returncode != 0:
        raise SystemExit(f"new base {new_base} is not a descendant of old base {old_base}")

    # A file labelled verbatim must actually be byte-for-byte equivalent to the
    # newly reviewed upstream baseline before we advance the pin.
    for item in data["tracked_upstream_files"]:
        path = item["path"]
        local_path = ROOT / path
        if not local_path.exists():
            raise SystemExit(f"tracked local file missing: {path}")
        upstream = git_file(repo, new_base, path)
        if item["mode"] == "upstream_verbatim_base":
            if upstream is None:
                raise SystemExit(f"verbatim upstream file missing at new base: {path}")
            if local_path.read_text() != upstream:
                raise SystemExit(
                    f"verbatim file differs from new upstream base: {path}; "
                    "update it or reclassify it as derived_modified before rebaselining"
                )
        item["local_sha256"] = sha_bytes(local_path)

    # Refresh local AST hashes for the explicitly tracked derived symbols. A
    # missing symbol is never silently accepted during rebaseline.
    for item in data["tracked_symbols"]:
        path = ROOT / item["path"]
        if not path.exists():
            raise SystemExit(f"tracked symbol file missing: {item['path']}")
        current = find_symbol(path.read_text(), item["symbol"])
        if current is None:
            raise SystemExit(f"tracked symbol missing locally: {item['path']}::{item['symbol']}")
        item["local_ast_sha256"] = current

    # Refresh all local release hashes already registered in the manifest.
    for item in data["files"]:
        path = ROOT / item["path"]
        if not path.exists():
            raise SystemExit(f"manifest file missing: {item['path']}")
        item["local_sha256"] = sha_bytes(path)

    data["upstream"]["base_commit"] = new_base
    data["upstream"]["checked_main_commit"] = new_base
    data["upstream"]["checked_date"] = date.today().isoformat()

    atomic_json_write(MANIFEST, data)
    print(f"Advanced reviewed Trellis2 baseline: {old_base} -> {new_base}")
    print(f"Reviewed findings recorded in: {args.reviewed_report}")
    print("Run: python tools/check_release.py")
    print("Then update the human-readable baseline/date in UPSTREAM.md if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
