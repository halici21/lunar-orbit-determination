"""Path-specific deletion of DELETE_CACHE_TEMP files with a manifest.

Only the five named __pycache__ directories of the target worktree are processed.
Each file is checked to be untracked AND git-ignored before deletion; the directory
is removed with os.rmdir only if it is then empty (no recursive delete).
"""
from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
CACHE_DIRS = ["examples/__pycache__", "lunar_od/__pycache__", "lunar_od/reference_config/__pycache__",
              "lunar_od/rf/__pycache__", "tests/__pycache__"]
MANIFEST = ROOT / "artifacts" / "repository_hygiene" / "deletion_manifest.csv"
REF_RESULT = ("git grep: no code/test/doc reference to cache files; README.md:449 only lists "
              "__pycache__/ as ignored")

tracked = set(subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True,
                             check=True).stdout.splitlines())
rows, deleted_bytes = [], 0
for d in CACHE_DIRS:
    dp = ROOT / d
    if not dp.is_dir():
        continue
    for p in sorted(dp.iterdir()):
        rel = p.relative_to(ROOT).as_posix()
        if not p.is_file():
            sys.exit(f"unexpected non-file in cache dir: {rel}")
        if rel in tracked:
            sys.exit(f"refusing: tracked file {rel}")
        ign = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", rel]).returncode
        if ign != 0:
            sys.exit(f"refusing: not git-ignored {rel}")
        data = p.read_bytes()
        kind = "Numba JIT cache" if p.suffix in {".nbi", ".nbc"} else "CPython bytecode"
        rows.append(dict(path=rel, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                         classification="DELETE_CACHE_TEMP",
                         reason=f"{kind}; regenerated automatically on next import/JIT; gitignored (__pycache__/)",
                         authoritative_replacement=f"source module {rel.split('/__pycache__/')[0]}/"
                         f"{p.name.split('.')[0]}.py (regenerates it)",
                         reference_search=REF_RESULT, tracked_status="ignored"))

MANIFEST.parent.mkdir(parents=True, exist_ok=True)
with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

for r in rows:
    (ROOT / r["path"]).unlink()
    deleted_bytes += r["size"]
removed_dirs = []
for d in CACHE_DIRS:
    dp = ROOT / d
    if dp.is_dir() and not any(dp.iterdir()):
        os.rmdir(dp)
        removed_dirs.append(d)
print(f"deleted files: {len(rows)}  bytes: {deleted_bytes}  empty dirs removed: {removed_dirs}")
