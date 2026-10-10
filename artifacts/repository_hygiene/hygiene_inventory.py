"""Read-only repository inventory for the local hygiene audit.

Usage: python hygiene_inventory.py <worktree> <pre|post>
Writes artifacts/repository_hygiene/<phase>_cleanup_inventory.json and _files.csv.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PHASE = sys.argv[2]
OUT = ROOT / "artifacts" / "repository_hygiene"
TEXT_EXT = {".py", ".md", ".json", ".txt", ".ini", ".toml", ".cfg", ".yml", ".yaml",
            ".ui", ".qss", ".html", ".js", ".css", ".sh", ".ps1", ".bat", ".csv"}
PROTECTED = {"PYQT5_DESKTOP_APP_README.md", "examples/r2_measurement_fidelity_validation.py",
             "docs/phase17_geo_literature_contract.md",
             "docs/phase17_geo_geometry_identifiability.md"}
ORPHANS = {"examples/phase17_r1od_fd_convergence.py", "examples/phase17_r1od_qualification.py"}
HYGIENE_OWN = ("artifacts/repository_hygiene/", "docs/local_repository_hygiene_status_report.md")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


tracked = set(git("ls-files", "-z").split("\0")) - {""}
untracked = set(git("ls-files", "-z", "--others", "--exclude-standard").split("\0")) - {""}
ignored = set(git("ls-files", "-z", "--others", "--ignored", "--exclude-standard").split("\0")) - {""}
porcelain = {}
for line in git("status", "--porcelain=v1", "--ignored", "-uall").splitlines():
    porcelain[line[3:].strip('"')] = line[:2]

files = sorted(tracked | untracked | ignored)
files = [f for f in files if (ROOT / f).is_file()]

# Reference corpus: every tracked or untracked text file (ignored outputs excluded).
corpus: dict[str, str] = {}
for f in sorted(tracked | untracked):
    if f.startswith(HYGIENE_OWN):
        continue  # the audit's own outputs list every path; they are not references
    p = ROOT / f
    if p.is_file() and p.suffix.lower() in TEXT_EXT and p.stat().st_size < 5_000_000:
        corpus[f] = p.read_text(encoding="utf-8", errors="replace")


def ref_bucket(referrer: str) -> str:
    if referrer.startswith("tests/"):
        return "tests"
    if referrer.endswith(".py"):
        return "code"
    if referrer.endswith(".md"):
        return "docs"
    return "other"


def find_refs(f: str) -> dict[str, list[str]]:
    base = Path(f).name
    needles = [base]
    if f.endswith(".py") and Path(f).stem not in {"__init__", "conftest"}:
        needles.append(Path(f).stem)
    if base in {"__init__.py", "README.md", "CLAUDE.md", "LICENSE", "conftest.py"}:
        needles = [f]  # generic names: only full-path mentions count
    out: dict[str, list[str]] = defaultdict(list)
    for r, text in corpus.items():
        if r == f:
            continue
        for n in needles:
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])", text):
                out[ref_bucket(r)].append(r)
                break
    return out


graph_files: set[str] = set()
gpath = ROOT / "graphify-out" / "graph.json"
if gpath.is_file():
    g = json.loads(gpath.read_text(encoding="utf-8"))
    graph_files = {n.get("source_file", "").replace("\\", "/") for n in g["nodes"]}


def classify(f: str, status: str) -> tuple[str, str, str]:
    top = f.split("/")[0]
    name = Path(f).name
    if f.startswith(HYGIENE_OWN):
        return "KEEP_GENERATED_EVIDENCE", "keep", "output of this hygiene audit"
    if "__pycache__/" in f or f.endswith((".pyc", ".pyo")):
        return "DELETE_CACHE_TEMP", "delete", "Python bytecode cache; regenerated on import; gitignored"
    if f in ORPHANS:
        return ("REVIEW_REQUIRED", "keep",
                "untracked R1O-D orphan; recorded in artifacts/r1od_manifest.json "
                "'orphaned_scripts_not_committed'; imports removed delta_dor API; holds "
                "first-pass FD history not in the R1O-D report")
    if f in PROTECTED:
        return ("KEEP_PROVENANCE" if f.startswith("docs/") or "r2_" in f else "KEEP_ACTIVE",
                "keep", "protected by task instruction")
    if top == "graphify-out":
        return "KEEP_ACTIVE", "keep", "gitignored Graphify graph used by CLAUDE.md discovery workflow; regenerable"
    if f.startswith("results/phase17_geo_cache/"):
        return ("KEEP_GENERATED_EVIDENCE", "keep",
                "GEO cached design matrices read by phase17_geo_figures/fixture_attribution; "
                "regeneration needs full grid re-propagation")
    if top == "artifacts":
        return "KEEP_GENERATED_EVIDENCE", "keep", "phase evidence artifact (tracked)"
    if f.startswith("docs/figures/"):
        return "KEEP_GENERATED_EVIDENCE", "keep", "report figure / figure data"
    if top == "08_manifests":
        return "KEEP_PROVENANCE", "keep", "phase manifest"
    if top == "docs":
        if re.match(r"docs/(phase\d|r\d|p0b|.*erratum|.*literature_contract|.*qualification|.*validation)", f):
            return "KEEP_PROVENANCE", "keep", "phase report / contract / erratum"
        return "KEEP_ACTIVE", "keep", "current technical documentation"
    if top == "examples":
        if name.endswith("_core.py"):
            return "KEEP_ACTIVE", "keep", "shared phase analysis helper imported by later phases"
        if name.startswith("phase17_geo"):
            return "KEEP_ACTIVE", "keep", "current (latest) phase script"
        if re.match(r"(phase\d|r\d_|p0b|m\d_)", name):
            return "KEEP_PROVENANCE", "keep", "historical phase reproduction script"
        return "KEEP_ACTIVE", "keep", "example / runner script"
    if top in {"lunar_od", "desktop_app", "ui", "tests", "fixtures", ".claude"}:
        return "KEEP_ACTIVE", "keep", {"lunar_od": "production source", "desktop_app": "desktop app source",
                                       "ui": "web UI asset", "tests": "test suite",
                                       "fixtures": "test/data fixture", ".claude": "project skills (canonical)"}[top]
    if "/" not in f:
        if name in {"DOPPLER_RANGE_RATE_MODEL_REVIEW.md", "SPICE_CN_CNPLUS_VALIDATION.md",
                    "LUNAR_OD_MEASUREMENT_PHYSICS_AND_FRAME_STATUS_README.md"}:
            return "KEEP_PROVENANCE", "keep", "root-level review/validation record (identical on main)"
        return "KEEP_ACTIVE", "keep", "root project/config file"
    return "UNKNOWN", "keep", "no rule matched"


rows = []
for f in files:
    p = ROOT / f
    st = p.stat()
    status = "tracked" if f in tracked else ("untracked" if f in untracked else "ignored")
    cls, action, why = classify(f, status)
    skip_refs = cls == "DELETE_CACHE_TEMP" or f.startswith(("graphify-out/", "results/"))
    refs = {} if skip_refs else find_refs(f)
    rows.append(dict(
        path=f, size=st.st_size, status=status, git_status=porcelain.get(f, "  " if status == "tracked" else ""),
        ext=p.suffix.lower(), sha256=sha256(p),
        mtime=dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        refs_code=len(refs.get("code", [])), refs_docs=len(refs.get("docs", [])),
        refs_tests=len(refs.get("tests", [])), refs_other=len(refs.get("other", [])),
        ref_examples=";".join(sorted({r for v in refs.values() for r in v})[:5]),
        graphify_visible=f in graph_files, classification=cls, proposed_action=action, reason=why))

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / f"{PHASE}_cleanup_files.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

by_hash = defaultdict(list)
for r in rows:
    if r["size"] > 0 and r["classification"] != "DELETE_CACHE_TEMP" and not r["path"].startswith("graphify-out/"):
        by_hash[r["sha256"]].append(r["path"])
dups = {h: ps for h, ps in by_hash.items() if len(ps) > 1}


def count(key, val):
    return sum(1 for r in rows if r[key] == val)


summary = dict(
    phase=PHASE, generated=dt.datetime.now().isoformat(timespec="seconds"),
    worktree=ROOT.name, branch=git("rev-parse", "--abbrev-ref", "HEAD").strip(),
    head=git("rev-parse", "HEAD").strip(), tree=git("rev-parse", "HEAD^{tree}").strip(),
    files_total=len(rows), tracked=count("status", "tracked"), untracked=count("status", "untracked"),
    ignored=count("status", "ignored"), bytes_total=sum(r["size"] for r in rows),
    classification_counts={c: count("classification", c) for c in sorted({r["classification"] for r in rows})},
    exact_duplicate_groups=dups,
    files=rows)
(OUT / f"{PHASE}_cleanup_inventory.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=1))
