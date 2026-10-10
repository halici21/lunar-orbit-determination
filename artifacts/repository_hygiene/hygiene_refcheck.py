"""Read-only reference checker: Markdown links and repo-relative path mentions.

Usage: python hygiene_refcheck.py <worktree> <out.json>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
tracked = [f for f in subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True,
                                     text=True, check=True).stdout.splitlines()]
all_files = set(tracked)
untracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--others"], capture_output=True,
                           text=True, check=True).stdout.splitlines()
exists_any = lambda rel: (ROOT / rel).exists()  # noqa: E731

MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")
# repo-relative paths named in prose/code: top-level dir + at least one segment + extension
PATH_MENTION = re.compile(
    r"(?<![\w./-])((?:artifacts|docs|examples|tests|lunar_od|desktop_app|fixtures|08_manifests|results)"
    r"/[\w./-]+\.(?:py|md|json|csv|png|svg|txt|npz|pdf))")
ABS_GRAD = re.compile(r"(?i)Documents[\\/]{1,2}Python[\\/]{1,2}Grad(?![\w-])")

broken_links, broken_mentions, abs_grad = [], [], []
for f in tracked:
    p = ROOT / f
    if p.suffix not in {".md", ".py", ".json", ".txt"} or f.startswith("artifacts/repository_hygiene/"):
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        if ABS_GRAD.search(line):
            abs_grad.append(dict(file=f, line=i))
        if p.suffix == ".md":
            for m in MD_LINK.finditer(line):
                tgt = m.group(1)
                if re.match(r"^[a-z]+:", tgt) or tgt.startswith("<"):
                    continue
                if not (p.parent / tgt).exists() and not exists_any(tgt):
                    broken_links.append(dict(file=f, line=i, target=tgt))
        for m in PATH_MENTION.finditer(line):
            tgt = m.group(1).rstrip(".")
            if "*" in tgt or "{" in tgt or "<" in tgt:
                continue
            if not exists_any(tgt):
                broken_mentions.append(dict(file=f, line=i, target=tgt))

result = dict(broken_markdown_links=broken_links, missing_path_mentions=broken_mentions,
              absolute_grad_path_lines=len(abs_grad),
              absolute_grad_path_files=sorted({a["file"] for a in abs_grad}))
Path(sys.argv[2]).write_text(json.dumps(result, indent=1), encoding="utf-8")
print("broken markdown links:", len(broken_links))
print("missing path mentions:", len(broken_mentions), "unique targets:",
      len({b["target"] for b in broken_mentions}))
print("absolute Grad path lines:", len(abs_grad), "in", len(result["absolute_grad_path_files"]), "files")
