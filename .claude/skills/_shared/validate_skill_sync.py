"""Validate that duplicated skill roots have not silently diverged.

Canonical root: the repository-tracked ``.claude/skills`` (see CLAUDE.md,
"Skill roots and synchronization"). For every skill name present in any root
this script compares, against the canonical copy:

- frontmatter ``name``,
- frontmatter ``description`` (whitespace-normalized, so YAML re-folding is
  not a false positive),
- the remaining frontmatter keys/values,
- a SHA-256 hash of the body below the frontmatter.

Every mismatch is reported; exit code 1 if any mismatch exists. Copies are
never modified or deleted here.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

CANONICAL = Path(__file__).resolve().parents[1]  # this repo's .claude/skills
ROOTS = {
    "user-level (~/.claude/skills)": Path.home() / ".claude" / "skills",
}
# Sibling worktrees of the same repository (best effort discovery).
_grad = CANONICAL.parents[1].parent
for candidate in sorted(_grad.glob("*/.claude/skills")):
    if candidate.resolve() != CANONICAL.resolve():
        ROOTS[f"worktree ({candidate.parents[1].name})"] = candidate

_FRONT = re.compile(r"(?s)\A---\n(.*?)\n---\n?(.*)\Z")
_DESC = re.compile(r"^description:(?:\s*>-?\n((?:[ \t]+\S.*\n?)+)|[ \t]*(.*)\n?)", re.MULTILINE)


def parse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = _FRONT.match(text)
    if not match:
        return {"error": "no frontmatter"}
    front, body = match.groups()
    desc_match = _DESC.search(front)
    if desc_match:
        raw = desc_match.group(1) or desc_match.group(2) or ""
        description = " ".join(raw.split())
        rest = front[: desc_match.start()] + front[desc_match.end():]
    else:
        description = None
        rest = front
    name_match = re.search(r"^name:[ \t]*(\S+)", front, re.MULTILINE)
    return {
        "name": name_match.group(1) if name_match else None,
        "description": description,
        "other_frontmatter": " ".join(rest.split()),
        "body_sha256": hashlib.sha256(body.strip().encode("utf-8")).hexdigest(),
    }


def main() -> int:
    canonical_skills = {
        p.parent.name: parse(p) for p in sorted(CANONICAL.glob("*/SKILL.md"))
    }
    mismatches = 0
    print(f"canonical root: {CANONICAL}")
    for label, root in ROOTS.items():
        if not root.is_dir():
            print(f"[{label}] root missing: {root}")
            continue
        other_skills = {p.parent.name: parse(p) for p in sorted(root.glob("*/SKILL.md"))}
        for skill in sorted(set(canonical_skills) | set(other_skills)):
            here = canonical_skills.get(skill)
            there = other_skills.get(skill)
            if here is None:
                print(f"[{label}] {skill}: present only in this root (not canonical)")
                continue
            if there is None:
                print(f"[{label}] {skill}: MISSING in this root")
                mismatches += 1
                continue
            for field in ("name", "description", "other_frontmatter", "body_sha256"):
                if here.get(field) != there.get(field):
                    print(f"[{label}] {skill}: MISMATCH in {field}")
                    mismatches += 1
    print(f"total mismatches: {mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
