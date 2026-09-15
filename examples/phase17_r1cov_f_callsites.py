"""PHASE 17-R1COV - COV-F, production impact audit (s38-s40).

ANALYSIS SPACE ONLY.  This module only READS source.

s39 forbids silently replacing the shared covariance routine everywhere. Before
any edit, every caller of the floored helpers is located, attributed to its
enclosing function, and classified by whether this phase is permitted to change
it.
"""
from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

REPO = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0")
ARTIFACTS = REPO / "artifacts"
HELPERS = ("_safe_covariance_from_information", "_sqrt_information_from_information")

#: Which enclosing functions are the K_SRP solve-for paths.  Only these may be
#: touched by R1COV; everything else is a protected default path (s39/s40).
K_PATH_FUNCTIONS = {
    "estimate_two_way_range_bls_lm",
    "estimate_two_way_range_srif",
}


def enclosing_function(tree: ast.AST, lineno: int) -> str:
    best, best_line = "<module>", -1
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno and node.lineno > best_line:
                end = getattr(node, "end_lineno", node.lineno)
                if lineno <= end:
                    best, best_line = node.name, node.lineno
    return best


def classify(path: Path, func: str, line_text: str, in_k_branch: bool) -> str:
    p = path.as_posix()
    if "/tests/" in p:
        return "TEST_UTILITY"
    if "/examples/" in p:
        return "ANALYSIS_ONLY_CALLER"
    if "scenarios.py" in p:
        return "OTHER_ESTIMATOR_PATH_duplicate_helper_in_scenarios"
    if func in K_PATH_FUNCTIONS and in_k_branch:
        return "K_AUGMENTED_" + ("BLS" if "bls" in func else "SRIF")
    return "DEFAULT_6STATE_ESTIMATOR"


def main() -> None:
    rows = []
    for path in sorted(REPO.glob("lunar_od/*.py")) + \
            sorted(REPO.glob("tests/*.py")) + sorted(REPO.glob("examples/*.py")):
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not any(h in src for h in HELPERS):
            continue
        lines = src.splitlines()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for i, text in enumerate(lines, start=1):
            for helper in HELPERS:
                if helper not in text:
                    continue
                if re.match(r"\s*(def |#)", text) and "def " + helper in text:
                    kind = "DEFINITION"
                elif text.lstrip().startswith("#"):
                    kind = "COMMENT"
                else:
                    kind = "CALL"
                func = enclosing_function(tree, i)
                # a call inside the solve_for_k_srp branch is preceded by the
                # scale conditioning line added in Phase 17-R
                window = "\n".join(lines[max(0, i - 6):i])
                in_k_branch = "scale.T @ posterior_information @ scale" in window
                rows.append(dict(
                    file=path.relative_to(REPO).as_posix(), line=i,
                    helper=helper, kind=kind, function=func,
                    in_k_branch=in_k_branch,
                    classification=classify(path, func, text, in_k_branch),
                    r1cov_may_modify=(kind == "CALL" and in_k_branch
                                      and func in K_PATH_FUNCTIONS),
                    source=text.strip()[:110]))

    with (ARTIFACTS / "r1cov_callsite_inventory.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("=" * 88)
    print("COV-F  s38  CALL-SITE INVENTORY FOR THE FLOORED COVARIANCE HELPERS")
    print("=" * 88)
    print("%-44s %5s %-34s %s" % ("file", "line", "classification", "may modify"))
    for r in rows:
        if r["kind"] != "CALL":
            continue
        print("%-44s %5d %-34s %s"
              % (r["file"], r["line"], r["classification"],
                 "YES" if r["r1cov_may_modify"] else "no"))
    calls = [r for r in rows if r["kind"] == "CALL"]
    mod = [r for r in calls if r["r1cov_may_modify"]]
    print()
    print("  total call sites            : %d" % len(calls))
    print("  R1COV may modify            : %d" % len(mod))
    print("  protected / out of scope    : %d" % (len(calls) - len(mod)))
    print()
    from collections import Counter
    for k, v in sorted(Counter(r["classification"] for r in calls).items()):
        print("    %-52s %d" % (k, v))
    print()
    print("  Note: scenarios.py carries its OWN duplicate copy of")
    print("  _safe_covariance_from_information. It is a separate definition on a")
    print("  separate path, is not part of the K solve-for chain, and is NOT")
    print("  touched by this phase (s39).")
    print("\n  wrote r1cov_callsite_inventory.csv (%d rows)" % len(rows))


if __name__ == "__main__":
    main()
