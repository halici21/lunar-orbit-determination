"""Generate public signatures and docstrings directly from Python source (no imports)."""
from __future__ import annotations

import ast
from pathlib import Path

PUBLIC = {
    "config": ("Station",),
    "geometry": ("geodetic_to_ecef_wgs84",),
    "dynamics": ("propagate_augmented_state",),
    "measurements": ("PassGeometry", "compute_position_residuals_analytic"),
    "estimators": ("EstimatorStats", "estimate_position_bls_lm", "estimate_position_srif"),
    "filters": ("UnscentedTransformConfig", "UKFAdaptiveConfig", "LunarUKFResult", "run_lunar_ukf"),
    "scenarios": ("PreparedArc", "run_batch_arc_sequence"),
    "scenario_config": ("ScenarioConfig", "scenario_config_from_mapping"),
}


def render_reference(root: Path) -> str:
    sections = ["## Generated public reference\n\n"
                "Signatures, defaults, annotated fields and docstrings below are extracted "
                "from this revision's source at every documentation build. Scientific "
                "dependencies are never imported by the generator.\n"]
    for module, names in PUBLIC.items():
        path = root / "src/lunar_od" / (module + ".py")
        nodes = {node.name: node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name in names:
            node = nodes[name]
            if isinstance(node, ast.FunctionDef):
                signature = f"def {name}({ast.unparse(node.args)})"
                if node.returns:
                    signature += " -> " + ast.unparse(node.returns)
            else:
                fields = [ast.unparse(field) for field in node.body
                          if isinstance(field, ast.AnnAssign)]
                signature = f"class {name}:\n" + "\n".join("    " + f for f in fields)
            url = ("https://github.com/halici21/lunar-orbit-determination/blob/main/"
                   f"src/lunar_od/{module}.py#L{node.lineno}")
            sections.append(f"### `{module}.{name}`\n\n[Source]({url})\n\n"
                            f"```python\n{signature}\n```\n\n"
                            + (ast.get_docstring(node) or "See the contracts and examples above.") + "\n")
    return "\n".join(sections)
