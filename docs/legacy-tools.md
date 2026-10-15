# Earlier interfaces and documents

The maintained desktop entry point is [desktop_app/main.py](../desktop_app/main.py).
Windows users can use [launch_lunar_od.cmd](../launch_lunar_od.cmd).
The current workflow is documented in [RESULT_WORKFLOW.md](../desktop_app/RESULT_WORKFLOW.md).

| File or directory | Status |
| --- | --- |
| lunar_od_app.py | Earlier Tkinter desktop tool; not the current recorded-run workspace |
| ui_server.py and ui/ | Earlier local HTTP/HTML tool; not the current PyQt5 workflow or a hosted site |
| docs/development/UI_README.md | Historical interface proposal |
| docs/development/PYQT5_DESKTOP_APP_BRIEF.md | Earlier desktop implementation brief |
| docs/development/PYQT5_DESKTOP_APP_README.md | Earlier desktop implementation brief, despite the README filename |
| docs/development/plan.md and docs/development/PROJECT_CONTEXT.md | Development planning/context; check dates and qualification reports before relying on status claims |
| archive/ | Preserved worktree snapshots and conflict evidence, not executable production modules |

Earlier tools are retained in their original locations to preserve references
and regression coverage. They are not part of the new public desktop acceptance
gate. They may still run, but do not have the current artifact/session contract.

Historical documents can mention old branch names, python_port paths, test
counts, models and planned features. Current installation guidance is in the root
README and Windows quickstart. Scientific authority belongs to the relevant
qualified model/report, not an old UI brief.

Historical branches remain reachable through the branch-history/2026-10-10/
tags. [The consolidation audit](development/branch_consolidation_2026-10-10/README.md)
records merge decisions and pending-work preservation.
