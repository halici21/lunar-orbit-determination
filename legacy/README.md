# Earlier interfaces

The maintained desktop application is [desktop_app/main.py](../desktop_app/main.py),
started with [launch_lunar_od.cmd](../launch_lunar_od.cmd) on Windows. This folder
keeps the earlier Tkinter and HTTP/HTML tools separate from the public entry points.

Install the checkout with `python -m pip install -e .` before using either tool.

| Earlier command from the repository root | Current command |
|---|---|
| `python lunar_od_app.py` | `python legacy/lunar_od_app.py` |
| `python ui_server.py --port 8420` | `python legacy/ui_server.py --port 8420` |

`python -m legacy.ui_server --help` is also available from the repository root.
The web resources now live in [ui/index.html](ui/index.html) and
[ui/assets/lunar-icon.svg](ui/assets/lunar-icon.svg). HTTP routes such as `/ui/`
and `/results/` retain their existing meanings.

Both tools use this checkout's root `results/` directory. Web experiment/test
subprocesses use this checkout's `examples/` and `tests/`, not a sibling
`python_port` worktree. The optional historical map image retains its previous
external parent-folder location; it is not bundled by this cleanup.

These source-only tools do not implement the maintained PyQt5 artifact/session
contract and are not included in the scientific library wheel. They are retained
for historical use, not promoted as current desktop alternatives.
