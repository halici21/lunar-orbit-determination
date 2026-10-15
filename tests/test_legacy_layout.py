"""Protect asset serving and checkout-local paths after moving earlier tools."""
import json
from http.server import HTTPServer
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

from legacy import lunar_od_app as tk_app
from legacy import ui_server as web

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def legacy_http():
    server = HTTPServer(("127.0.0.1", 0), web.LunarODHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize(("route", "asset"), [
    ("/", "index.html"), ("/ui/assets/lunar-icon.svg", "assets/lunar-icon.svg"),
])
def test_legacy_http_routes_serve_the_moved_assets(legacy_http, route, asset):
    expected = (ROOT / "legacy/ui" / asset).read_bytes()
    with urlopen(legacy_http + route, timeout=5) as response:
        assert response.status == 200
        assert response.read() == expected


@pytest.mark.parametrize("dry_run", [True, False])
def test_legacy_experiment_uses_this_checkout(legacy_http, monkeypatch, dry_run):
    monkeypatch.setattr(web, "_load_experiments", lambda: [
        {"experiment_id": "sample", "script": "quickstart_od.py"}])
    calls = []
    def spawn(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(web.subprocess, "Popen", spawn)
    request = Request(legacy_http + "/api/run-experiment", method="POST",
                      data=json.dumps({"experiment_id": "sample", "dry_run": dry_run}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=5) as response:
        result = json.load(response)
    expected = [sys.executable, str(ROOT / "examples/quickstart_od.py")]
    if dry_run:
        assert result["command"] == expected
        assert Path(result["cwd"]) == ROOT
        assert not calls
    else:
        assert result["status"] == "started"
        assert calls[0][0] == expected
        assert Path(calls[0][1]["cwd"]) == ROOT


def test_legacy_test_runner_uses_this_checkout(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stderr="Ran 3 tests in 0.1s\nOK", stdout="")
    monkeypatch.setattr(web, "_test_status_cache", None)
    monkeypatch.setattr(web.subprocess, "run", run)
    assert web._run_test_status()["test_count"] == 3
    command, options = calls[0]
    assert Path(command[command.index("-s") + 1]) == ROOT / "tests"
    assert Path(command[command.index("-t") + 1]) == ROOT
    assert Path(options["cwd"]) == ROOT


def test_legacy_web_help_from_an_unrelated_directory(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "legacy/ui_server.py"), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--port" in result.stdout


def test_legacy_web_module_command():
    result = subprocess.run([sys.executable, "-m", "legacy.ui_server", "--help"],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--port" in result.stdout


def test_legacy_result_directories_stay_at_the_checkout_root():
    assert tk_app.RESULTS_DIR == ROOT / "results"
    assert tk_app.SCRIPT_DIR == ROOT
    assert web._RESULTS_DIR == ROOT / "results"
