#!/usr/bin/env python3
"""Lunar OD Web UI — lightweight HTTP server.

Serves the single-page web UI and exposes JSON API endpoints for
inspecting results, launching experiments, and querying configuration.

Usage:
    python python_port/ui_server.py [--port 8420]

Only stdlib + the project-local ``lunar_od`` package are required.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import mimetypes
import os
import subprocess
import sys
import threading
import traceback
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

# ---------------------------------------------------------------------------
# Path setup — make sure ``lunar_od`` is importable regardless of cwd.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # python_port/
_PROJECT_ROOT = _THIS_DIR.parent                     # Grad/
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Directories we serve from
_UI_DIR = _THIS_DIR / "ui"
_RESULTS_DIR = _THIS_DIR / "results"
_WORLD_MAP_PATH = _PROJECT_ROOT / "dunya-haritasi-dilsiz.png"

# ---------------------------------------------------------------------------
# Lazy imports from the project package (deferred to avoid hard crash at import
# time if e.g. numpy is missing — the server can still serve static files).
# ---------------------------------------------------------------------------
_STATIONS_CACHE: list[dict[str, Any]] | None = None
_SCHEMA_CACHE: dict[str, Any] | None = None


def _load_stations() -> list[dict[str, Any]]:
    """Import station definitions and return JSON-safe dicts."""
    global _STATIONS_CACHE
    if _STATIONS_CACHE is not None:
        return _STATIONS_CACHE

    from lunar_od.config import (
        POSITION_ONLY_STATION_DEFS,
        RANGE_RATE_STATION_DEFS,
    )

    seen: set[str] = set()
    stations: list[dict[str, Any]] = []

    def _family(name: str) -> str:
        for tag in ("ITU", "DSN", "KGS", "ESA", "RUS", "ISRO"):
            if tag in name:
                return tag
        return "Other"

    # Position-only stations
    for name, lat, lon, alt, color in POSITION_ONLY_STATION_DEFS:
        if name in seen:
            continue
        seen.add(name)
        sigma_range = 94.0 if "ITU" in name else 5.0
        sigma_angle = 0.005 if "ITU" in name else 0.001
        stations.append({
            "name": name,
            "lat_deg": lat,
            "lon_deg": lon,
            "alt_m": alt,
            "sigma_range_m": sigma_range,
            "sigma_angle_deg": sigma_angle,
            "sigma_range_rate_mps": None,
            "color_rgb": list(color),
            "family": _family(name),
            "supports_range_rate": False,
        })

    # Range-rate stations (update existing or add new)
    for name, lat, lon, alt, color in RANGE_RATE_STATION_DEFS:
        sigma_rr = 1e-3 if "ITU" in name else 1e-4
        if name in seen:
            # Update the existing entry
            for s in stations:
                if s["name"] == name:
                    s["sigma_range_rate_mps"] = sigma_rr
                    s["supports_range_rate"] = True
                    break
        else:
            seen.add(name)
            sigma_range = 94.0 if "ITU" in name else 5.0
            sigma_angle = 0.005 if "ITU" in name else 0.001
            stations.append({
                "name": name,
                "lat_deg": lat,
                "lon_deg": lon,
                "alt_m": alt,
                "sigma_range_m": sigma_range,
                "sigma_angle_deg": sigma_angle,
                "sigma_range_rate_mps": sigma_rr,
                "color_rgb": list(color),
                "family": _family(name),
                "supports_range_rate": True,
            })

    _STATIONS_CACHE = stations
    return stations


def _load_schema() -> dict[str, Any]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    from lunar_od.scenario_config import scenario_config_schema
    _SCHEMA_CACHE = scenario_config_schema()
    return _SCHEMA_CACHE


# ---------------------------------------------------------------------------
# Result-file helpers
# ---------------------------------------------------------------------------
_CATEGORY_RULES: list[tuple[str, str]] = [
    ("campaign_", "campaign"),
    ("visibility_", "visibility"),
    ("long_", "visibility"),     # long_*visibility* — prefix match is enough
    ("formal_", "handoff"),
    ("thesis_", "factorial"),
    ("range_rate_", "doppler"),
    ("synthetic_", "synthetic"),
    ("run_all_", "orchestrator"),
]


def _categorise(filename: str) -> str:
    lower = filename.lower()
    for prefix, category in _CATEGORY_RULES:
        if lower.startswith(prefix):
            return category
    return "other"


def _scan_results() -> list[dict[str, Any]]:
    """Return metadata for every file in the results directory."""
    if not _RESULTS_DIR.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for entry in sorted(_RESULTS_DIR.iterdir()):
        if not entry.is_file():
            continue
        stat = entry.stat()
        ext = entry.suffix.lstrip(".").lower()
        items.append({
            "name": entry.name,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "type": ext,
            "category": _categorise(entry.name),
        })
    return items


def _read_csv(filename: str) -> dict[str, Any]:
    """Parse a CSV from results/ and return {headers, rows}."""
    filepath = (_RESULTS_DIR / filename).resolve()
    # Security: ensure the resolved path is still within results/
    if not str(filepath).startswith(str(_RESULTS_DIR.resolve())):
        raise ValueError("Path traversal blocked")
    if not filepath.is_file():
        raise FileNotFoundError(f"CSV file not found: {filename}")
    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {"headers": [], "rows": []}
    return {"headers": rows[0], "rows": rows[1:]}


# ---------------------------------------------------------------------------
# Experiments manifest
# ---------------------------------------------------------------------------
_EXPERIMENTS_PATH = _RESULTS_DIR / "run_all_experiments_summary.json"


def _load_experiments() -> list[dict[str, Any]]:
    if not _EXPERIMENTS_PATH.is_file():
        return []
    return json.loads(_EXPERIMENTS_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Test status (cached)
# ---------------------------------------------------------------------------
_test_status_cache: dict[str, Any] | None = None
_test_status_lock = threading.Lock()


def _run_test_status() -> dict[str, Any]:
    global _test_status_cache
    with _test_status_lock:
        if _test_status_cache is not None:
            return _test_status_cache
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s",
             "python_port/tests", "-t", "python_port"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_PROJECT_ROOT),
        )
        output = result.stderr or result.stdout
        # Parse unittest output for counts
        passed = result.returncode == 0
        # Try to extract "Ran N tests" from output
        test_count = 0
        for line in output.splitlines():
            if line.startswith("Ran "):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        test_count = int(parts[1])
                    except ValueError:
                        pass
        status = {
            "status": "pass" if passed else "fail",
            "returncode": result.returncode,
            "test_count": test_count,
            "output": output[-2000:],  # last 2 KB
            "cached": False,
        }
    except subprocess.TimeoutExpired:
        status = {
            "status": "timeout",
            "returncode": -1,
            "test_count": 0,
            "output": "Test run timed out after 120 s.",
            "cached": False,
        }
    except Exception as exc:
        status = {
            "status": "error",
            "returncode": -1,
            "test_count": 0,
            "output": str(exc),
            "cached": False,
        }

    with _test_status_lock:
        _test_status_cache = {**status, "cached": True}
    return status


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class LunarODHandler(BaseHTTPRequestHandler):
    """Routes requests to static files or JSON API endpoints."""

    server_version = "LunarOD-UI/1.0"

    # ----- helpers --------------------------------------------------------

    def _set_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # Client disconnected — silently ignore

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status=status)

    def _send_file(self, filepath: Path, base_dir: Path) -> None:
        """Serve a file with proper MIME type; block directory traversal."""
        filepath = filepath.resolve()
        if not str(filepath).startswith(str(base_dir.resolve())):
            self._send_error_json(403, "Forbidden")
            return
        if not filepath.is_file():
            self._send_error_json(404, f"Not found: {filepath.name}")
            return
        content_type, _ = mimetypes.guess_type(str(filepath))
        if content_type is None:
            content_type = "application/octet-stream"
        try:
            data = filepath.read_bytes()
        except OSError as exc:
            self._send_error_json(500, str(exc))
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self._set_cors()
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # Client disconnected

    # ----- routing --------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path).rstrip("/") or "/"

        # --- Static routes ------------------------------------------------
        if path == "/":
            self._send_file(_UI_DIR / "index.html", _UI_DIR)
            return

        if path.startswith("/ui/"):
            rel = path[len("/ui/"):]
            self._send_file(_UI_DIR / rel, _UI_DIR)
            return

        if path == "/assets/dunya-haritasi-dilsiz.png":
            self._send_file(_WORLD_MAP_PATH, _PROJECT_ROOT)
            return

        if path.startswith("/results/"):
            rel = path[len("/results/"):]
            self._send_file(_RESULTS_DIR / rel, _RESULTS_DIR)
            return

        # --- API routes ---------------------------------------------------
        if path == "/api/results":
            self._api_results()
            return

        if path == "/api/experiments":
            self._api_experiments()
            return

        if path == "/api/stations":
            self._api_stations()
            return

        if path == "/api/scenario-schema":
            self._api_scenario_schema()
            return

        if path.startswith("/api/csv/"):
            filename = path[len("/api/csv/"):]
            self._api_csv(filename)
            return

        if path == "/api/test-status":
            self._api_test_status()
            return

        # --- 404 ----------------------------------------------------------
        self._send_error_json(404, f"Not found: {path}")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path).rstrip("/")

        if path in ("/api/run-experiment", "/api/experiments/run"):
            self._api_run_experiment()
            return

        if path == "/api/scenario/validate":
            self._api_validate_scenario()
            return

        self._send_error_json(404, f"Not found: {path}")

    # ----- API implementations -------------------------------------------

    def _api_results(self) -> None:
        try:
            self._send_json(_scan_results())
        except Exception as exc:
            self._send_error_json(500, str(exc))

    def _api_experiments(self) -> None:
        try:
            self._send_json(_load_experiments())
        except Exception as exc:
            self._send_error_json(500, str(exc))

    def _api_stations(self) -> None:
        try:
            self._send_json(_load_stations())
        except Exception as exc:
            self._send_error_json(500, f"Failed to load stations: {exc}")

    def _api_scenario_schema(self) -> None:
        try:
            self._send_json(_load_schema())
        except Exception as exc:
            self._send_error_json(500, f"Failed to load schema: {exc}")

    def _api_csv(self, filename: str) -> None:
        try:
            self._send_json(_read_csv(filename))
        except FileNotFoundError as exc:
            self._send_error_json(404, str(exc))
        except ValueError as exc:
            self._send_error_json(403, str(exc))
        except Exception as exc:
            self._send_error_json(500, str(exc))

    def _api_test_status(self) -> None:
        try:
            self._send_json(_run_test_status())
        except Exception as exc:
            self._send_error_json(500, str(exc))

    def _api_run_experiment(self) -> None:
        # Read body
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            body: dict[str, Any] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_error_json(400, f"Invalid JSON body: {exc}")
            return

        experiment_id = body.get("experiment_id")
        dry_run = bool(body.get("dry_run", False))

        if not experiment_id:
            self._send_error_json(400, "Missing required field: experiment_id")
            return

        # Look up the experiment in the manifest
        experiments = _load_experiments()
        match = None
        for exp in experiments:
            if exp.get("experiment_id") == experiment_id:
                match = exp
                break
        if match is None:
            self._send_error_json(
                404, f"Unknown experiment_id: {experiment_id}")
            return

        script = match.get("script")
        if not script:
            self._send_error_json(
                500, f"Experiment {experiment_id!r} has no script defined")
            return

        cmd = [sys.executable, f"python_port/examples/{script}"]

        if dry_run:
            self._send_json({
                "status": "dry_run",
                "message": f"Would run: {' '.join(cmd)}",
                "command": cmd,
                "cwd": str(_PROJECT_ROOT),
            })
            return

        # Actually launch the experiment as a detached subprocess
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(_PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._send_json({
                "status": "started",
                "message": (
                    f"Experiment {experiment_id!r} started (PID {proc.pid}). "
                    f"Running: {' '.join(cmd)}"
                ),
                "pid": proc.pid,
            })
        except Exception as exc:
            self._send_error_json(500, f"Failed to start experiment: {exc}")

    def _api_validate_scenario(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            body: dict[str, Any] = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, ValueError) as exc:
            self._send_error_json(400, f"Invalid JSON body: {exc}")
            return

        try:
            from lunar_od.scenario_config import scenario_config_from_mapping

            payload = dict(body)
            if "duration_hours" in payload and "duration_h" not in payload:
                payload["duration_h"] = payload.pop("duration_hours")
            if "max_iterations" in payload and "max_iter" not in payload:
                payload["max_iter"] = payload.pop("max_iterations")
            if payload.get("bias_mode") == "none":
                payload["bias_mode"] = None

            config = scenario_config_from_mapping(payload)
            self._send_json({
                "valid": True,
                "normalized": config.to_dict(),
                "summary": (
                    f"{config.name}: {config.network} {config.measurement_type} "
                    f"{config.estimator_type}/{config.start_mode}"
                ),
            })
        except Exception as exc:
            self._send_json({
                "valid": False,
                "error": str(exc),
            }, status=200)

    # ----- logging --------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        # Coloured status code in the log
        sys.stderr.write(
            f"[{self.log_date_time_string()}] "
            f"{self.address_string()} — {format % args}\n"
        )


# ---------------------------------------------------------------------------
# Server entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lunar OD Web UI server")
    parser.add_argument(
        "--port", type=int, default=8420,
        help="TCP port to listen on (default: 8420)")
    args = parser.parse_args()

    # Ensure mimetypes knows about common extensions
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("application/json", ".json")
    mimetypes.add_type("image/png", ".png")
    mimetypes.add_type("image/svg+xml", ".svg")

    server = HTTPServer(("", args.port), LunarODHandler)

    banner = f"""
╔══════════════════════════════════════════════════╗
║         🌙  Lunar OD Web UI Server               ║
╠══════════════════════════════════════════════════╣
║  URL:     http://localhost:{args.port:<5}                 ║
║  UI dir:  {str(_UI_DIR):<39}║
║  Results: {str(_RESULTS_DIR):<39}║
╚══════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"  Serving on http://localhost:{args.port}")
    print("  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down…")
        server.shutdown()
        print("  Server stopped.")


if __name__ == "__main__":
    main()
