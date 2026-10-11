"""Recorded mission geometry and display-only orbital summaries; no propagation."""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def station_catalog():
    data = json.loads((ROOT / "fixtures/station_config.json").read_text(encoding="utf-8"))
    seen = {}
    for section in data.values():
        if isinstance(section, dict):
            for station in section.get("stations", []):
                name = station["name"]
                seen.setdefault(name, {key: station[key] for key in ("name", "lat_deg", "lon_deg", "alt_m")})
    return tuple(seen.values())


def recorded_geometry(arcs, config):
    """Serialize actual station identity and observed epochs, not inferred visibility."""
    stations, windows = None, []
    station_column = {"position": 4, "range_rate": 5, "two_way_range": 2}
    for arc in arcs:
        geometry = getattr(arc, "pass_geo", None)
        if geometry is None:
            return None  # Legacy/test producers do not have station provenance.
        current = [{"station_id": index, "name": s.name, "lat_deg": float(s.lat_deg),
                    "lon_deg": float(s.lon_deg), "alt_m": float(s.alt_m)}
                   for index, s in enumerate(geometry.stations)]
        if stations is not None and stations != current:
            raise ValueError("Yaylar arasında istasyon kimliği değişti.")
        stations = current
        column = station_column[geometry.measurement_type]
        grouped = {}
        for row in arc.obs_data:
            sid, t_s = float(row[column]) - 1, float(row[0])
            if not sid.is_integer() or not 0 <= sid < len(stations) or not math.isfinite(t_s) or t_s < 0:
                raise ValueError("Ölçüm istasyon kimliği veya epoku geçersiz.")
            grouped.setdefault(int(sid), set()).add(t_s)
        for sid, times in grouped.items():
            run = []
            def flush():
                if run:
                    windows.append({"station_id": sid, "arc_id": int(arc.arc_id),
                                    "start_s": run[0], "end_s": run[-1], "samples": len(run)})
            for t_s in sorted(times):
                if run and t_s - run[-1] > float(config.sample_step_s) * 1.01:
                    flush()
                    run = []
                run.append(t_s)
            flush()
    if stations is None:
        return None
    return {"schema": "lunar-od-mission-geometry-v1", "scope": "recorded_measurement_samples",
            "sample_step_s": float(config.sample_step_s), "stations": stations, "windows": windows}


def validate_geometry(data):
    if not isinstance(data, dict) or data.get("schema") != "lunar-od-mission-geometry-v1" or data.get("scope") != "recorded_measurement_samples":
        raise ValueError("Görev geometrisi şeması veya kapsamı geçersiz.")
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    step = data.get("sample_step_s")
    if not finite(step) or step <= 0:
        raise ValueError("Geçerli görev örnek adımı gerekli.")
    stations, windows = data.get("stations"), data.get("windows")
    if not isinstance(stations, list) or not 1 <= len(stations) <= 100 or not isinstance(windows, list) or len(windows) > 250000:
        raise ValueError("Görev istasyon/pencere sayısı geçersiz.")
    ids, names = set(), set()
    for s in stations:
        if (not isinstance(s, dict) or type(s.get("station_id")) is not int or s["station_id"] < 0
                or not isinstance(s.get("name"), str) or not s["name"] or s["name"] in names
                or s["station_id"] in ids or not finite(s.get("lat_deg")) or not -90 <= s["lat_deg"] <= 90
                or not finite(s.get("lon_deg")) or not -180 <= s["lon_deg"] <= 180 or not finite(s.get("alt_m"))):
            raise ValueError("Görev istasyon koordinatları/kimliği geçersiz.")
        ids.add(s["station_id"])
        names.add(s["name"])
    for w in windows:
        if (not isinstance(w, dict) or type(w.get("station_id")) is not int or w["station_id"] not in ids
                or type(w.get("arc_id")) is not int or w["arc_id"] < 0
                or not finite(w.get("start_s")) or not finite(w.get("end_s"))
                or not 0 <= w["start_s"] <= w["end_s"]
                or type(w.get("samples")) is not int or w["samples"] < 1):
            raise ValueError("Görev ölçüm aralığı geçersiz.")
    return data


def orbit_summary(state, *, mu=None):
    """Osculating two-body descriptors in the stored frame, never a new orbit."""
    r = [state[k] for k in ("x_m", "y_m", "z_m")]
    v = [state[k] for k in ("vx_mps", "vy_mps", "vz_mps")]
    radius = math.sqrt(sum(x*x for x in r))
    h = [r[1]*v[2]-r[2]*v[1], r[2]*v[0]-r[0]*v[2], r[0]*v[1]-r[1]*v[0]]
    norm_h = math.sqrt(sum(x*x for x in h))
    result = {"radius_km": radius/1000, "altitude_km": radius/1000-1737.4,
              "inclination_deg": math.degrees(math.acos(max(-1., min(1., h[2]/norm_h)))) if norm_h else None,
              "a_km": None, "e": None, "period_h": None}
    if radius and type(mu) in (int, float) and math.isfinite(mu) and mu > 0:
        v2, rv = sum(x*x for x in v), sum(x*y for x,y in zip(r,v))
        ev = [((v2-mu/radius)*x-rv*y)/mu for x,y in zip(r,v)]
        result["e"] = math.sqrt(sum(x*x for x in ev))
        inv_a = 2/radius-v2/mu
        if inv_a > 0:
            a = 1/inv_a
            result["a_km"] = a/1000
            result["period_h"] = 2*math.pi*math.sqrt(a**3/mu)/3600
    return result
