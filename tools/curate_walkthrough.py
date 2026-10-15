"""Copy verified real recording media and result evidence into the public docs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curate(source):
    report_path = source / "recording.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    video = source / "desktop-walkthrough.mp4"
    if sha(video) != report["video_sha256"] or report["duration_s"] != 60:
        raise ValueError("Capture identity/duration mismatch")
    for name, expected in report["source"]["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Recorded producer changed: {name}")
    destination = ROOT / "docs/media/desktop"
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(video, destination / video.name)
    shutil.copy2(source / "frame-160.png", destination / "preview.png")
    shutil.copy2(source / "frame-320.png", destination / "comparison.png")
    data = destination / "data"
    data.mkdir()
    sources = [Path(name) for name in report["result_sources"]]
    for path in sources:
        if sha(path) != report["result_sha256"][path.name]:
            raise ValueError("Recorded result changed")
        prefix = path.name.removesuffix("_ukf_updates.csv")
        for artifact in path.parent.glob(prefix + "*"):
            shutil.copy2(artifact, data / artifact.name)
    report["original_capture_record_sha256"] = sha(report_path)
    report["result_sources"] = ["data/" + path.name for path in sources]
    report["artifact_sha256"] = {p.relative_to(destination).as_posix(): sha(p)
                                for p in sorted(destination.rglob("*")) if p.is_file()}
    (destination / "recording.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Published {video.stat().st_size:,}-byte real recording and {len(sources)} source runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    curate(parser.parse_args().source)
