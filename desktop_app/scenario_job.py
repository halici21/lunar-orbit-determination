"""Run a persisted desktop scenario job in a separate Python process."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    job_path = args.job.resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("schema") != "lunar-od-desktop-job-v1":
        raise ValueError("Desteklenmeyen masaüstü koşu sözleşmesi.")
    from desktop_app.models.scenario_model import ScenarioModel
    from desktop_app.services.scenario_execution import execution_payload
    expected_config, options = execution_payload(ScenarioModel.from_dict(job["form"]))
    configured = dict(job["config"])
    folder = Path(configured.pop("output_dir")).resolve()
    if configured != expected_config or options != job["options"] or folder != job_path.parent:
        raise ValueError("Kaydedilen form, etkili yapılandırma ve çıktı klasörü uyuşmuyor.")
    from lunar_od import scenario_config_from_mapping
    config = scenario_config_from_mapping(job["config"])
    (folder / "effective_config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    print(f"Koşu {job['run_id']}: {config.estimator_type} / {config.measurement_type}, {config.duration_h:g} sa", flush=True)
    print(f"Seed: ölçüm={options['measurement_seed']}, başlangıç={options['cold_start_seed']}; {options['schedule_policy']}", flush=True)
    if args.dry_run:
        print("Yapılandırma doğrulandı; hesaplama yapılmadı.", flush=True)
        return 0
    from run_scenario_config import run_configured_scenario
    from lunar_od import write_scenario_summary_csv, plot_scenario_comparison
    from desktop_app.services.scenario_artifacts import write_console_bundle, publish_completion
    result_path = None
    def save_result(result, arcs, times, truth, epoch):
        nonlocal result_path
        result_path = write_console_bundle(result, arcs, times, truth, epoch,
            config=config, job_path=job_path, options=options, project_root=ROOT)
    result = run_configured_scenario(config, measurement_seed=options["measurement_seed"],
        cold_start_seed=options["cold_start_seed"], manifest_dir=folder,
        desktop_options=options, artifact_callback=save_result)
    summary = write_scenario_summary_csv([result], folder / f"{config.name}_summary.csv")
    figure = plot_scenario_comparison([result], folder / f"{config.name}_comparison.png", title=config.name)
    if result_path is None:
        raise RuntimeError("Çözücü sonuç paketi üretmedi.")
    publish_completion(folder, job["run_id"], result_path, [summary, figure, result_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
