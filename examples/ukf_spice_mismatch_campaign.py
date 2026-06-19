"""Run a reproducible UKF SPICE ephemeris mismatch sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import load_scenario_config_json, plot_scenario_comparison, write_scenario_summary_csv
from run_scenario_config import run_configured_scenario


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="UKF scenario JSON path.")
    parser.add_argument(
        "--earth-position-bias-m",
        default="0,100,1000",
        help="Comma-separated Earth ephemeris position-bias magnitudes along J2000 X.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_scenario_config_json(args.config)
    if config.estimator_type != "ukf":
        raise ValueError("SPICE mismatch campaign requires estimator_type='ukf'.")
    magnitudes = tuple(float(value) for value in args.earth_position_bias_m.split(","))
    output_dir = Path(config.output_dir)
    csv_path = output_dir / f"{config.name}_spice_mismatch.csv"
    png_path = output_dir / f"{config.name}_spice_mismatch.png"
    if args.dry_run:
        print(f"Cases: {magnitudes}")
        print(f"Would write {csv_path}")
        print(f"Would write {png_path}")
        return 0

    scenarios = []
    for magnitude in magnitudes:
        scenario = run_configured_scenario(
            config,
            earth_position_bias_m=(magnitude, 0.0, 0.0),
        )
        scenarios.append(
            type(scenario)(
                label=f"{config.name} Earth dX={magnitude:g} m",
                measurement_type=scenario.measurement_type,
                start_mode=scenario.start_mode,
                arc_results=scenario.arc_results,
                estimator_type=scenario.estimator_type,
                range_rate_physics=scenario.range_rate_physics,
                count_interval_s=scenario.count_interval_s,
            )
        )

    write_scenario_summary_csv(scenarios, csv_path)
    plot_scenario_comparison(scenarios, png_path, title=f"{config.name} SPICE mismatch")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
