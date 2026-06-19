"""Validate and normalize Lunar OD JSON scenario configs.

Run from the project root:

    python python_port/examples/scenario_config_cli.py config.json
    python python_port/examples/scenario_config_cli.py --schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lunar_od import (  # noqa: E402
    load_scenario_config_json,
    scenario_config_schema,
    scenario_config_summary,
    write_normalized_scenario_config,
)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.schema:
        print(json.dumps(scenario_config_schema(), indent=2))
        return 0
    if args.config is None:
        raise SystemExit("Provide a config JSON path, or use --schema.")

    config = load_scenario_config_json(args.config)
    print(scenario_config_summary(config))
    if args.write_normalized:
        output_path = write_normalized_scenario_config(config, args.write_normalized)
        print(f"Wrote {output_path}")
    return 0


def _parse_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Lunar OD JSON scenario config.")
    parser.add_argument("config", nargs="?", help="Scenario config JSON path.")
    parser.add_argument("--schema", action="store_true", help="Print the compact JSON schema and exit.")
    parser.add_argument("--write-normalized", help="Write normalized config JSON to this path.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
