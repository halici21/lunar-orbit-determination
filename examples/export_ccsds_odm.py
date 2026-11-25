"""Export an estimated trajectory and covariance as CCSDS OEM and OPM (KVN, 502.0-B-3).

python examples/export_ccsds_odm.py --demo --output-dir outputs/ccsds_odm
python examples/export_ccsds_odm.py --input trajectory.npz --object-name LRO --object-id 2009-031A

``--input`` takes an ``.npz`` with ``epochs_et_s`` (N; SPICE ET = TDB s past
J2000), ``states_m`` (N x 6, Moon-centred J2000, m and m/s) and optionally
``covariances_m`` (N x 6 x 6, or N x n x n for an augmented state, whose leading
6x6 block is exported). ``--demo`` writes a closed-form circular lunar orbit
with a constant illustrative covariance; it needs no kernels and is not an
estimation result. The OPM holds the first epoch's state and covariance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lunar_od.ccsds_odm import OemCovariance, OemSegment, write_oem, write_opm
from lunar_od.constants import MU_MOON_M3S2, R_MOON_M


def demo_trajectory(n: int = 61, step_s: float = 60.0, et0_s: float = 8.0e8):
    """Circular 100 km polar lunar orbit (two-body, closed form) and a fixed covariance."""
    radius = R_MOON_M + 100.0e3
    rate = np.sqrt(MU_MOON_M3S2 / radius**3)
    t = step_s * np.arange(n)
    c, s = np.cos(rate * t), np.sin(rate * t)
    speed = rate * radius
    states = np.column_stack([radius * c, np.zeros(n), radius * s,
                              -speed * s, np.zeros(n), speed * c])
    cov = np.diag([50.0, 50.0, 50.0, 0.05, 0.05, 0.05]) ** 2
    return et0_s + t, states, np.repeat(cov[None], n, axis=0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help=".npz with epochs_et_s, states_m[, covariances_m]")
    source.add_argument("--demo", action="store_true", help="write a closed-form demo orbit")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ccsds_odm"))
    parser.add_argument("--object-name", default="LUNAR_OD_DEMO")
    parser.add_argument("--object-id", default="DEMO")
    parser.add_argument("--originator", default="LUNAR_OD")
    parser.add_argument("--covariance-stride", type=int, default=10,
                        help="write every k-th covariance in the OEM (0: none)")
    parser.add_argument("--creation-date", default=None,
                        help="UTC CREATION_DATE (default now); fix it for byte-reproducible files")
    args = parser.parse_args(argv)

    if args.demo:
        epochs, states, covs = demo_trajectory()
    else:
        data = np.load(args.input)
        epochs, states = data["epochs_et_s"], data["states_m"][:, :6]
        covs = data["covariances_m"][:, :6, :6] if "covariances_m" in data else None

    stride = max(0, args.covariance_stride)
    covariances = () if covs is None or stride == 0 else tuple(
        OemCovariance(float(epochs[k]), covs[k]) for k in range(0, len(epochs), stride))
    common = dict(originator=args.originator, creation_date=args.creation_date)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oem = write_oem(args.output_dir / f"{args.object_id}.oem",
                    OemSegment(epochs, states, args.object_name, args.object_id,
                               covariances=covariances),
                    **common)
    opm = write_opm(args.output_dir / f"{args.object_id}.opm", float(epochs[0]), states[0],
                    object_name=args.object_name, object_id=args.object_id,
                    covariance_m=None if covs is None else covs[0], **common)
    print(f"Wrote {oem} ({len(epochs)} states, {len(covariances)} covariances)")
    print(f"Wrote {opm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
