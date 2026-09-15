"""PHASE 17-R1M - multi-arc K_SRP identifiability and covariance diagnostic.

ANALYSIS SPACE ONLY.  No production file is modified.

Campaign schedule declared BEFORE any information result was inspected (s29),
and anchored on the measured DSN pass structure rather than on round numbers
(see CAMPAIGNS below for the visibility table it was read from):

    window length          0.5 lunar orbit

    Two different spans can both be called "elapsed", so both are tabulated and
    only the first is ever reported as `elapsed_orbits`:

        SPAN   = last window end - FIRST window start  (the dynamics actually
                 linking the arcs; this is what `elapsed_orbits` means)
        REACH  = last window end - CAMPAIGN EPOCH      (the outer bound of the
                 propagation, always SPAN + 0.50 here)

    case  arcs  starts                            SPAN    REACH
    M1    2     0.50, 2.50                        2.50    3.00
    M2    3     0.50, 2.50, 5.50                  5.50    6.00
    M3    5     0.50, 1.50, 2.50, 3.50, 5.50      5.50    6.00

    M2 and M3 happen to make the two agree with the round number 5.5 and M1
    does not, which is exactly how an earlier draft of this docstring came to
    label M1 as "3.0 orbits elapsed" while the code computed 2.50.

An earlier draft used evenly spaced starts at 0/2/4/6/8 orbits. That was
discarded, not tuned: Canberra stops seeing the spacecraft after ~4.56 orbits,
so the 6.0 and 8.0 windows contained no data at all and M2/M3 silently
collapsed onto the same result. The replacement anchors are the pass starts
themselves, with every complex enabled so the handover happens naturally.

Each configuration is paired with a CONTIGUOUS control carrying the same total
observation time (0.5*M orbits) but no elapsed gap, which is the only way to
separate "more data" from "more elapsed dynamics" (s30).

Two architectures are compared on identical observation windows (s37):

    MODEL A  pure multi-arc     6M + 1 unknowns, one local state per arc
    MODEL B  continuity-linked   6 + 1 unknowns, one state at the campaign
                                 epoch; the STM and dx/dK accumulate through
                                 the unobserved gaps because the trajectory is
                                 never re-initialised

Model B is obtained by building ONE long arc and masking observations to the
windows, which is what makes its STM and S_K genuinely continuous rather than
reset per arc.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase17_r1m_core import (  # noqa: E402
    K_TRUTH, SCALE_K, build_range_arc, campaign_epoch, information_matrix,
    orthogonal_decomposition, scale_matrix, schur_conditional, spectrum,
    whitened,
)

WINDOW_ORBITS = 0.5

#: DSN visibility on this arc, measured BEFORE any K result was inspected:
#:
#:   Canberra  0.48-1.09, 1.49-2.09, 2.49-3.09, 3.49-4.09, 4.49-4.56 orbits
#:   Madrid    4.93-5.09, 5.49-6.10, 6.50-7.10, 7.50-8.09
#:   Goldstone 8.55-9.00
#:
#: A single complex therefore cannot cover a long elapsed span -- Earth
#: rotation hands the spacecraft over, exactly as a real DSN schedule does.
#: Arc windows are anchored at the START OF A VISIBLE PASS and all three
#: complexes are enabled, so each window uses whichever complex actually sees
#: the spacecraft. The anchors come from the visibility structure alone; they
#: were not tuned against any K singular value (s29).
CAMPAIGNS = {
    "M1": [0.50, 2.50],
    "M2": [0.50, 2.50, 5.50],
    "M3": [0.50, 1.50, 2.50, 3.50, 5.50],
}
#: Both architectures must see the same complexes for the comparison in s37 to
#: mean anything; auto-selecting "the best station per arc" would silently give
#: Model A and Model B different data.
ALL_STATIONS = ("Goldstone DSN", "Madrid DSN", "Canberra DSN")
ARTIFACTS = Path(r"C:/Users/erayh/Documents/Python/Grad/python_port_phase17r0/artifacts")
OUT: dict = {}


def hdr(t):
    print()
    print("=" * 86)
    print(t)
    print("=" * 86)


def write_csv(name, fieldnames, rows):
    path = ARTIFACTS / name
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print("  wrote %s (%d rows)" % (name, len(rows)))


# ======================================================================
# MODEL A - pure multi-arc: independent local states + one shared global K
# ======================================================================
def model_a_information(arcs) -> dict:
    """Assemble the (6M+1) global information matrix, block-diagonal in state.

    Row block for arc i:   [ 0 ... A_i ... 0 | b_i ]

    The shared K column is the LAST unknown, so the same schur_conditional /
    spectrum helpers apply unchanged.
    """
    m = len(arcs)
    n_unknown = 6 * m + 1
    blocks, weights = [], []
    for i, arc in enumerate(arcs):
        n = arc.n_obs
        row = np.zeros((n, n_unknown))
        row[:, 6 * i:6 * (i + 1)] = arc.h_x0
        row[:, -1] = arc.h_k
        blocks.append(row)
        weights.append(arc.w)
    h_global = np.vstack(blocks)
    w_global = np.concatenate(weights)
    scale = np.diag([1e6, 1e6, 1e6, 1e3, 1e3, 1e3] * m + [SCALE_K])
    info = information_matrix(h_global, w_global, scale)

    # whitened orthogonal decomposition against ALL local state columns
    a_x, b_k = whitened(h_global[:, :-1], h_global[:, -1], w_global)
    dec = orthogonal_decomposition(a_x, b_k)
    sp = spectrum(info)
    return dict(
        info=info, spectrum=sp, decomposition=dec,
        n_unknown=n_unknown, n_obs=int(h_global.shape[0]),
        conditional_k=schur_conditional(info) / SCALE_K ** 2,
    )


def model_b_information(long_arc, windows_s) -> dict:
    """One continuous state at the campaign epoch; observations masked to windows.

    Because the trajectory is never re-initialised, Phi and dx/dK accumulate
    through the unobserved gaps -- which is precisely the mechanism this model
    exists to test.
    """
    t_obs = np.asarray(long_arc.obs[:, 0], float)
    keep = np.zeros(t_obs.shape, dtype=bool)
    for t0, t1 in windows_s:
        keep |= (t_obs >= t0 - 1e-6) & (t_obs <= t1 + 1e-6)
    h_x0 = long_arc.h_x0[keep]
    h_k = long_arc.h_k[keep]
    w = long_arc.w[keep]
    h_full = np.hstack([h_x0, h_k[:, None]])
    info = information_matrix(h_full, w, scale_matrix())
    a_x, b_k = whitened(h_x0, h_k, w)
    dec = orthogonal_decomposition(a_x, b_k)
    return dict(
        info=info, spectrum=spectrum(info), decomposition=dec,
        n_unknown=7, n_obs=int(keep.sum()),
        conditional_k=schur_conditional(info) / SCALE_K ** 2,
    )


def main() -> None:
    from lunar_od.spice_loader import load_spice_kernels

    load_spice_kernels(None, clear=True)
    et0, t_orbit = campaign_epoch()
    hdr("PHASE 17-R1M - MULTI-ARC K_SRP IDENTIFIABILITY")
    print("  campaign epoch %.5f, orbit period %.4f s" % (et0, t_orbit))
    print("  window length %.2f orbit; schedules declared before inspection"
          % WINDOW_ORBITS)

    window_s = WINDOW_ORBITS * t_orbit

    # ---------------- single-arc reference (G0..G3) -------------------
    hdr("R1M-A / R1M-C  SINGLE CONTIGUOUS ARCS")
    single_rows = []
    for label, orbits in (("G0", 1.3), ("G1", 2.0), ("G2", 3.0), ("G3", 5.0)):
        arc = build_range_arc(0.0, orbits * t_orbit, label=label)
        a_x, b_k = whitened(arc.h_x0, arc.h_k, arc.w)
        dec = orthogonal_decomposition(a_x, b_k)
        info = information_matrix(np.hstack([arc.h_x0, arc.h_k[:, None]]),
                                  arc.w, scale_matrix())
        sp = spectrum(info)
        row = dict(case=label, arc_orbits=orbits, observations=arc.n_obs,
                   elapsed_s=round(arc.elapsed_s, 3),
                   k_column_norm=dec["k_column_norm"],
                   projected_norm=dec["projected_norm"],
                   orthogonal_norm=dec["orthogonal_norm"],
                   orthogonal_fraction=dec["orthogonal_fraction"],
                   raw_i_kk=dec["i_kk"],
                   conditional_k_information=dec["i_k_given_x"],
                   smallest_scaled_singular_value=sp["smallest"],
                   scaled_condition=sp["condition"],
                   rank=sp["rank_default"],
                   weakest_mode_k_component=sp["weakest_k_component"])
        single_rows.append(row)
        print("  %-3s %.1f orb  n=%3d  I_KK=%.6e  I_K|x=%.6e  f_perp=%.6f  "
              "rank %d/7  weakK=%.12f"
              % (label, orbits, arc.n_obs, dec["i_kk"], dec["i_k_given_x"],
                 dec["orthogonal_fraction"], sp["rank_default"],
                 sp["weakest_k_component"]))
    write_csv("r1m_single_arc_projection.csv", list(single_rows[0].keys()),
              single_rows)
    OUT["single_arc"] = single_rows

    # ---------------- multi-arc campaigns -----------------------------
    hdr("R1M-D / R1M-E  MULTI-ARC (MODEL A) vs CONTINUITY-LINKED (MODEL B)")
    multi_rows, sum_rule_rows, compare_rows = [], [], []
    for name, starts in CAMPAIGNS.items():
        m = len(starts)
        windows = [(s * t_orbit, s * t_orbit + window_s) for s in starts]
        elapsed = windows[-1][1] - windows[0][0]
        assert windows[-1][1] <= 9.0 * t_orbit
        obs_time = m * window_s

        arcs = [build_range_arc(w0, window_s, label="%s_a%d" % (name, i),
                                station_filter=ALL_STATIONS)
                for i, (w0, _) in enumerate(windows)]
        a = model_a_information(arcs)

        long_arc = build_range_arc(0.0, windows[-1][1], label="%s_long" % name,
                                   station_filter=ALL_STATIONS)
        b = model_b_information(long_arc, windows)

        # Schur sum rule for MODEL A (s32): independent local states must make
        # the global conditional K information the SUM of the per-arc ones.
        per_arc = []
        for arc in arcs:
            ax, bk = whitened(arc.h_x0, arc.h_k, arc.w)
            per_arc.append(orthogonal_decomposition(ax, bk)["i_k_given_x"])
        sum_rule = float(np.sum(per_arc))
        rel = abs(a["conditional_k"] - sum_rule) / max(sum_rule, 1e-300)
        sum_rule_rows.append(dict(case=name, arcs=m,
                                  sum_of_per_arc=sum_rule,
                                  global_model_a=a["conditional_k"],
                                  relative_difference=rel,
                                  passes=bool(rel < 1e-6)))
        print("\n  %s  arcs=%d  elapsed=%.2f orb  obs_time=%.2f orb" %
              (name, m, elapsed / t_orbit, obs_time / t_orbit))
        print("    MODEL A  unknowns=%3d n_obs=%3d rank=%d/%d I_K|x=%.6e "
              "f_perp=%.6f weakK=%.9f"
              % (a["n_unknown"], a["n_obs"], a["spectrum"]["rank_default"],
                 a["n_unknown"], a["conditional_k"],
                 a["decomposition"]["orthogonal_fraction"],
                 a["spectrum"]["weakest_k_component"]))
        print("    MODEL B  unknowns=%3d n_obs=%3d rank=%d/7 I_K|x=%.6e "
              "f_perp=%.6f weakK=%.9f"
              % (b["n_unknown"], b["n_obs"], b["spectrum"]["rank_default"],
                 b["conditional_k"], b["decomposition"]["orthogonal_fraction"],
                 b["spectrum"]["weakest_k_component"]))
        print("    Schur sum rule: sum(per-arc)=%.6e  global=%.6e  rel=%.2e  %s"
              % (sum_rule, a["conditional_k"], rel,
                 "PASS" if rel < 1e-6 else "FAIL"))

        for model, res in (("A_independent_arcs", a), ("B_continuity_linked", b)):
            multi_rows.append(dict(
                case=name, model=model, arcs=m,
                elapsed_orbits=round(elapsed / t_orbit, 4),
                observation_orbits=round(obs_time / t_orbit, 4),
                elapsed_s=round(elapsed, 3),
                observations=res["n_obs"], unknowns=res["n_unknown"],
                rank=res["spectrum"]["rank_default"],
                smallest_scaled_singular_value=res["spectrum"]["smallest"],
                scaled_condition=res["spectrum"]["condition"],
                weakest_mode_k_component=res["spectrum"]["weakest_k_component"],
                raw_i_kk=res["decomposition"]["i_kk"],
                conditional_k_information=res["conditional_k"],
                orthogonal_fraction=res["decomposition"]["orthogonal_fraction"]))

        compare_rows.append(dict(
            case=name, arcs=m,
            model_a_conditional_k=a["conditional_k"],
            model_b_conditional_k=b["conditional_k"],
            gain_b_over_a=b["conditional_k"] / max(a["conditional_k"], 1e-300),
            model_a_f_perp=a["decomposition"]["orthogonal_fraction"],
            model_b_f_perp=b["decomposition"]["orthogonal_fraction"],
            model_a_weak_k=a["spectrum"]["weakest_k_component"],
            model_b_weak_k=b["spectrum"]["weakest_k_component"],
            model_a_rank_full=bool(a["spectrum"]["rank_default"] == a["n_unknown"]),
            model_b_rank_full=bool(b["spectrum"]["rank_default"] == 7)))

    write_csv("r1m_multi_arc_information.csv", list(multi_rows[0].keys()),
              multi_rows)
    write_csv("r1m_schur_sum_rule.csv", list(sum_rule_rows[0].keys()),
              sum_rule_rows)
    write_csv("r1m_pure_vs_linked_multi_arc.csv", list(compare_rows[0].keys()),
              compare_rows)
    OUT["multi_arc"] = multi_rows
    OUT["schur_sum_rule"] = sum_rule_rows
    OUT["pure_vs_linked"] = compare_rows

    # ---------------- contiguous controls (s30) -----------------------
    hdr("s30  ELAPSED-TIME CONTROL: same observation time, no gaps")
    control_rows = []
    for name, starts in CAMPAIGNS.items():
        m = len(starts)
        contiguous_orbits = m * WINDOW_ORBITS
        arc = build_range_arc(0.0, contiguous_orbits * t_orbit,
                              label="%s_contig" % name,
                              station_filter=ALL_STATIONS)
        a_x, b_k = whitened(arc.h_x0, arc.h_k, arc.w)
        dec = orthogonal_decomposition(a_x, b_k)
        info = information_matrix(np.hstack([arc.h_x0, arc.h_k[:, None]]),
                                  arc.w, scale_matrix())
        sp = spectrum(info)
        linked = [r for r in multi_rows
                  if r["case"] == name and r["model"] == "B_continuity_linked"][0]
        control_rows.append(dict(
            case=name, arcs=m, contiguous_orbits=contiguous_orbits,
            observations=arc.n_obs,
            contiguous_conditional_k=dec["i_k_given_x"],
            linked_conditional_k=linked["conditional_k_information"],
            linked_over_contiguous=linked["conditional_k_information"]
            / max(dec["i_k_given_x"], 1e-300),
            contiguous_f_perp=dec["orthogonal_fraction"],
            linked_f_perp=linked["orthogonal_fraction"],
            contiguous_weak_k=sp["weakest_k_component"]))
        print("  %s  contiguous %.1f orb (n=%d): I_K|x=%.6e f_perp=%.6f"
              % (name, contiguous_orbits, arc.n_obs, dec["i_k_given_x"],
                 dec["orthogonal_fraction"]))
        print("      linked  %.2f orb elapsed (n=%d): I_K|x=%.6e f_perp=%.6f  "
              "ratio %.3fx"
              % (linked["elapsed_orbits"], linked["observations"],
                 linked["conditional_k_information"], linked["orthogonal_fraction"],
                 control_rows[-1]["linked_over_contiguous"]))
    write_csv("r1m_elapsed_time_control.csv", list(control_rows[0].keys()),
              control_rows)
    OUT["elapsed_control"] = control_rows

    (ARTIFACTS / "r1m_multi_arc_results.json").write_text(
        json.dumps(OUT, indent=2, default=float))
    print("\n  wrote r1m_multi_arc_results.json")


if __name__ == "__main__":
    main()
