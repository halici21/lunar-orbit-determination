"""PHASE 17-GEO - stage artifacts (s61) and figures (s60).

ANALYSIS SPACE ONLY.  Reads artifacts/phase17_geo_results.json (+ fixture
attribution, phase robustness) and the cached design matrices.  Traceability
(R1 invariant): every figure is drawn from its own CSV written alongside it, and
before anything is plotted each case's f_perp is RECOMPUTED from its cached
design matrix and required to equal the artifact value.

Every figure keeps information DIRECTION (f_perp, theta_K) and information
MAGNITUDE (sigma_K/K, |b_K|, I_K|x) in separate panels.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent
ART = ROOT / "artifacts"
CACHE = ROOT / "results" / "phase17_geo_cache"
FIG = ROOT / "docs" / "figures" / "phase17_geo"

OKABE = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermil="#D55E00",
             purple="#CC79A7", sky="#56B4E9", yellow="#F0E442", black="#000000")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def row(r: dict) -> dict:
    return dict(
        id=r["id"], stage=r["stage"], beta_mid_deg=r["beta_mid_deg"],
        earth_view_mid_deg=r["earth_view_mid_deg"], inc_deg=r["inc_deg"],
        alt_mean_km=r["alt_mean_km"], period_s=r["period_s"], revolutions=r["revolutions"],
        duration_h=r["duration_h"], eclipse_fraction=r["eclipse_fraction"],
        n_eclipses=r["n_eclipses"], mean_eclipse_duration_s=r["mean_eclipse_duration_s"],
        srp_rtn_share_R=r["srp_rtn_share_R"], srp_rtn_share_T=r["srp_rtn_share_T"],
        srp_rtn_share_N=r["srp_rtn_share_N"], n_passes=r["n_passes"],
        n_obs=r["full_n_obs"], f_perp=r["full_f_perp"], theta_k_deg=r["full_theta_k_deg"],
        conditional_k_information=r["full_conditional_k_information"],
        k_column_norm=r["full_k_column_norm"], sigma_k_frac=r["full_sigma_k_frac"],
        max_abs_k_state_correlation=r["full_max_abs_k_state_correlation"],
        weakest_mode_k_component=r["full_weakest_mode_k_component"],
        smallest_design_sv=r["full_smallest_design_sv"],
        matched_n_obs=r.get("matched_n_obs", ""), matched_f_perp=r.get("matched_f_perp", ""),
        matched_sigma_k_frac=r.get("matched_sigma_k_frac", ""),
        delta_f_perp_full=r.get("delta_f_perp_full", ""), class_full=r.get("class_full", ""),
        class_matched=r.get("class_matched", ""),
        beta_target=r.get("beta_target", ""), normalization=r.get("normalization", ""),
        k_srp=r["k_srp"], shadow_model=r["shadow_model"], stations=r["stations"],
        epoch_offset_days=(r["epoch_et"] - RESULTS_EPOCH0) / 86400.0,
    )


def main() -> None:
    global RESULTS_EPOCH0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import phase17_geo_core as G

    data = json.loads((ART / "phase17_geo_results.json").read_text())
    res = [r for r in data["results"] if r.get("valid") and "full_f_perp" in r]
    by = {r["id"]: r for r in res}
    RESULTS_EPOCH0 = by["G0_consistent"]["epoch_et"]

    # ---- traceability: recompute every f_perp from its cached design ------
    worst = 0.0
    for r in res:
        z = np.load(CACHE / f"{r['id']}.npz")
        f = G.information_metrics(z["h_x0"], z["h_k"], z["w"], k_truth=r["k_srp"])["f_perp"]
        worst = max(worst, abs(f - r["full_f_perp"]))
    print(f"traceability: max |f_perp(recomputed) - f_perp(artifact)| = {worst:.3e} over {len(res)} cases")
    if worst > 1e-12:
        raise SystemExit("artifact f_perp does not recompute from cached design -- STOP")

    # ---- s61 stage artifacts ---------------------------------------------
    rows = [row(r) for r in res]
    stage_files = {
        "phase17_geo_beta_eclipse.csv": ("G1", "CF"),
        "phase17_geo_altitude.csv": ("G2",),
        "phase17_geo_inclination.csv": ("G3",),
        "phase17_geo_tracking_geometry.csv": ("G4",),
        "phase17_geo_k_magnitude_control.csv": ("G5",),
    }
    for name, stages in stage_files.items():
        sel = [x for x in rows if x["stage"] in stages]
        if name == "phase17_geo_k_magnitude_control.csv":
            sel = [x for x in rows if x["id"] == "G0_consistent"] + sel
        write_csv(ART / name, sel)
    write_csv(ART / "phase17_geo_matched_count.csv",
              [dict(id=x["id"], stage=x["stage"], n_obs=x["n_obs"], f_perp_full=x["f_perp"],
                    matched_n_obs=x["matched_n_obs"], f_perp_matched=x["matched_f_perp"],
                    sigma_k_frac_full=x["sigma_k_frac"], sigma_k_frac_matched=x["matched_sigma_k_frac"],
                    class_full=x["class_full"], class_matched=x["class_matched"]) for x in rows])
    (ART / "phase17_geo_baseline.json").write_text(json.dumps(
        dict(published=by["G0_published"], consistent=by["G0_consistent"],
             fixture_attribution=json.loads((ART / "phase17_geo_fixture_attribution.json").read_text())),
        indent=1, default=float))
    rep_ids = ("G0_consistent", "G1_b00", "G1_b71", "G1_b88", "G2_alt30_b20_fixD",
               "G2_alt1000_b20_fixD", "G3_i26.7_bcan", "G4_quarter_b20", "CF_noshadow_G0_consistent")
    (ART / "phase17_geo_representative_cases.json").write_text(json.dumps(
        {i: by[i] for i in rep_ids}, indent=1, default=float))

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    f_ref = by["G0_consistent"]["full_f_perp"]
    beta_b = json.loads((ART / "phase17_geo_predeclared_grid.json").read_text())["full_sun_boundary_beta_deg"]

    # ---- F1 beta: direction vs magnitude ---------------------------------
    g1 = sorted([x for x in rows if x["stage"] == "G1"], key=lambda x: x["beta_mid_deg"])
    gq = sorted([x for x in rows if x["id"].startswith("G4_quarter")], key=lambda x: x["beta_mid_deg"])
    write_csv(FIG / "f1_beta.csv", g1 + gq)
    d = read_csv(FIG / "f1_beta.csv")
    nm = [x for x in d if x["stage"] == "G1"]
    qt = [x for x in d if x["stage"] == "G4"]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    panels = (("f_perp", "f_perp (-)", False, "A  Direction: f_perp"),
              ("theta_k_deg", "theta_K (deg)", False, "B  Direction: theta_K"),
              ("sigma_k_frac", "sigma_K / K (-)", True, "C  Magnitude: fractional sigma_K"),
              ("k_column_norm", "||b_K|| (whitened, per m^2/kg)", True, "D  Magnitude: K-column norm"))
    for a, (k, lab, lg, title) in zip(ax.flat, panels):
        a.plot([x["beta_mid_deg"] for x in nm], [x[k] for x in nm], "o-", color=OKABE["blue"],
               label="polar, new-moon epoch (G1)")
        a.plot([x["beta_mid_deg"] for x in qt], [x[k] for x in qt], "s", color=OKABE["vermil"],
               label="polar, quarter-phase epoch (G4)")
        a.axvline(beta_b, color="0.4", ls="--", lw=0.8)
        if k == "f_perp":
            a.axhline(f_ref, color=OKABE["green"], ls=":", lw=1, label="canonical f_ref")
            a.axhspan(0.35, 1.0, color="0.92", zorder=0)
            a.text(2, 0.36, "MODERATE region (f >= 0.35)", fontsize=7, color="0.35")
            a.set_ylim(0, 0.5)
        if lg:
            a.set_yscale("log")
        a.set_xlabel("solar beta, mid-arc (deg)")
        a.set_ylabel(lab)
        a.set_title(title, loc="left", fontsize=9)
    ax[0, 0].legend(fontsize=7, frameon=False)
    ax[0, 1].text(beta_b + 1, ax[0, 1].get_ylim()[0] + 0.5, "full-Sun\nboundary", fontsize=7, color="0.4")
    fig.suptitle("Beta sweep: magnitude moves ~20x, direction stays in the weak regime", fontsize=10)
    fig.savefig(FIG / "f1_beta_direction_vs_magnitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- F2 f_perp vs actual eclipse fraction ----------------------------
    pts = [x for x in rows if x["stage"] in ("G1", "G2", "G3", "G4") and "net_" not in x["id"]]
    cf = [x for x in rows if x["stage"] == "CF"]
    write_csv(FIG / "f2_eclipse.csv", pts + cf)
    d = read_csv(FIG / "f2_eclipse.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    colors = dict(G1=OKABE["blue"], G2=OKABE["orange"], G3=OKABE["green"], G4=OKABE["vermil"])
    for st, c in colors.items():
        s = [x for x in d if x["stage"] == st]
        ax[0].scatter([100 * x["eclipse_fraction"] for x in s], [x["f_perp"] for x in s], s=16, color=c, label=st)
        ax[1].scatter([100 * x["eclipse_fraction"] for x in s], [x["sigma_k_frac"] for x in s], s=16, color=c, label=st)
    for x in (y for y in d if y["stage"] == "CF"):
        src = by[x["id"].replace("CF_noshadow_", "")]
        ax[0].annotate("", xy=(100 * src["eclipse_fraction"], x["f_perp"]),
                       xytext=(100 * src["eclipse_fraction"], src["full_f_perp"]),
                       arrowprops=dict(arrowstyle="->", color=OKABE["black"], lw=0.8))
    ax[0].axhline(f_ref, color="0.4", ls=":", lw=0.8)
    ax[0].set_xlabel("eclipse fraction of arc (%)")
    ax[0].set_ylabel("f_perp (-)")
    ax[0].set_title("A  Direction (arrows: same orbit, SRP shadow removed)", loc="left", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("eclipse fraction of arc (%)")
    ax[1].set_ylabel("sigma_K / K (-)")
    ax[1].set_title("B  Magnitude", loc="left", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    fig.savefig(FIG / "f2_fperp_vs_eclipse_fraction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- F3 altitude ------------------------------------------------------
    g2 = [x for x in rows if x["stage"] == "G2"]
    write_csv(FIG / "f3_altitude.csv", g2)
    d = read_csv(FIG / "f3_altitude.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for tag, norm, c, m in (("b20", "FIXED_DURATION", OKABE["blue"], "o"),
                            ("b20", "FIXED_REVOLUTIONS", OKABE["sky"], "^"),
                            ("b85", "FIXED_DURATION", OKABE["vermil"], "s")):
        s = sorted([x for x in d if tag in x["id"] and x["normalization"] == norm], key=lambda x: x["alt_mean_km"])
        lab = f"beta {tag[1:]}, {'fixed duration' if norm == 'FIXED_DURATION' else '15 revolutions'}"
        ax[0].plot([x["alt_mean_km"] for x in s], [x["f_perp"] for x in s], m + "-", color=c, label=lab)
        ax[1].plot([x["alt_mean_km"] for x in s], [x["sigma_k_frac"] for x in s], m + "-", color=c, label=lab)
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("mean altitude (km)")
    ax[0].axhline(f_ref, color="0.4", ls=":", lw=0.8)
    ax[0].set_ylim(0, 0.5)
    ax[0].set_ylabel("f_perp (-)")
    ax[0].set_title("A  Direction", loc="left", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("sigma_K / K (-)")
    ax[1].set_title("B  Magnitude", loc="left", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    fig.savefig(FIG / "f3_altitude.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- F4 inclination ---------------------------------------------------
    g3 = [x for x in rows if x["stage"] == "G3"]
    write_csv(FIG / "f4_inclination.csv", g3)
    d = read_csv(FIG / "f4_inclination.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for tag, c, lab in (("b00", OKABE["blue"], "beta 0 (eclipse-rich)"),
                        ("bcan", OKABE["vermil"], "beta 21.8 (canonical)")):
        s = sorted([x for x in d if x["id"].endswith(tag)], key=lambda x: x["inc_deg"])
        ax[0].plot([x["inc_deg"] for x in s], [x["f_perp"] for x in s], "o-", color=c, label=lab)
        ax[1].plot([x["inc_deg"] for x in s], [x["sigma_k_frac"] for x in s], "o-", color=c, label=lab)
    ax[0].axhline(f_ref, color="0.4", ls=":", lw=0.8)
    ax[0].set_ylim(0, 0.5)
    ax[0].set_ylabel("f_perp (-)")
    ax[0].set_title("A  Direction", loc="left", fontsize=9)
    ax[1].set_ylabel("sigma_K / K (-)")
    ax[1].set_title("B  Magnitude", loc="left", fontsize=9)
    for a in ax:
        a.set_xlabel("inclination to lunar mean equator (deg)")
    ax[0].legend(fontsize=7, frameon=False)
    fig.savefig(FIG / "f4_inclination.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- F5 full vs matched count ----------------------------------------
    mm = [x for x in rows if x["matched_f_perp"] != ""]
    write_csv(FIG / "f5_matched.csv", mm)
    d = read_csv(FIG / "f5_matched.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.4), constrained_layout=True)
    ax.plot([0, 0.35], [0, 0.35], color="0.6", lw=0.8)
    ax.scatter([x["f_perp"] for x in d], [x["matched_f_perp"] for x in d], s=14, color=OKABE["blue"])
    ax.set_xlabel("f_perp, full operational data (-)")
    ax.set_ylabel("f_perp, matched observation count (-)")
    ax.set_title("Direction is not an observation-count effect", fontsize=9)
    fig.savefig(FIG / "f5_full_vs_matched_count.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- F6 SRP RTN + eclipse timelines ----------------------------------
    reps = ("G1_b00", "G0_consistent", "G1_b88")
    fig, ax = plt.subplots(len(reps), 1, figsize=(7.2, 6.0), sharex=True, constrained_layout=True)
    trace = []
    for a, cid in zip(ax, reps):
        z = np.load(CACHE / f"{cid}.npz")
        t = z["t"] / 3600.0
        keep = t <= 8.0
        g = z["g_rtn"] * 1e6          # um/s^2 per (m^2/kg)
        for j, (lab, c) in enumerate((("R", OKABE["blue"]), ("T", OKABE["orange"]), ("N", OKABE["green"]))):
            a.plot(t[keep], g[keep, j], color=c, lw=0.9, label=lab)
        a.fill_between(t[keep], a.get_ylim()[0] if False else g[keep].min() - 0.05, g[keep].max() + 0.05,
                       where=z["nu"][keep] < 0.5, color="0.85", step="mid", label="eclipse")
        r = by[cid]
        a.set_title(f"{cid}: beta {r['beta_mid_deg']:.1f} deg, eclipse {100 * r['eclipse_fraction']:.0f}%, "
                    f"f_perp {r['full_f_perp']:.3f}, sigma_K/K {100 * r['full_sigma_k_frac']:.0f}%",
                    loc="left", fontsize=8)
        a.set_ylabel("g (um/s^2)")
        for i in np.where(keep)[0][::4]:
            trace.append(dict(case=cid, t_h=float(t[i]), R=float(g[i, 0]), T=float(g[i, 1]),
                              N=float(g[i, 2]), nu=float(z["nu"][i])))
    ax[0].legend(fontsize=7, frameon=False, ncol=4)
    ax[-1].set_xlabel("time from epoch (h), first 8 h shown")
    fig.supylabel("SRP kernel g = da/dK in orbit RTN axes (um/s^2 per m^2/kg)", fontsize=8)
    write_csv(FIG / "f6_srp_rtn.csv", trace)
    fig.savefig(FIG / "f6_srp_rtn_and_eclipse_timelines.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- F7 Earth view / measurement projection ---------------------------
    ev = [x for x in rows if x["stage"] in ("G1", "G4") and "net_" not in x["id"]]
    net = [x for x in rows if "net_" in x["id"]] + [x for x in rows if x["id"] == "G0_consistent"]
    write_csv(FIG / "f7_earth_view.csv", ev + net)
    d = read_csv(FIG / "f7_earth_view.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    view = [x for x in d if "net_" not in x["id"] and x["id"] != "G0_consistent"]
    sc = ax[0].scatter([x["earth_view_mid_deg"] for x in view], [x["f_perp"] for x in view],
                       c=[abs(x["beta_mid_deg"]) for x in view], cmap="viridis", s=22,
                       marker="o", edgecolor="k", linewidth=0.3)
    for x in view:
        if x["epoch_offset_days"] > 1:
            ax[0].scatter([x["earth_view_mid_deg"]], [x["f_perp"]], s=70, facecolor="none",
                          edgecolor=OKABE["vermil"], linewidth=1.0)
    fig.colorbar(sc, ax=ax[0], label="|beta| (deg)")
    ax[0].set_xlabel("orbit normal vs Moon-Earth line (deg)\n0/180 face-on, 90 edge-on")
    ax[0].set_ylabel("f_perp (-)")
    ax[0].set_title("A  Direction vs Earth view (red rings: quarter phase)", loc="left", fontsize=8)
    lab = [x["stations"].replace(" DSN", "") for x in d if x in net or x["id"] in
           ("G0_consistent",) or "net_" in x["id"]]
    nets = [x for x in d if "net_" in x["id"] or x["id"] == "G0_consistent"]
    ax[1].bar(range(len(nets)), [x["f_perp"] for x in nets], color=OKABE["blue"])
    ax[1].set_xticks(range(len(nets)))
    ax[1].set_xticklabels([x["stations"].replace(" DSN", "").replace(";", "+") for x in nets],
                          rotation=20, fontsize=7)
    ax[1].set_ylabel("f_perp (-)")
    ax[1].set_title("B  Same orbit, different station sets", loc="left", fontsize=8)
    fig.savefig(FIG / "f7_earth_view_and_network.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- F8 fixture attribution ------------------------------------------
    fa = json.loads((ART / "phase17_geo_fixture_attribution.json").read_text())["rows"]
    fa = [x for x in fa if x["mode"] != "CAMPAIGN_TABLE"]
    write_csv(FIG / "f8_fixture.csv", fa)
    d = read_csv(FIG / "f8_fixture.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    names = ["published\n(5.83 d early\n+ clamped)", "offset only", "clamp only", "consistent\n(GEO)"]
    ax[0].bar(names, [x["f_perp"] for x in d], color=[OKABE["vermil"], OKABE["orange"], OKABE["sky"], OKABE["blue"]])
    ax[0].set_ylabel("f_perp (-)")
    ax[0].set_title("A  Direction, canonical orbit", loc="left", fontsize=9)
    ax[1].bar(names, [x["sigma_k_frac"] for x in d], color=[OKABE["vermil"], OKABE["orange"], OKABE["sky"], OKABE["blue"]])
    ax[1].set_ylabel("sigma_K / K (-)")
    ax[1].set_title("B  Magnitude, canonical orbit", loc="left", fontsize=9)
    for a in ax:
        a.tick_params(axis="x", labelsize=7)
    fig.savefig(FIG / "f8_fixture_attribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- F9 K magnitude control -------------------------------------------
    g5 = sorted([x for x in rows if x["stage"] == "G5" or x["id"] == "G0_consistent"], key=lambda x: x["k_srp"])
    write_csv(FIG / "f9_k_control.csv", g5)
    d = read_csv(FIG / "f9_k_control.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    k = np.array([x["k_srp"] for x in d])
    ax[0].plot(k, [x["f_perp"] for x in d], "o-", color=OKABE["blue"])
    ax[0].set_xscale("log")
    ax[0].set_ylim(0, 0.5)
    ax[0].set_ylabel("f_perp (-)")
    ax[0].set_title("A  Direction: invariant to K_truth", loc="left", fontsize=9)
    ax[1].loglog(k, [x["sigma_k_frac"] for x in d], "o-", color=OKABE["vermil"], label="measured")
    ax[1].loglog(k, d[2]["sigma_k_frac"] * k[2] / k, "--", color="0.4", lw=0.8, label="1/K prediction")
    ax[1].set_ylabel("sigma_K / K (-)")
    ax[1].set_title("B  Magnitude: relative precision ~ 1/K", loc="left", fontsize=9)
    ax[1].legend(fontsize=7, frameon=False)
    for a in ax:
        a.set_xlabel("K_SRP truth (m^2/kg)")
    fig.savefig(FIG / "f9_k_magnitude_control.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


RESULTS_EPOCH0 = 0.0

if __name__ == "__main__":
    main()
