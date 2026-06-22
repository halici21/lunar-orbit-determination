# Figure style guide for lunar OD

## Palette
- Estimator-consistent: BLS-LM navy `#012855`, SR-UKF gold `#c89a2b`, third
  series teal `#0f9b8e`, reference / limit lines red `#b00` dotted.
- Colorblind-safe baseline: **Okabe-Ito**. Sequential maps: viridis / plasma;
  diverging: coolwarm / RdBu. Never jet / rainbow.

## Matplotlib conventions
- Headless: `import matplotlib; matplotlib.use("Agg")`.
- `constrained_layout=True`; set `figsize` at creation.
- Multi-panel: `plt.subplots`, `plt.subplot_mosaic`, or `GridSpec` for complex
  asymmetric layouts; label panels with bold letters (A, B, ...).
- Global defaults via `rcParams` (font 10-11; titles bold, navy).
- Save: `fig.savefig(path, dpi=300, bbox_inches="tight")`.

## Accessibility & units
- Units in every axis label: m, m/s, arcsec, rad, s, deg, Hz.
- Test in grayscale; series must stay distinguishable without color (markers /
  linestyles, not color alone).
- `np.nan` breaks for wrapped or discontinuous tracks (longitude wrap, polar
  singularity) so matplotlib draws no spurious connecting lines.

## Targets & formats
- **Thesis**: vector PDF/SVG, single-column width, >= 11 pt fonts, one-sentence
  caption beneath.
- **README**: 300-dpi PNG, wide multi-panel, self-explanatory title, legend
  inside the axes.
- **Slides**: 300-dpi PNG, <= 2 visible series, large fonts, high contrast,
  minimal grid.
- Output dir: `python_port/docs/figures/` (documentation),
  `python_port/results/` (experiment outputs).

## Reference example
`python_port/examples/plot_aberration_corrections.py` already follows these
conventions (3-panel figure, navy/gold/teal, Agg, dpi 150+).
