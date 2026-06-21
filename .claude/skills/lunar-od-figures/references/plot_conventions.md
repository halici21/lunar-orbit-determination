# Figure conventions for lunar OD

## Palette (colorblind-safe, consistent with docs/figures/)
- BLS-LM: navy `#012855`
- SR-UKF: gold `#c89a2b`
- third / accent series: teal `#0f9b8e`
- reference / limit lines: red `#b00`, dotted
- grid: `alpha` 0.2–0.25

## Style
- `font.size` 10–11; titles bold and navy.
- Headless: `import matplotlib; matplotlib.use("Agg")`.
- Save: `fig.savefig(path, dpi=150)`; documentation figures go in
  `python_port/docs/figures/`.
- Units always in axis labels: m, m/s, arcsec, rad, s, deg, Hz.
- Use `np.nan` breaks for wrapped or discontinuous tracks (longitude wrap, polar
  singularity) so matplotlib does not draw spurious connecting lines.

## Target layouts
- **Thesis**: single-column width, ≥ 11 pt fonts, one-sentence caption beneath.
- **README**: wide multi-panel, legend inside axes, self-explanatory title.
- **Slides**: at most two visible series, large fonts, high contrast, minimal grid.

## Reference example
`python_port/examples/plot_aberration_corrections.py` already follows these
conventions (3-panel figure, navy/gold/teal, Agg, dpi 150) and is a good template.
