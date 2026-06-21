# Canonical sources for lunar OD literature work

| Source | Authoritative for |
|---|---|
| Moyer, *Formulation for Observed and Computed Values of DSN Data Types for Navigation*, JPL DESCANSO Monograph 2 (2000) | two-/three-way counted Doppler, range, light-time, relativistic and media terms |
| Thornton & Border, *Radiometric Tracking Techniques for Deep-Space Navigation*, DESCANSO Monograph 1 (2003) | Doppler / range / VLBI tracking, precision budgets |
| Tapley, Schutz & Born, *Statistical Orbit Determination* (2004) | batch & sequential estimation, observation models, NIS/NEES, observability |
| Montenbruck & Gill, *Satellite Orbits: Models, Methods and Applications* (2000) | dynamics, observation models, light-time |
| Vallado, *Fundamentals of Astrodynamics and Applications* | RAZEL / SEZ geometry, instantaneous range-rate |
| DSN 810-005, Telecommunications Link Design Handbook | coherent turnaround ratios (S 240/221, X 880/749), band definitions |
| NAIF/SPICE Required Reading — aberration corrections | NONE / LT / CN / LT+S / CN+S definitions, reception vs transmission |

## Usage notes
- When the answer depends on a specific equation or numeric bound (e.g. the
  one-iteration light-time relative error ~ beta^2), name the source and tell the
  user to confirm the exact clause before quoting it in the thesis.
- Prefer the in-repo validation documents as the citable evidence for what this
  implementation actually does (they contain measured numbers), and the books
  above for the theory those numbers are checked against.
