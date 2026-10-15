# PHASE 17-R1O-OPT — LUNAR LANDMARK OPTICAL MEASUREMENT LITERATURE CONTRACT

For every physical/error term the production model represents or defers: literature source, exact
value/equation, repository representation, and status. Read before `lunar_od/lunar_landmark_optical.py`.

## Primary references

- **Federici, Genova, Andolfo, Ciambellini, Teodori, Torrini**, *Optical Camera Characterization for
  Feature-Based Navigation in Lunar Orbit*, Aerospace **12**(5), 374 (2025). Modern anchor for
  feature-based lunar-orbit optical measurement noise: optical measurement noise set to **2.5
  pixels**, corresponding to a ground resolution of **~160 m** based on their modeled camera's focal
  length, pixel pitch, and altitude — i.e. an implied ground-sample-distance (GSD) of **~64 m/pixel**
  at their (unstated, plausibly comparable low-lunar-orbit) altitude. The paper's exact focal
  length/pixel-pitch/FOV values were not independently recoverable (access-restricted); the GSD figure
  is used as a plausibility anchor for this phase's own camera choice (§ below), not copied verbatim.
- **Apollo Guidance and Navigation System** (NASA/MIT documentation; ION Museum; NTRS 19720026900).
  The CM space sextant: 28× magnification, 1.8° field of view, **RMS sighting accuracy 10 arcsec**
  (≈ 50 μrad) for landmark/star angular measurements — the historical precedent this phase's §6
  discusses.
- **CubeSat-class star tracker performance surveys** (satsearch; BCT XACT flight performance;
  representative compact-tracker literature): attitude knowledge **8–10 arcsec (1-σ)** for
  higher-quality compact trackers (BCT XACT: 8 arcsec 1-σ demonstrated in-orbit on MinXSS), up to
  **~40–120 arcsec (3-σ)** class for coarser/compact sensors. A representative compact optical
  sensor's own pixel dimension (~40 arcsec/pixel) with sub-pixel centroiding to ~0.245 arcsec
  illustrates the achievable centroiding-to-pixel ratio (~1/163 pixel) at the high-quality end.
- **R1O's own prior citations** (carried forward, not re-derived): LONEStar (Lunar Flashlight
  extended mission, 2023–2024) empirical LOS errors 0.25–1 pixel; a commonly-cited star-tracker
  attitude-knowledge floor of ≈0.5 mrad (≈103 arcsec) 1-σ conservative.

`OPT_LITERATURE_CONTRACT_GATE = PASS` — every term below is traced to a cited source or explicitly
marked deferred with a stated reason.

## Camera parameter choice — reasoned, not copied

Federici et al.'s exact camera intrinsics were not recoverable from available sources. Rather than
invent unlabeled numbers, a representative navigation camera is chosen and its resulting ground
resolution is compared explicitly against Federici's cited ~64 m/pixel GSD as a plausibility check:

```
focal length            f = 35 mm
pixel pitch              p = 5.5 um   (common CMOS pitch)
sensor format             2048 x 2048 px  (11.3 mm x 11.3 mm)
IFOV per pixel            p/f = 1.571e-4 rad/px  ~= 32.4 arcsec/px
full field of view        ~18.3 deg
GSD at this campaign's ~105.5 km altitude (measured directly from the campaign trajectory,
  not assumed):  altitude * IFOV ~= 105,470 m * 1.571e-4 rad ~= 16.6 m/pixel
```

16.6 m/pixel is the same order of magnitude as Federici's implied 64 m/pixel (finer by ~4x, consistent
with choosing a longer-focal-length dedicated navigation camera over a wider-FOV descent/context
camera) — a defensible, explicitly-reasoned choice, not a forced match.

## Term-by-term contract

| # | Term | Literature source | Repo representation | Status |
|---|---|---|---|---|
| 1 | Pinhole/gnomonic projection | Standard photogrammetric model (any camera-calibration text; matches the pinhole model Federici et al. and the wider feature-based-nav literature all use as the baseline) | `lunar_landmark_optical.pinhole_project()` | **Implemented** |
| 2 | Lens distortion | Not modeled in this phase | — | `DISTORTION_MODEL = DEFERRED_CHARACTERIZED` (radial distortion at few-arcmin FOV corners is typically sub-pixel for well-corrected nav optics, but this is asserted, not verified, here) |
| 3 | Camera intrinsics (focal length, pixel pitch, principal point) | Representative choice above, GSD-checked against Federici | `CameraIntrinsics` dataclass | **Implemented** |
| 4 | Camera-to-body / body-to-inertial attitude chain | No existing attitude infrastructure in this repository (audited, §10 of main report) | Explicit external DCM input, nadir-pointing boresight by default (matching R1O's own landmark surrogate convention), injectable bias/noise (§18-20 of main report) | **Implemented** as a documented external input |
| 5 | Attitude random knowledge error | CubeSat star-tracker surveys: 8-10 arcsec (1-sigma) high-quality; R1O's own 500 urad (~103 arcsec) conservative floor as a degraded anchor | noise sweep, §33 of main report | **Implemented**, literature-anchored sweep |
| 6 | Attitude/boresight bias | Not a standard single literature number; characterized as a controlled synthetic case, mirroring the DDOR session-bias methodology (R1O-D) | §34 of main report | **Implemented**, methodology reused from R1O-D |
| 7 | Landmark lunar-fixed frame | Already-qualified `moon_pa_de440_rotation_at_et` (versioned DE440 realization, not the load-order-dependent generic alias — R1M/R1O precedent) | reused unmodified | **Implemented**, reused qualified infrastructure |
| 8 | Landmark position model | Spherical-Moon reference radius (R1O precedent) | `R_MOON_M` from `lunar_od.constants`, spherical | `SPHERICAL_REFERENCE` (§23 of main report classifies the limitation) |
| 9 | Optical light time | Moyer-consistent one-way light-time significance test (R1O-D precedent methodology) | tested directly, §18 of main report | **Implemented**, tested not assumed |
| 10 | Landmark visibility (occultation, horizon, FOV, front-of-camera) | Standard OD geometric visibility logic | §19 of main report | **Implemented** |
| 11 | Illumination / feature detectability | Not modeled as a photometric renderer | solar-incidence-angle characterization only | `FEATURE_DETECTABILITY_MODEL = CHARACTERIZED_NOT_FULLY_IMPLEMENTED` |
| 12 | Centroiding / image-plane measurement noise | Federici et al.: 2.5 px representative; LONEStar: 0.25-1 px high-quality; sub-pixel (~0.1-0.3 px) dedicated-camera best case | pixel-domain noise sweep, §32 of main report | **Implemented**, literature-anchored sweep |
| 13 | Landmark catalog / map uncertainty | R1O's own prior 50/200 m anchors (carried forward) | §35-36 of main report | **Implemented**, reused prior literature grounding |
| 14 | Camera calibration error (focal length, principal point, boresight alignment) | Generic camera-calibration-residual treatment; no single literature number adopted | §37 of main report | **Implemented** as controlled perturbation cases |
| 15 | K_SRP sensitivity composition | Phase 17-R's qualified `_two_way_range_k_srp_column` / R1O-D's implicit-event-matrix substitution pattern | analogous chain-through-S_K composition | **Implemented**, reusing the qualified substitution principle |

## What this module does NOT implement

Per the governing spec's explicit scope boundary (§13): no raw image rendering, no crater-detection
CNN, no feature-descriptor extraction, no image segmentation, no full vision pipeline. Landmark
identity and catalog coordinates are assumed known; the measured image centroid (u,v) is the OD-level
observable this module produces and consumes — analogous to how the production radiometric OD
consumes processed tracking observables rather than raw RF voltages.
