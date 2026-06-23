# UI Design System

Use when designing Lunar OD desktop screens: dashboard, diagnostics, result
browsers, configuration workflows.

## Style direction
Professional scientific desktop-app style — a dark technical cockpit theme unless
the existing app theme differs. It should feel like mission-analysis /
estimator-diagnostics / repeatable-experiment-control software, **not** a marketing
landing page, decorative dashboard, slideshow, or one-off demo.

## Visual principles
Use: restrained colors, clear hierarchy, compact-but-readable panels, consistent
spacing / typography / icon behavior, clear status indicators, readable plots and
tables, visible units, clear warning/failed states.
Avoid: decorative clutter, marketing hero sections inside the app, excessive
gradients, oversized or nested cards, hiding scientific assumptions behind polish,
color-only status communication.

## Plot-first
Plots, tables, residual diagnostics, estimator metrics, run logs, and result
provenance have priority over decorative elements. Keep scientific areas readable
before making a screen impressive.

## Cards
Use cards only for grouped tools, repeated result summaries, diagnostics panels,
estimator-comparison blocks, scenario-config groups. Avoid nested cards.

## Scientific values
Numerical values should show unit, reference frame (where relevant), time system /
epoch (where relevant), and estimator/scenario context (where relevant).

## Accessibility
Sufficient contrast; readable fonts for long sessions; status not by color alone;
distinct warning / failed / running / success states.

## Critical rule
The UI should feel like a mission-analysis and estimator-diagnostics tool, not a
landing page.
