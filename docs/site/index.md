# Lunar Orbit Determination

Build reproducible lunar tracking experiments, fit orbital states, and inspect
recorded runs in a native desktop workspace.

**Start here:** [Run a data-free OD fit](../tutorials/quickstart.md) ·
[Explore the results](../research/results-gallery.md) ·
[Check supported combinations](../validation/support-matrix.md)

![Native desktop mission workspace](../images/desktop-mission.png)

## Two ways to use the project

| Python library | Windows desktop |
|---|---|
| Configure and reproduce numerical experiments from scripts | Configure scenarios, monitor execution and compare recorded runs |
| Install with `python -m pip install -e .` | Run `setup_windows.cmd`, then `launch_lunar_od.cmd` |
| Start with the seeded batch-fit tutorial | Follow the [Windows guide](../windows-quickstart.md) |

## Evidence you can inspect

The [gallery](../research/results-gallery.md) links every new plot to its data,
configuration, seed and producing source. The [validation guide](validation.md)
explains what CI checks and where research qualification begins. The
[study guide](studies.md) points to current conclusions and superseded records.

The project is research software. Its main evidence is synthetic and does not
establish flight qualification or real-mission navigation accuracy.

## Reuse and citation

The source is [MIT licensed](../../LICENSE). Use [CITATION.cff](../../CITATION.cff)
and record the exact revision, study configuration, seed and external inputs.
No DOI has been assigned. [Release and DOI procedure](../development/releases.md).
