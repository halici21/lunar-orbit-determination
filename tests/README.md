# Test Profiles

Fast default suite:

```powershell
python -m unittest discover -s tests
```

Full slow regression suite:

```powershell
$env:LUNAR_OD_RUN_SLOW_TESTS = "1"
python -m unittest discover -s tests
Remove-Item Env:\LUNAR_OD_RUN_SLOW_TESTS
```

Slow tests cover seeded Monte Carlo, long-arc UKF, two-way Doppler light-time,
30+ state square-root stress, and scenario end-to-end handoff regressions.
