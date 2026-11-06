# StochVol Lab

![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![release](https://img.shields.io/github/v/release/risootooo/stochvol-lab)
![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)
![Data](https://img.shields.io/badge/data-real%20NSE%20bhavcopy-orange)
![Models](https://img.shields.io/badge/models-Heston%20%C2%B7%20SABR%20%C2%B7%20SVI-purple)

**Heston and SABR, implemented from scratch and calibrated to the real NIFTY implied-volatility smile** — because on a derivatives desk, model calibration is the daily craft: fit the smile, audit the parameters, and know what your model implies for the hedge before spot moves.

Every number below comes from the **real NSE bhavcopy chain of 3 Nov 2026** (spot 23,810.85), OTM options only, and is reproduced by running `main.py`.

## Headline scoreboard (RMSE vs market IV, vol points)

| Slice | SVI | SABR β=0.5 | SABR β=1.0 | Heston |
|---|---|---|---|---|
| 9 Nov 2026 · 6 DTE | **1.16** | 2.42 | 2.53 | 5.83 |
| 23 Nov 2026 · 20 DTE | **0.72** | 0.81 | 0.79 | 0.83 |
| 29 Dec 2026 · 56 DTE | **0.41** | 0.48 | 0.47 | 0.42 |

The story in one paragraph: Heston **fails the weekly expiry** (worst single error 25.9 vol points in the far put wing — its short-dated skew is structurally too weak) and **wins the monthly among dynamic models**, matching SVI to 0.10 vol pts against the SVI smooth. SABR is never the best and never embarrassing. And every unconstrained Heston calibration **violates the Feller condition** with parameters no economist would sign (long-run vol of 122%?) — see the audit in [docs/CASES.md](docs/CASES.md). Three models that fit the 20-DTE smile within 0.05 vol pts of each other then disagree by **2.8 delta points** after a −2% spot shock. Calibration is a hedging decision, not a fitting contest.

## What's inside

| Module | Contents |
|---|---|
| `src/stochvol/heston.py` | Heston characteristic function ("little trap" formulation), Gil–Pelaez semi-analytic pricing, bump-and-revalue delta, full-truncation Euler Monte Carlo (cross-check only), least-squares calibration with optional Feller penalty |
| `src/stochvol/sabr.py` | Hagan et al. lognormal SABR formula, α-from-ATM solver, (α, ρ, ν) calibration at fixed β ∈ {0.5, 1.0} |
| `src/stochvol/svimodel.py` | Raw SVI fit — the desk-standard benchmark curve every model is also scored against |
| `src/stochvol/dynamics.py` | Smile-dynamics experiment: sticky-strike vs Heston repricing vs SABR backbone under a spot shock |
| `src/stochvol/fitreport.py` | Per-slice RMSE / max-error / worst-strike / bias tables vs market and vs SVI |
| `src/stochvol/bs.py` | Black–Scholes–Merton primitives + IV solver that returns NaN on arbitrage-violating prints instead of inventing a vol |

## The maths

**Heston** (risk-neutral): dS/S = (r−q)dt + √v dW₁, dv = κ(θ−v)dt + σ√v dW₂, corr = ρ.
Pricing uses the characteristic function of log-spot in the little-trap form
(Albrecher et al. 2007), with d = √((κ−ρσiu)² + σ²(iu+u²)),
g = (κ−ρσiu−d)/(κ−ρσiu+d),
C = (r−q)iuT + (κθ/σ²)[(κ−ρσiu−d)T − 2 ln((1−g e^(−dT))/(1−g))],
D = ((κ−ρσiu−d)/σ²)·(1−e^(−dT))/(1−g e^(−dT)),
and Gil–Pelaez inversion for the two exercise probabilities (the share-measure
CF is normalised by φ(−i) = E[S_T]). Integration: trapezoid on u ∈ (0, 300],
2048 nodes (|φ(300)| < 10⁻⁴ on all slices; probability error ~10⁻⁵).

**SABR** (Hagan et al. 2002), lognormal implied vol with the standard
z/x(z) correction and the (1 + [ (1−β)²α²/(24(FK)^(1−β)) + ρβνα/(4(FK)^((1−β)/2)) + (2−3ρ²)ν²/24 ] T) term — see `sabr.py`.

**SVI** (raw): w(k) = a + b[ρ(k−m) + √((k−m)² + s²)], σ = √(w/T).

## Data & provenance

- Source file: `BhavCopy_NSE_FO_0_0_0_20261103_F_0000.csv` (NSE F&O bhavcopy,
  trade date 2026-11-03). The 9 Nov / 23 Nov slices were taken via the
  vol-surface-atlas panel extracted from the same file; the 29 Dec slice was
  read directly from the raw file. Regenerate with
  `python scripts/extract_data.py` (needs the sibling data files; the
  extracted result is committed at `data/slices_2026-11-03.csv`).
- **OTM only.** ITM settlement prints routinely violate no-arbitrage bounds
  (documented in the sibling greeks-lab / vol-surface-atlas repos), so they
  are excluded by construction.
- Rates: r = 6.5% (the sibling labs' convention), q = 1.3% (stated
  assumption for the NIFTY dividend yield).
- Cleaning: premiums ≥ 1 point, OI > 0, |log(K/F)| ≤ 0.20, IV ∈ [3%, 150%],
  plus a rolling-median spike filter that removed 8 isolated bad settlement
  prints (one was ~8 vol points off both neighbours). Calibration window
  |log(K/F)| ≤ 0.15 → 103 / 112 / 88 points per slice.
- Calibration objective: least squares on IVs for SABR/SVI; for Heston the
  standard vega-scaled price difference (first-order IV difference), with
  deterministic multi-start restarts.

## Run it

```bash
/tmp/venv-test/bin/python main.py                    # Cases A/B/C + docs/results.json  (~5 min: Heston calibrations)
/tmp/venv-test/bin/python scripts/build_dashboard.py # docs/dashboard.html (self-contained, offline)
/tmp/venv-test/bin/python -m pytest tests -q         # 17 tests
```

## Honest limitations

- **Settlement prices are not traded prices.** The weekly smile carries a
  parity-break step at the money (adjacent put/call IVs differ by ~8 vol
  points at 6 DTE) that no smooth model can fit; it inflates every model's
  weekly RMSE and dominates the max-error column. Far-wing weekly IVs are
  implied by 2–6 point premiums.
- **Per-slice Heston calibration is curve fitting, not estimation** (Case B):
  parameters are unstable across slices of the same day. A production
  calibration is joint across the term structure; that is the obvious next
  step and is deliberately not faked here.
- **SABR's formula is an asymptotic approximation**, and on the 6-DTE slice
  its ν pins at the 5.00 bound — the formula is being stretched past its
  design range, and the scoreboard says so.
- **One day, one underlying.** No term-structure dynamics, no American
  exercise (NIFTY options are European), no transaction costs in the
  dynamics experiment; the ±2% shock is instantaneous, so the variance state
  does not move with spot — the experiment isolates the smile-slide effect
  by design.
- Fourier-integration absolute error (~10⁻⁵ in probability ≈ 0.2 index
  points on spot-sized terms) means deep-OTM short-dated model prices can
  print slightly negative before flooring; documented in `heston.py`.

## Disclaimer

Research and education only. Nothing here is investment advice, a trading
recommendation, or a claim that any model describes how NIFTY options will
behave on any other day.

## License

MIT
