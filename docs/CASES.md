# Case studies — real NIFTY smile, 3 Nov 2026

All numbers below are produced by `python main.py` from the real NSE bhavcopy
chain of 3 Nov 2026 (spot 23,810.85; r = 6.5%, q = 1.3%). Slices: expiry
2026-11-09 (6 DTE, 103 fit points), 2026-11-23 (20 DTE, 112 points),
2026-12-29 (56 DTE, 88 points), OTM options only, calibration window
|log(K/F)| ≤ 0.15. Errors are RMSE / max-abs in **volatility points** against
the raw market IVs of the same slice.

One data feature hangs over everything: the settlement smile has a
**parity-break step at the money** — on the 6-DTE slice the 23,800 put implies
19.7% while the adjacent 23,850 call implies 11.4%. No smooth model can fit a
step; every model's worst weekly error sits exactly on that seam (SVI max
error 4.07 pts and SABR's 6.17–6.33 pts are both at K = 23,850). Read the
scoreboard with that in mind: it is partly a measurement of the data.

## Case A — the calibration scoreboard

| Slice | SVI | SABR β=0.5 | SABR β=1.0 | Heston | Heston (Feller) |
|---|---|---|---|---|---|
| 6 DTE | **1.163** | 2.416 | 2.531 | 5.826 | 5.273 |
| 20 DTE | **0.723** | 0.805 | 0.789 | 0.831 | 2.303 |
| 56 DTE | **0.410** | 0.480 | 0.470 | 0.422 | 0.941 |

(RMSE vs market, vol pts. SVI also scores 0.000 against itself by construction
and is excluded from "best dynamic model" claims.)

**Short expiry (6 DTE): SABR wins among dynamic models; Heston fails.**
The weekly smile is savage: IV runs from 52% in the far put wing to 11.4% at
the money over Δk ≈ 0.17, and Heston's skew is structurally too weak at this
maturity — its worst error is **25.9 vol points** at K = 20,650, and its
calibration shoves σ (vol-of-vol) into the 3.00 bound trying to manufacture
curvature. SABR, refit per slice with ν pinned at its 5.00 bound, bends far
enough to keep RMSE at 2.4–2.5. This is Heston's known short-expiry weakness,
measured on real data rather than asserted.

**Long expiry (56 DTE): Heston wins among dynamic models — and essentially
reproduces SVI.** Heston RMSE 0.422 vs SVI's 0.410, and against the SVI
*smooth* Heston's RMSE is just **0.104 vol pts**: at this maturity the smile
is exactly the shape Heston dynamics generate. SABR (0.470–0.480) is close
behind. At 20 DTE all three dynamic models sit within 0.05 vol pts of each
other (0.79–0.83) — the smile there is gentle enough that model choice barely
matters for *fitting*. (It still matters for hedging — Case C.)

**β choice in SABR barely matters here** (ΔRMSE ≤ 0.12 vol pts between β = 0.5
and β = 1.0 on every slice): with α, ρ, ν all free, the two backbones fit the
same static smile almost identically. β is a *dynamics* assumption, not a fit
parameter — which is precisely Case C's point. SABR's ρ is strikingly stable
(−0.47…−0.55 across slices and both β) and agrees with Heston's ρ
(−0.52…−0.59): two unrelated models extract the same spot/vol correlation
from the data. SABR's ν decays with maturity (5.00 → 3.70 → 2.07), the
textbook pattern.

## Case B — the Feller / parameter-sanity audit

| Slice | κ | θ (vol) | σ | ρ | v₀ (vol) | Feller 2κθ−σ² | RMSE → Feller-forced |
|---|---|---|---|---|---|---|---|
| 6 DTE | 0.88 | 1.500 (122.5%) | 3.00 | −0.519 | 0.0227 (15.1%) | **−6.37 violated** | 5.826 → 5.273 |
| 20 DTE | 16.40 | 0.0655 (25.6%) | 3.00 | −0.535 | 0.0036 (6.0%) | **−6.85 violated** | 0.831 → 2.303 |
| 56 DTE | 0.20 | 1.465 (121.0%) | 1.21 | −0.588 | 0.0003 (1.8%) | **−0.89 violated** | 0.422 → 0.941 |

Every unconstrained per-slice calibration **violates the Feller condition** —
the fitted variance process can hit zero. Worse, the parameters are
economically implausible and mutually inconsistent: θ (long-run variance)
swings from 1.50 to 0.066 to 1.46 across slices of the *same day*, σ sits on
its upper bound on both weekly slices, and the 56-DTE fit wants variance to
start at 1.8% vol and mean-revert (κ = 0.20!) toward 121% vol. These are not
estimates of anything real; they are five knobs contorting to match one
smile each. ρ ≈ −0.55 is the only parameter that behaves like a measurement.

Forcing Feller (penalised calibration) lands essentially on the boundary
(residual violation ≤ 3×10⁻⁴) but buys it with v₀ pinned at its lower bound.
The fit cost is slice-dependent: **+1.47 vol pts at 20 DTE, +0.52 at 56 DTE**
— and −0.55 at 6 DTE, where the penalty happened to steer the optimiser out
of a worse local optimum (a regularisation accident, reported as measured,
not as a virtue of the constraint).

**Lesson:** a per-slice Heston "calibration" is a curve-fitting exercise.
Production desks calibrate Heston *jointly* across the term structure
precisely so that one parameter set must explain all slices at once — the
inconsistency above is what that discipline exists to prevent.

## Case C — smile dynamics: same smile, three different hedges

Setup: 20-DTE slice, the old-ATM call (K = 23,800). Spot drops 2%:
23,810.85 → 23,334.63. Each scenario reprices the option its own way.

| Model | IV before | IV after | ΔIV (vol pts) | Delta after |
|---|---|---|---|---|
| Sticky-strike | 13.74% | 13.74% | 0.00 | 0.305 |
| Heston | 13.08% | 11.47% | −1.61 | 0.278 |
| SABR (β=0.5) | 13.50% | 12.18% | −1.32 | 0.281 |

Two results, both counter-intuitive until you think in moneyness:

1. **The dynamic models *lower* the fixed-strike IV after a spot fall.** The
   variance state hasn't moved in an instantaneous shock; what moves is the
   strike's position on the smile. K = 23,800 slides from at-the-money to
   ~2% out-of-the-money, down the steep negative skew — worth −1.3 to −1.6
   vol points, swamping any level effect. "Vol spikes when the market
   falls" is a statement about ATM vol and about time passing; at a *fixed
   strike, instantaneously*, the skew-slide can dominate.
2. **The hedge differs by 2.8 delta points** (0.305 sticky vs 0.278 Heston)
   on the same option after the same shock. A sticky-strike hedger — anyone
   running plain Black-Scholes with yesterday's IVs — systematically
   over-hedges this call relative to either dynamic model, because they
   keep the IV (and hence the delta) too high. Scaled to a book, 2.8 delta
   points per option is the difference between hedged and quietly long.

Model choice is not a fitting contest: three models within 0.05 vol pts of
each other on the static smile (Case A, 20 DTE) disagree by 2.8 delta
points the moment spot moves. You calibrate dynamics to hedge, not to fit.
