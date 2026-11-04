"""Heston stochastic-volatility model, implemented from scratch.

Dynamics (risk-neutral):  dS/S = (r - q) dt + sqrt(v) dW1,
dv = kappa (theta - v) dt + sigma sqrt(v) dW2,  corr(dW1, dW2) = rho.

Pricing is semi-analytic: the characteristic function of log-spot in the
"little Heston trap" formulation (Albrecher, Mayer, Schoutens & Tistaert 2007,
which avoids the branch-cut discontinuities of the original 1993 form), and
Gil-Pelaez Fourier inversion for the two exercise probabilities. The integral
is evaluated on a fixed trapezoid grid u in (0, 300] with 2048 nodes
(|phi(300)| < 1e-4 on every calibration slice; absolute probability error
~1e-5). One honest consequence: a deep-OTM short-dated option worth ~0.3 pts
can come back at -0.2 pts, because its price is a small difference of two
~spot-sized terms. Prices are therefore floored at 1e-6 before IV inversion,
and calibration residuals use the standard vega-scaled price difference
(first-order IV difference) instead of repeated exact inversion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .bs import bs_price, implied_vol

_U = np.linspace(1e-3, 300.0, 2048)         # integration grid (module-level, fixed)
_DU = _U[1] - _U[0]


@dataclass
class HestonParams:
    kappa: float   # mean-reversion speed of variance
    theta: float   # long-run variance
    sigma: float   # vol of variance
    rho: float     # spot/variance correlation
    v0: float      # initial variance

    def as_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.sigma, self.rho, self.v0])

    @property
    def feller(self) -> float:
        """2*kappa*theta - sigma^2: positive => Feller condition holds (variance stays > 0)."""
        return 2 * self.kappa * self.theta - self.sigma**2

    @property
    def feller_ok(self) -> bool:
        return self.feller > 0


def _char_func(u: np.ndarray, S, T, r, q, p: HestonParams) -> np.ndarray:
    """Characteristic function of log S_T (little-trap formulation)."""
    iu = 1j * u
    d = np.sqrt((p.kappa - p.rho * p.sigma * iu) ** 2 + p.sigma**2 * (iu + u**2))
    g = (p.kappa - p.rho * p.sigma * iu - d) / (p.kappa - p.rho * p.sigma * iu + d)
    exp_dt = np.exp(-d * T)
    C = (r - q) * iu * T + (p.kappa * p.theta / p.sigma**2) * (
        (p.kappa - p.rho * p.sigma * iu - d) * T
        - 2.0 * np.log((1 - g * exp_dt) / (1 - g))
    )
    D = ((p.kappa - p.rho * p.sigma * iu - d) / p.sigma**2) * (1 - exp_dt) / (1 - g * exp_dt)
    return np.exp(C + D * p.v0 + iu * np.log(S))


def _probabilities(S, K, T, r, q, p: HestonParams):
    """P1, P2 of the Gil-Pelaez inversion, vectorised over strikes K (array)."""
    K = np.atleast_1d(np.asarray(K, float))
    lnK = np.log(K)
    phi = _char_func(_U, S, T, r, q, p)                       # (n_u,)
    phi_mi = _char_func(_U - 1j, S, T, r, q, p)               # phi(u - i)
    phi_mi0 = _char_func(np.array([-1j]), S, T, r, q, p)[0]   # phi(-i) = E[S_T]
    phi_mi = phi_mi / phi_mi0                                 # share-measure CF
    phase = np.exp(-1j * np.outer(_U, lnK))                    # (n_u, n_K)
    int2 = np.trapezoid(np.real(phase * (phi / (1j * _U))[:, None]), dx=_DU, axis=0)
    int1 = np.trapezoid(np.real(phase * (phi_mi / (1j * _U))[:, None]), dx=_DU, axis=0)
    p2 = 0.5 + int2 / np.pi
    p1 = 0.5 + int1 / np.pi
    return p1, p2


def heston_price(S, K, T, r, q, p: HestonParams, kind):
    K = np.atleast_1d(np.asarray(K, float))
    p1, p2 = _probabilities(S, K, T, r, q, p)
    call = S * np.exp(-q * T) * p1 - K * np.exp(-r * T) * p2
    if kind == "CE":
        out = call
    else:
        out = call - S * np.exp(-q * T) + K * np.exp(-r * T)   # put-call parity
    return out if out.size > 1 else float(out[0])


def heston_iv(S, K, T, r, q, p: HestonParams, kind):
    prices = np.atleast_1d(heston_price(S, np.atleast_1d(K), T, r, q, p, kind))
    Karr = np.atleast_1d(np.asarray(K, float))
    kinds = [kind] * Karr.size if isinstance(kind, str) else list(kind)
    return np.array([
        implied_vol(max(pr, 1e-6), S, k, T, r, q, kd)
        for pr, k, kd in zip(prices, Karr, kinds)
    ])


def heston_delta(S, K, T, r, q, p: HestonParams, kind, bump_frac=0.0025):
    """Model delta by central bump-and-revalue of the semi-analytic pricer."""
    h = S * bump_frac
    up = heston_price(S + h, K, T, r, q, p, kind)
    dn = heston_price(S - h, K, T, r, q, p, kind)
    return (np.asarray(up) - np.asarray(dn)) / (2 * h)


def calibrate_heston(S, K, T, r, q, kinds, ivs, feller: bool = False,
                     restarts: int = 3, seed: int = 7) -> HestonParams:
    """Least squares on implied vols. With feller=True the variance process is
    constrained to satisfy 2*kappa*theta >= sigma^2 (penalised boundary)."""
    K = np.asarray(K, float); ivs = np.asarray(ivs, float)
    atm = float(np.sqrt(np.mean(ivs**2)))
    starts = [
        HestonParams(2.0, atm**2, 0.60, -0.60, atm**2),
        HestonParams(0.8, atm**2, 1.20, -0.85, atm**2),
        HestonParams(5.0, atm**2, 0.35, -0.40, atm**2),
    ][:restarts]
    rng = np.random.default_rng(seed)

    lo = np.array([0.05, 1e-4, 0.02, -0.999, 1e-4])
    hi = np.array([30.0, 1.50, 3.00, 0.999, 1.50])

    # Vega-scaled price residuals: (model - market price) / market vega is a
    # first-order IV difference — the standard fast calibration objective.
    from .bs import bs_price, bs_vega
    mkt_price = np.array([float(bs_price(S, k, T, r, q, iv, kind))
                          for k, iv, kind in zip(K, ivs, kinds)])
    vega = np.array([float(bs_vega(S, k, T, r, q, iv))
                     for k, iv in zip(K, ivs)])
    vega = np.maximum(vega, 1.0)   # floor: far-wing vega -> 0 would explode residuals

    kinds = np.asarray(kinds)
    idx = {kd: np.where(kinds == kd)[0] for kd in ("CE", "PE") if (kinds == kd).any()}

    def resid(x):
        p = HestonParams(*x)
        model = np.empty(K.size)
        for kd, ii in idx.items():
            model[ii] = np.atleast_1d(heston_price(S, K[ii], T, r, q, p, kd))
        res = (model - mkt_price) / vega
        if feller:
            viol = max(0.0, p.sigma**2 - 2 * p.kappa * p.theta)
            res = np.append(res, 5.0 * viol)   # penalty in vol units
        return res

    best, best_cost = None, np.inf
    for s in starts:
        x0 = np.clip(s.as_array() + rng.normal(0, 1e-6, 5), lo + 1e-9, hi - 1e-9)
        sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                            xtol=1e-9, ftol=1e-9, max_nfev=250)
        if sol.cost < best_cost:
            best, best_cost = sol, sol.cost
    return HestonParams(*best.x)


def mc_heston_price(S, K, T, r, q, p: HestonParams, kind, n_paths=80_000,
                    n_steps=200, seed=3):
    """Full-truncation Euler Monte Carlo of the Heston SDE — used only as an
    independent cross-check of the semi-analytic pricer in the test suite."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2
    branches = []
    for sgn in (1.0, -1.0):   # antithetic pair
        branches.append([np.full(half, float(S)), np.full(half, p.v0), sgn])
    for _ in range(n_steps):
        z1 = rng.standard_normal(half); z2 = rng.standard_normal(half)
        w2 = p.rho * z1 + np.sqrt(1 - p.rho**2) * z2
        for b in branches:
            Sx, vx, sgn = b
            vp = np.maximum(vx, 0.0)
            sq = np.sqrt(vp * dt)
            Sx *= np.exp((r - q - 0.5 * vp) * dt + sgn * sq * z1)
            vx += p.kappa * (p.theta - vp) * dt + sgn * p.sigma * sq * w2
            np.maximum(vx, 0.0, out=vx)
    ST = np.concatenate([b[0] for b in branches])
    payoff = np.maximum(ST - K, 0.0) if kind == "CE" else np.maximum(K - ST, 0.0)
    disc = np.exp(-r * T)
    return float(disc * payoff.mean()), float(disc * payoff.std(ddof=1) / np.sqrt(payoff.size))
