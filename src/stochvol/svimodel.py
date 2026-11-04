"""Raw SVI (Gatheral) — the desk-standard smile interpolation, used here as
the smooth benchmark curve each slice is also scored against.

Total implied variance w(k) = a + b * [ rho (k - m) + sqrt((k - m)^2 + s^2) ],
k = log(K/F). Implied vol = sqrt(w / T).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    s: float

    def w(self, k):
        k = np.asarray(k, float)
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.s**2))

    def iv(self, k, T):
        return np.sqrt(np.maximum(self.w(k), 1e-10) / T)


def fit_svi(k, ivs, T) -> SVIParams:
    """Least squares on implied vols (same objective as the other models, so
    the scoreboard compares like with like)."""
    k = np.asarray(k, float); ivs = np.asarray(ivs, float)
    atm_w = float(ivs[np.argmin(np.abs(k))] ** 2 * T)
    starts = [
        np.array([atm_w * 0.5, 0.10, -0.5, 0.0, 0.05]),
        np.array([atm_w * 0.2, 0.30, -0.8, -0.02, 0.10]),
    ]
    lo = np.array([-0.05, 0.0, -0.999, -0.5, 1e-4])
    hi = np.array([2.0, 5.0, 0.999, 0.5, 1.0])

    def resid(x):
        p = SVIParams(*x)
        return p.iv(k, T) - ivs

    best, best_cost = None, np.inf
    for x0 in starts:
        sol = least_squares(resid, np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                            method="trf", xtol=1e-12, ftol=1e-12, max_nfev=400)
        if sol.cost < best_cost:
            best, best_cost = sol, sol.cost
    return SVIParams(*best.x)
