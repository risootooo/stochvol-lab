"""SABR model — Hagan et al. (2002) lognormal implied-volatility formula.

dF = alpha F^beta dW1,  dalpha = nu alpha dW2,  corr = rho.
The closed-form approximation below is the standard Hagan lognormal formula
(valid for the moderate strikes/maturities it is used on here; it is an
asymptotic expansion, not an exact price — one of the honest limitations).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float


def sabr_iv(F, K, T, alpha, beta, rho, nu):
    """Hagan lognormal implied vol. Vectorised over K."""
    F = float(F); K = np.asarray(K, float)
    logFK = np.log(F / K)
    FK = F * K
    z = (nu / alpha) * FK ** ((1 - beta) / 2) * logFK
    xz = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(logFK) < 1e-10, 1.0, z / np.where(np.abs(logFK) < 1e-10, 1.0, xz))
    denom = FK ** ((1 - beta) / 2) * (
        1 + (1 - beta) ** 2 / 24 * logFK**2 + (1 - beta) ** 4 / 1920 * logFK**4
    )
    corr = 1 + (
        (1 - beta) ** 2 / 24 * alpha**2 / FK ** (1 - beta)
        + 0.25 * rho * beta * nu * alpha / FK ** ((1 - beta) / 2)
        + (2 - 3 * rho**2) / 24 * nu**2
    ) * T
    return alpha * ratio / denom * corr


def sabr_alpha_from_atm(F, T, sigma_atm, beta, rho, nu):
    """Solve alpha from the ATM implied vol (the usual desk workflow: beta is
    chosen, rho/nu fitted to the skew, alpha implied by the ATM level)."""
    from scipy.optimize import brentq
    # alpha scales like F^(1-beta): for an index at ~24k and beta=0.5 the
    # calibrated alpha is O(15), not O(0.15). Bracket accordingly.
    hi = 10.0 * float(F) ** (1 - beta)
    f = lambda a: float(sabr_iv(F, np.array([F]), T, a, beta, rho, nu)[0]) - sigma_atm
    return brentq(f, 1e-6, hi, xtol=1e-12)


def calibrate_sabr(F, K, T, ivs, beta: float) -> SABRParams:
    """Fit (alpha, rho, nu) at fixed beta by least squares on implied vols,
    from two deterministic starts (flat and skewed) — best cost wins."""
    K = np.asarray(K, float); ivs = np.asarray(ivs, float)
    atm_iv = float(ivs[np.argmin(np.abs(np.log(K / F)))])
    starts = []
    for rho0, nu0 in ((-0.3, 0.5), (-0.7, 1.2)):
        a0 = sabr_alpha_from_atm(F, T, atm_iv, beta, rho0, nu0)
        starts.append(np.array([a0, rho0, nu0]))
    lo = np.array([1e-4, -0.999, 0.01]); hi = np.array([20.0 * float(F) ** (1 - beta), 0.999, 5.0])
    best, best_cost = None, np.inf
    for x0 in starts:
        sol = least_squares(lambda x: sabr_iv(F, K, T, x[0], beta, x[1], x[2]) - ivs,
                            np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                            method="trf", xtol=1e-12, ftol=1e-12, max_nfev=300)
        if sol.cost < best_cost:
            best, best_cost = sol, sol.cost
    return SABRParams(alpha=float(best.x[0]), beta=beta, rho=float(best.x[1]), nu=float(best.x[2]))
