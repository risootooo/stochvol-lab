"""Black-Scholes-Merton primitives with continuous dividend yield.

Own implementation (numpy/scipy only). Conventions used across the lab:
  S spot, K strike, T years to expiry, r risk-free (cont.), q dividend yield (cont.),
  kind in {"CE", "PE"}. Prices in index points.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

FLOOR_PRICE = 1e-10


def forward(S: float, T: float, r: float, q: float) -> float:
    return S * np.exp((r - q) * T)


def bs_price(S, K, T, r, q, sigma, kind):
    """Vectorised Black-Scholes price."""
    S, K, sigma = np.asarray(S, float), np.asarray(K, float), np.asarray(sigma, float)
    if T <= 0:
        return np.maximum(S - K, 0.0) if kind == "CE" else np.maximum(K - S, 0.0)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    if kind == "CE":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_delta(S, K, T, r, q, sigma, kind):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if kind == "CE":
        return np.exp(-q * T) * norm.cdf(d1)
    return -np.exp(-q * T) * norm.cdf(-d1)


def bs_vega(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r, q, kind):
    """Implied vol from a price; NaN when the price violates no-arbitrage bounds
    (the documented settlement-print problem: ITM/stale prints are not invertible)."""
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    intrinsic = max(S * disc_q - K * disc_r, 0.0) if kind == "CE" else max(K * disc_r - S * disc_q, 0.0)
    upper = S * disc_q if kind == "CE" else K * disc_r
    if not (intrinsic - 1e-9 <= price <= upper) or price <= FLOOR_PRICE:
        return float("nan")
    try:
        return brentq(
            lambda s: float(bs_price(S, K, T, r, q, s, kind)) - price,
            1e-4, 3.0, xtol=1e-10, rtol=1e-10,
        )
    except (ValueError, RuntimeError):
        return float("nan")
