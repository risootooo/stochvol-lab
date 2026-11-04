"""Smile dynamics under a spot shock — the hedging-relevant difference
between models that can fit the same smile almost equally well.

Three scenarios for how the smile at a FIXED strike K responds when spot moves:
  * sticky-strike : IV(K) does not move at all (the market convention a naive
                    Black-Scholes hedger implicitly assumes). The pre-shock
                    market smile is represented by the slice's SVI fit.
  * Heston        : the calibrated model is repriced at the new spot; the
                    smile moves however the model's dynamics say it moves.
  * SABR backbone : the calibrated (alpha, beta, rho, nu) are held fixed and
                    the Hagan formula is evaluated at the new forward — the
                    smile slides along the SABR "backbone" dF ~ F^beta.

Deltas are each model's own: bump-and-revalue for Heston, Black-Scholes
delta at the model's implied vol for SABR / sticky-strike (the market
practice for smile models).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bs import bs_delta, forward
from .data import DIV_YIELD, RISK_FREE
from .heston import HestonParams, heston_delta, heston_iv
from .sabr import SABRParams, sabr_iv
from .svimodel import SVIParams


def smile_dynamics(slc, heston_p: HestonParams, sabr_p: SABRParams,
                   svi_p: SVIParams, shock: float = -0.02, r=RISK_FREE, q=DIV_YIELD):
    """Returns (table, summary). Table: per-strike IVs before/after per model.
    Summary: the old-ATM call — IV change and delta under each scenario."""
    S0, T = slc.spot, slc.T
    S1 = S0 * (1 + shock)
    F0 = forward(S0, T, r, q); F1 = forward(S1, T, r, q)
    K = slc.strikes.to_numpy(float); kinds = slc.kinds.to_numpy()
    k_old = np.log(K / F0)

    def _h_ivs(Sx):
        out = np.empty(K.size)
        for kd in ("CE", "PE"):
            m = kinds == kd
            if m.any():
                out[m] = heston_iv(Sx, K[m], T, r, q, heston_p, kd)
        return out

    iv_h0 = _h_ivs(S0)
    iv_h1 = _h_ivs(S1)
    iv_s0 = sabr_iv(F0, K, T, sabr_p.alpha, sabr_p.beta, sabr_p.rho, sabr_p.nu)
    iv_s1 = sabr_iv(F1, K, T, sabr_p.alpha, sabr_p.beta, sabr_p.rho, sabr_p.nu)
    iv_sticky = svi_p.iv(k_old, T)     # unchanged after the shock, by construction

    table = pd.DataFrame({
        "strike": K, "log_moneyness": k_old,
        "iv_heston_before": iv_h0, "iv_heston_after": iv_h1,
        "iv_sabr_before": iv_s0, "iv_sabr_after": iv_s1,
        "iv_sticky": iv_sticky,
    })

    # The old-ATM traded call: strike nearest the pre-shock spot.
    K0 = float(K[np.argmin(np.abs(K - S0))])
    i0 = int(np.argmin(np.abs(K - K0)))
    sig = {
        "sticky": float(iv_sticky[i0]),
        "heston": float(iv_h1[i0]),
        "sabr": float(iv_s1[i0]),
    }
    sig_before = {
        "sticky": float(iv_sticky[i0]),
        "heston": float(iv_h0[i0]),
        "sabr": float(iv_s0[i0]),
    }
    delta = {
        "sticky": float(bs_delta(S1, K0, T, r, q, sig["sticky"], "CE")),
        "heston": float(heston_delta(S1, K0, T, r, q, heston_p, "CE")),
        "sabr": float(bs_delta(S1, K0, T, r, q, sig["sabr"], "CE")),
    }
    summary = {
        "shock": shock, "spot_before": S0, "spot_after": S1, "atm_strike": K0,
        "iv_before": sig_before, "iv_after": sig,
        "iv_change_vol_pts": {m: (sig[m] - sig_before[m]) * 100 for m in sig},
        "delta_after": delta,
        "delta_spread": max(delta.values()) - min(delta.values()),
    }
    return table, summary
