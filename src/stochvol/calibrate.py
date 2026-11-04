"""One-call calibration driver: fits SVI, SABR (beta 0.5 / 1.0), Heston and
Feller-constrained Heston to every real slice and builds the scoreboard.
Shared by main.py, the dashboard generator and the cases.
"""
from __future__ import annotations

import numpy as np

from .data import DIV_YIELD, RISK_FREE, Slice
from .fitreport import scoreboard_frame, scoreboard_row
from .heston import calibrate_heston, heston_iv
from .sabr import calibrate_sabr, sabr_iv
from .svimodel import fit_svi


def _heston_ivs(slc: Slice, params, r, q) -> np.ndarray:
    K = slc.strikes.to_numpy(float); kinds = slc.kinds.to_numpy()
    out = np.empty(K.size)
    for kd in ("CE", "PE"):
        m = kinds == kd
        if m.any():
            out[m] = heston_iv(slc.spot, K[m], slc.T, r, q, params, kd)
    return out


def calibrate_slice(slc: Slice, r=RISK_FREE, q=DIV_YIELD, with_feller: bool = True) -> dict:
    K = slc.strikes.to_numpy(float); kinds = slc.kinds.to_numpy()
    ivs = slc.ivs.to_numpy(float); ks = slc.ks.to_numpy(float); F = slc.forward

    svi = fit_svi(ks, ivs, slc.T)
    svi_iv = svi.iv(ks, slc.T)
    sabr05 = calibrate_sabr(F, K, slc.T, ivs, beta=0.5)
    sabr10 = calibrate_sabr(F, K, slc.T, ivs, beta=1.0)
    heston = calibrate_heston(slc.spot, K, slc.T, r, q, kinds, ivs)
    out = {
        "slice": slc, "svi": svi, "sabr05": sabr05, "sabr10": sabr10, "heston": heston,
        "model_ivs": {
            "SVI": svi_iv,
            "SABR b=0.5": sabr_iv(F, K, slc.T, sabr05.alpha, sabr05.beta, sabr05.rho, sabr05.nu),
            "SABR b=1.0": sabr_iv(F, K, slc.T, sabr10.alpha, sabr10.beta, sabr10.rho, sabr10.nu),
            "Heston": _heston_ivs(slc, heston, r, q),
        },
    }
    if with_feller:
        hf = calibrate_heston(slc.spot, K, slc.T, r, q, kinds, ivs, feller=True, restarts=2)
        out["heston_feller"] = hf
        out["model_ivs"]["Heston (Feller)"] = _heston_ivs(slc, hf, r, q)
    rows = [scoreboard_row(slc.label, name, miv, ivs, svi_iv, K)
            for name, miv in out["model_ivs"].items()]
    out["scoreboard"] = scoreboard_frame(rows)
    return out
