"""Fit-quality reporting: every model is scored in implied-vol space against
both the raw market IVs and the SVI smooth of the same slice.
Errors are reported in volatility points (1 vol pt = 0.01 of IV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_stats(model_iv, market_iv, strikes) -> dict:
    err = (np.asarray(model_iv) - np.asarray(market_iv)) * 100.0   # vol points
    i = int(np.argmax(np.abs(err)))
    return {
        "rmse_vol_pts": float(np.sqrt(np.mean(err**2))),
        "max_abs_vol_pts": float(np.abs(err[i])),
        "worst_strike": float(np.asarray(strikes)[i]),
        "mean_bias_vol_pts": float(np.mean(err)),
    }


def scoreboard_row(slice_label: str, model: str, model_iv, market_iv, svi_iv, strikes) -> dict:
    row = {"slice": slice_label, "model": model}
    for tag, ref in (("vs_market", market_iv), ("vs_svi", svi_iv)):
        s = fit_stats(model_iv, ref, strikes)
        row[f"rmse_{tag}"] = s["rmse_vol_pts"]
        row[f"max_{tag}"] = s["max_abs_vol_pts"]
        if tag == "vs_market":
            row["worst_strike"] = s["worst_strike"]
            row["bias_vol_pts"] = s["mean_bias_vol_pts"]
    return row


def scoreboard_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = ["slice", "model", "rmse_vs_market", "max_vs_market", "worst_strike",
            "bias_vol_pts", "rmse_vs_svi", "max_vs_svi"]
    return df[cols].round(3)
