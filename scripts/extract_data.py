"""Extract the real NIFTY option slices of 3 Nov 2026 used by StochVol Lab.

Sources (both real, read-only):
  1. vol-surface-atlas IV panel (premiums already extracted from NSE bhavcopy)
     -> expiries 2026-11-09 (6 DTE) and 2026-11-23 (20 DTE).
  2. The raw NSE bhavcopy file BhavCopy_NSE_FO_0_0_0_20261103_F_0000.csv
     -> expiry 2026-12-29 (56 DTE, the liquid monthly), settlement prices.

All implied vols are recomputed here with this repo's own Black-Scholes solver
(spot 23,810.85 from the file, r = 6.5%, q = 1.3%), OTM options only:
ITM settlement prints routinely violate no-arbitrage bounds (documented in
greeks-lab / vol-surface-atlas), so they are excluded by construction.

Filters applied (all counts printed): premium >= 1.0 index point, open
interest > 0, |log(K/F)| <= 0.20, implied vol finite and within [0.03, 1.50],
plus a spike filter: within each (expiry, kind) run sorted by strike, a point
whose IV deviates from the rolling 5-strike median by more than 2 vol points
is a bad settlement print (the raw file contains isolated prints ~8 vol points
off both neighbours) and is dropped. The column `in_fit` marks the calibration
window |log(K/F)| <= 0.15.

Output: data/slices_2026-11-03.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stochvol.bs import forward, implied_vol  # noqa: E402

PANEL = Path.home() / "workspace/github-build-2026-11-04/vol-surface-atlas/data/iv_panel_sample.csv"
BHAV = Path.home() / "workspace/trading/india-options-paper/data/bhavcopy/BhavCopy_NSE_FO_0_0_0_20261103_F_0000.csv"
TRADE_DATE = "2026-11-03"
SPOT = 23810.85
R, Q = 0.065, 0.013
PANEL_EXPIRIES = ["2026-11-09", "2026-11-23"]
RAW_EXPIRY = "2026-12-29"


def _finish(df: pd.DataFrame, src: str) -> pd.DataFrame:
    df = df.copy()
    F = forward(SPOT, df["T"].iloc[0], R, Q)
    df["forward"] = F
    df["log_moneyness"] = np.log(df["strike"] / F)
    df = df[(df.premium >= 1.0) & (df.oi > 0) & (df.log_moneyness.abs() <= 0.20)]
    df["iv"] = [
        implied_vol(p, SPOT, k, t, R, Q, kind)
        for p, k, t, kind in zip(df.premium, df.strike, df["T"], df.kind)
    ]
    n_bad = int(df.iv.isna().sum())
    df = df[df.iv.between(0.03, 1.50)].reset_index(drop=True)
    # Spike filter: isolated bad settlement prints vs the local rolling median.
    keep = pd.Series(True, index=df.index)
    for _, idx in df.groupby(["expiry", "kind"]).groups.items():
        g = df.loc[idx].sort_values("strike")
        med = g.iv.rolling(5, center=True, min_periods=3).median()
        keep.loc[g.index] = (g.iv - med).abs() <= 0.02
    n_spike = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    df["in_fit"] = df.log_moneyness.abs() <= 0.15
    print(f"  {src}: kept {len(df)} OTM points ({n_bad} not invertible / out of vol band, {n_spike} spike prints dropped)")
    return df


def from_panel() -> pd.DataFrame:
    p = pd.read_csv(PANEL)
    p = p[(p.trade_date == TRADE_DATE) & (p.expiry.isin(PANEL_EXPIRIES))]
    out = p.rename(columns={"days_to_expiry": "dte"})[
        ["expiry", "dte", "T", "strike", "kind", "premium", "oi", "volume"]
    ]
    out["spot"] = SPOT
    return _finish(out, "panel " + ", ".join(PANEL_EXPIRIES))


def from_bhavcopy() -> pd.DataFrame:
    df = pd.read_csv(
        BHAV,
        usecols=["FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric", "OptnTp",
                 "SttlmPric", "OpnIntrst", "TtlTradgVol"],
    )
    n = df[(df.TckrSymb == "NIFTY") & (df.FinInstrmTp == "IDO") & (df.XpryDt == RAW_EXPIRY)]
    otm = n[((n.OptnTp == "CE") & (n.StrkPric >= SPOT)) | ((n.OptnTp == "PE") & (n.StrkPric <= SPOT))]
    dte = (pd.Timestamp(RAW_EXPIRY) - pd.Timestamp(TRADE_DATE)).days
    out = pd.DataFrame({
        "expiry": RAW_EXPIRY, "dte": dte, "T": dte / 365.0,
        "strike": otm.StrkPric.values, "kind": otm.OptnTp.values,
        "premium": otm.SttlmPric.values, "oi": otm.OpnIntrst.values,
        "volume": otm.TtlTradgVol.values, "spot": SPOT,
    })
    return _finish(out, f"bhavcopy {RAW_EXPIRY}")


def main() -> None:
    print(f"Extracting real NIFTY slices, trade date {TRADE_DATE}, spot {SPOT}")
    data = pd.concat([from_panel(), from_bhavcopy()], ignore_index=True)
    data.insert(0, "trade_date", TRADE_DATE)
    data = data.sort_values(["expiry", "strike", "kind"]).reset_index(drop=True)
    dest = ROOT / "data" / "slices_2026-11-03.csv"
    data.to_csv(dest, index=False)
    print(f"Wrote {dest} ({len(data)} rows)")
    print(data.groupby("expiry").agg(n=("iv", "size"), atm_iv=("iv", lambda s: s.iloc[(data.loc[s.index, 'log_moneyness']).abs().argmin()])))


if __name__ == "__main__":
    main()
