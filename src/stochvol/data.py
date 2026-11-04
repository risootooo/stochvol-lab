"""Load the extracted real NIFTY slices of 3 Nov 2026 (see scripts/extract_data.py)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "slices_2026-11-03.csv"
RISK_FREE = 0.065   # as used across the sibling labs (greeks-lab / vol-surface-atlas)
DIV_YIELD = 0.013   # stated assumption: NIFTY dividend yield ~1.3%


@dataclass
class Slice:
    expiry: str
    dte: int
    T: float
    spot: float
    forward: float
    strikes: pd.Series
    kinds: pd.Series
    ivs: pd.Series
    ks: pd.Series          # log-moneyness log(K/F)

    @property
    def label(self) -> str:
        return f"{self.expiry} ({self.dte} DTE)"


def load_slices(fit_only: bool = True) -> list[Slice]:
    df = pd.read_csv(DATA_FILE)
    if fit_only:
        df = df[df.in_fit]
    slices = []
    for expiry, g in df.groupby("expiry"):
        g = g.sort_values("strike")
        slices.append(Slice(
            expiry=expiry, dte=int(g.dte.iloc[0]), T=float(g["T"].iloc[0]),
            spot=float(g.spot.iloc[0]), forward=float(g.forward.iloc[0]),
            strikes=g.strike, kinds=g.kind, ivs=g.iv, ks=g.log_moneyness,
        ))
    return sorted(slices, key=lambda s: s.T)
