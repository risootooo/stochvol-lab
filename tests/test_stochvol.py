"""StochVol Lab test suite. Run: /tmp/venv-test/bin/python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stochvol.bs import bs_price, implied_vol          # noqa: E402
from stochvol.heston import (HestonParams, calibrate_heston, heston_iv,  # noqa: E402
                             heston_price, mc_heston_price)
from stochvol.sabr import SABRParams, calibrate_sabr, sabr_alpha_from_atm, sabr_iv  # noqa: E402
from stochvol.svimodel import SVIParams, fit_svi       # noqa: E402
from stochvol.dynamics import smile_dynamics           # noqa: E402
from stochvol.data import Slice, load_slices           # noqa: E402

S, R, Q = 23810.85, 0.065, 0.013


# ---------------- Black-Scholes base ----------------

def test_bs_put_call_parity():
    c = float(bs_price(S, 24000, 0.1, R, Q, 0.15, "CE"))
    p = float(bs_price(S, 24000, 0.1, R, Q, 0.15, "PE"))
    assert c - p == pytest.approx(S * np.exp(-Q * 0.1) - 24000 * np.exp(-R * 0.1), abs=1e-8)


def test_bs_iv_roundtrip():
    px = float(bs_price(S, 22000, 0.05, R, Q, 0.23, "PE"))
    assert implied_vol(px, S, 22000, 0.05, R, Q, "PE") == pytest.approx(0.23, abs=1e-8)


def test_iv_rejects_arbitrage_violating_price():
    assert np.isnan(implied_vol(0.5, S, 20000, 0.05, R, Q, "CE"))  # below intrinsic


# ---------------- Heston ----------------

def test_heston_collapses_to_bs_when_vol_of_vol_zero():
    p = HestonParams(kappa=3.0, theta=0.0225, sigma=1e-4, rho=-0.5, v0=0.0225)
    for kind in ("CE", "PE"):
        hv = float(heston_price(S, 24000, 0.05, R, Q, p, kind))
        bv = float(bs_price(S, 24000, 0.05, R, Q, 0.15, kind))
        assert hv == pytest.approx(bv, abs=0.05)


def test_heston_put_call_parity():
    p = HestonParams(2.0, 0.03, 0.6, -0.7, 0.025)
    c = float(heston_price(S, 23000, 0.08, R, Q, p, "CE"))
    put = float(heston_price(S, 23000, 0.08, R, Q, p, "PE"))
    assert c - put == pytest.approx(S * np.exp(-Q * 0.08) - 23000 * np.exp(-R * 0.08), abs=0.05)


def test_heston_mc_crosscheck():
    p = HestonParams(2.0, 0.03, 0.6, -0.7, 0.025)
    sa = float(heston_price(S, 24000, 0.05, R, Q, p, "CE"))
    mc, se = mc_heston_price(S, 24000, 0.05, R, Q, p, "CE", n_paths=60_000, n_steps=100)
    assert abs(sa - mc) < 4 * se


def test_heston_negative_rho_makes_downside_skew():
    p = HestonParams(2.0, 0.0225, 0.7, -0.8, 0.0225)
    iv_otm_put = heston_iv(S, np.array([21000.0]), 0.1, R, Q, p, "PE")[0]
    iv_otm_call = heston_iv(S, np.array([26500.0]), 0.1, R, Q, p, "CE")[0]
    assert iv_otm_put > iv_otm_call


def test_heston_feller_property():
    ok = HestonParams(2.0, 0.04, 0.3, -0.5, 0.03)
    bad = HestonParams(0.5, 0.01, 0.9, -0.5, 0.03)
    assert ok.feller_ok and not bad.feller_ok
    assert ok.feller == pytest.approx(2 * 2.0 * 0.04 - 0.09)


def _ivs_in_order(params, K, kinds, T):
    out = np.empty(K.size)
    for kd in ("PE", "CE"):
        m = kinds == kd
        out[m] = heston_iv(S, K[m], T, R, Q, params, kd)
    return out


def test_heston_calibration_deterministic():
    K = np.linspace(21000, 26500, 12)
    kinds = np.where(K < S, "PE", "CE")
    true = HestonParams(1.8, 0.028, 0.55, -0.65, 0.024)
    ivs = _ivs_in_order(true, K, kinds, 0.1)
    a = calibrate_heston(S, K, 0.1, R, Q, kinds, ivs, restarts=2)
    b = calibrate_heston(S, K, 0.1, R, Q, kinds, ivs, restarts=2)
    assert np.allclose(a.as_array(), b.as_array(), atol=1e-8)
    fit = _ivs_in_order(a, K, kinds, 0.1)
    assert np.sqrt(np.mean((fit - ivs) ** 2)) < 0.002   # recovers the smile


# ---------------- SABR ----------------

def test_sabr_atm_limit_matches_alpha_scaling():
    F, T = 24000.0, 0.25
    for beta in (0.5, 1.0):
        a = sabr_alpha_from_atm(F, T, 0.14, beta, -0.4, 0.8)
        atm = float(sabr_iv(F, np.array([F]), T, a, beta, -0.4, 0.8)[0])
        assert atm == pytest.approx(0.14, abs=1e-9)


def test_sabr_recovers_synthetic_smile():
    F, T = 24000.0, 0.25
    K = F * np.exp(np.linspace(-0.12, 0.12, 15))
    true = SABRParams(alpha=18.0, beta=0.5, rho=-0.45, nu=1.1)
    ivs = sabr_iv(F, K, T, true.alpha, true.beta, true.rho, true.nu)
    fit = calibrate_sabr(F, K, T, ivs, beta=0.5)
    model = sabr_iv(F, K, T, fit.alpha, fit.beta, fit.rho, fit.nu)
    assert np.sqrt(np.mean((model - ivs) ** 2)) < 1e-4
    assert fit.rho == pytest.approx(true.rho, abs=0.05)
    assert fit.nu == pytest.approx(true.nu, abs=0.15)


def test_sabr_negative_rho_put_skew():
    F, T = 24000.0, 0.25
    iv_put = float(sabr_iv(F, np.array([22000.0]), T, 18.0, 0.5, -0.5, 1.0)[0])
    iv_call = float(sabr_iv(F, np.array([26000.0]), T, 18.0, 0.5, -0.5, 1.0)[0])
    assert iv_put > iv_call


def test_sabr_beta_backbone_direction():
    # The SABR backbone: ATM implied vol behaves like alpha / F^(1-beta).
    # beta = 1  -> ATM vol is invariant when the forward moves (alpha held);
    # beta = 0.5 -> ATM vol rises ~1% (relative) when the forward falls 2%.
    F, T = 24000.0, 0.25
    atm1 = lambda f: float(sabr_iv(f, np.array([f]), T, 0.15, 1.0, -0.4, 0.9)[0])
    assert atm1(F * 0.98) == pytest.approx(atm1(F), rel=1e-9)
    atmh = lambda f: float(sabr_iv(f, np.array([f]), T, 18.0, 0.5, -0.4, 0.9)[0])
    assert atmh(F * 0.98) > atmh(F) * 1.005


# ---------------- SVI ----------------

def test_svi_recovers_synthetic_smile():
    T = 0.1
    k = np.linspace(-0.15, 0.10, 21)
    true = SVIParams(a=0.002, b=0.12, rho=-0.6, m=-0.01, s=0.06)
    ivs = true.iv(k, T)
    fit = fit_svi(k, ivs, T)
    assert np.sqrt(np.mean((fit.iv(k, T) - ivs) ** 2)) < 1e-5


def test_svi_wings_rise():
    p = SVIParams(a=0.002, b=0.12, rho=-0.6, m=0.0, s=0.06)
    assert p.w(np.array([0.15]))[0] > p.w(np.array([0.0]))[0]


# ---------------- Dynamics & real data ----------------

def _synthetic_slice():
    import pandas as pd
    T = 0.05
    F = S * np.exp((R - Q) * T)
    strikes = np.arange(22500, 25200, 100.0)
    kinds = np.where(strikes < S, "PE", "CE")
    ks = np.log(strikes / F)
    svi = SVIParams(a=0.0011, b=0.10, rho=-0.55, m=-0.005, s=0.05)
    return Slice(expiry="2099-01-01", dte=18, T=T, spot=S, forward=F,
                 strikes=pd.Series(strikes), kinds=pd.Series(kinds),
                 ivs=pd.Series(svi.iv(ks, T)), ks=pd.Series(ks))


def test_dynamics_sticky_strike_iv_unchanged():
    slc = _synthetic_slice()
    h = HestonParams(2.0, 0.02, 0.5, -0.6, 0.02)
    s = SABRParams(alpha=16.0, beta=0.5, rho=-0.4, nu=0.9)
    svi = SVIParams(a=0.0011, b=0.10, rho=-0.55, m=-0.005, s=0.05)
    table, summary = smile_dynamics(slc, h, s, svi, shock=-0.02)
    assert np.allclose(table.iv_sticky, svi.iv(table.log_moneyness, slc.T))
    assert summary["iv_change_vol_pts"]["sticky"] == pytest.approx(0.0, abs=1e-9)
    # The dynamic models DO move the fixed-strike IV — and in this steep-skew
    # configuration both move it DOWN when spot falls: the strike slides along
    # the skew faster than any level effect (the same sign the real-data
    # Case C produces). Sticky-strike misses the move entirely.
    assert summary["iv_change_vol_pts"]["heston"] < 0
    assert summary["iv_change_vol_pts"]["sabr"] < 0
    assert summary["delta_spread"] > 0


def test_real_slices_load():
    slices = load_slices()
    assert [s.dte for s in slices] == [6, 20, 56]
    for s in slices:
        assert len(s.strikes) > 50
        assert s.ivs.between(0.03, 1.5).all()
        assert s.spot == pytest.approx(23810.85)
