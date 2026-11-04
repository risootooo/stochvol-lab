"""StochVol Lab — calibrate Heston, SABR and SVI to the real NIFTY smile of
3 Nov 2026 and print the three case studies. Writes docs/results.json.

Run:  /tmp/venv-test/bin/python main.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from stochvol.calibrate import calibrate_slice          # noqa: E402
from stochvol.data import load_slices                    # noqa: E402
from stochvol.dynamics import smile_dynamics             # noqa: E402

pd.set_option("display.width", 130)


def main() -> None:
    slices = load_slices()
    results, all_rows = {}, []
    fits = {}
    for slc in slices:
        print(f"Calibrating {slc.label}  (n={len(slc.strikes)} real OTM points) ...")
        res = calibrate_slice(slc, with_feller=True)
        fits[slc.expiry] = res
        all_rows.append(res["scoreboard"])
        results[slc.expiry] = {
            "dte": slc.dte, "T": slc.T, "spot": slc.spot, "forward": slc.forward,
            "strikes": slc.strikes.tolist(), "kinds": slc.kinds.tolist(),
            "market_iv": slc.ivs.tolist(), "log_moneyness": slc.ks.tolist(),
            "model_iv": {k: np.asarray(v).tolist() for k, v in res["model_ivs"].items()},
            "scoreboard": res["scoreboard"].to_dict("records"),
            "params": {
                "heston": vars(res["heston"]) | {"feller": res["heston"].feller},
                "heston_feller": vars(res["heston_feller"]) | {"feller": res["heston_feller"].feller},
                "sabr05": vars(res["sabr05"]), "sabr10": vars(res["sabr10"]),
                "svi": vars(res["svi"]),
            },
        }

    board = pd.concat(all_rows, ignore_index=True)
    print("\n================ CASE A — calibration scoreboard (errors in vol points) ================")
    print(board.to_string(index=False))

    print("\n================ CASE B — Heston parameter / Feller audit ================")
    for exp, res in fits.items():
        h, hf = res["heston"], res["heston_feller"]
        sb = res["scoreboard"].set_index("model")
        print(f"{res['slice'].label}: kappa={h.kappa:.2f} theta={h.theta:.4f} (vol {np.sqrt(h.theta)*100:.1f}%) "
              f"sigma={h.sigma:.2f} rho={h.rho:+.3f} v0={h.v0:.4f} (vol {np.sqrt(h.v0)*100:.1f}%)")
        print(f"   Feller 2*kappa*theta - sigma^2 = {h.feller:+.4f}  -> {'HOLDS' if h.feller_ok else 'VIOLATED'}"
              f" | unconstrained RMSE {sb.loc['Heston', 'rmse_vs_market']:.3f} vol pts"
              f" | Feller-forced RMSE {sb.loc['Heston (Feller)', 'rmse_vs_market']:.3f} vol pts"
              f" (cost {sb.loc['Heston (Feller)', 'rmse_vs_market'] - sb.loc['Heston', 'rmse_vs_market']:+.3f})")

    print("\n================ CASE C — smile dynamics, spot -2% (20-DTE slice) ================")
    mid = fits[slices[1].expiry]
    table, summary = smile_dynamics(mid["slice"], mid["heston"], mid["sabr05"], mid["svi"], shock=-0.02)
    results["dynamics"] = {
        "slice": mid["slice"].label, "table": table.to_dict("records"), "summary": summary,
    }
    K0 = summary["atm_strike"]
    print(f"Old-ATM call strike {K0:.0f}; spot {summary['spot_before']:.2f} -> {summary['spot_after']:.2f}")
    print(f"{'model':<10} {'IV before':>10} {'IV after':>10} {'dIV (vol pts)':>14} {'delta after':>12}")
    for m in ("sticky", "heston", "sabr"):
        print(f"{m:<10} {summary['iv_before'][m]*100:>9.2f}% {summary['iv_after'][m]*100:>9.2f}% "
              f"{summary['iv_change_vol_pts'][m]:>14.2f} {summary['delta_after'][m]:>12.3f}")
    print(f"Delta spread across models: {summary['delta_spread']:.3f} "
          f"({summary['delta_spread']*100:.1f} delta points on the same option, same shock)")

    dest = ROOT / "docs" / "results.json"
    dest.write_text(json.dumps(results, indent=1))
    print(f"\nWrote {dest}")


if __name__ == "__main__":
    main()
