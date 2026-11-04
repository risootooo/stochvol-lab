"""Build docs/dashboard.html from docs/results.json (produced by main.py).
Self-contained: inline CSS + inline SVG, zero external references.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "docs" / "results.json").read_text())

W, H, PAD = 860, 300, 46
COLORS = {"market": "#e8e8e8", "SVI": "#8b93a7", "Heston": "#f0b429",
          "SABR b=0.5": "#38bdf8", "SABR b=1.0": "#a78bfa", "Heston (Feller)": "#f472b6"}


def _scale(vals, lo=None, hi=None):
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    pad = (hi - lo) * 0.08 or 1.0
    return lo - pad, hi + pad


def line_chart(series: dict[str, tuple[list, list]], xlabel: str, ylabel: str,
               points: tuple[list, list] | None = None) -> str:
    xs_all = [x for xs, _ in series.values() for x in xs]
    ys_all = [y for _, ys in series.values() for y in ys]
    if points:
        xs_all += points[0]; ys_all += points[1]
    x0, x1 = _scale(xs_all); y0, y1 = _scale(ys_all)

    def X(x): return PAD + (x - x0) / (x1 - x0) * (W - 2 * PAD)
    def Y(y): return H - PAD - (y - y0) / (y1 - y0) * (H - 2 * PAD)

    out = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        yv = y0 + frac * (y1 - y0)
        out.append(f'<line x1="{PAD}" y1="{Y(yv):.1f}" x2="{W-PAD}" y2="{Y(yv):.1f}" stroke="#232833"/>')
        out.append(f'<text x="{PAD-6}" y="{Y(yv)+4:.1f}" fill="#8b93a7" font-size="10" text-anchor="end">{yv:.2f}</text>')
        xv = x0 + frac * (x1 - x0)
        out.append(f'<text x="{X(xv):.1f}" y="{H-PAD+16}" fill="#8b93a7" font-size="10" text-anchor="middle">{xv:.2f}</text>')
    out.append(f'<text x="{W/2}" y="{H-4}" fill="#8b93a7" font-size="11" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="12" y="{H/2}" fill="#8b93a7" font-size="11" transform="rotate(-90 12 {H/2})" text-anchor="middle">{ylabel}</text>')
    if points:
        for x, y in zip(*points):
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.2" fill="{COLORS["market"]}" opacity="0.85"/>')
    for name, (xs, ys) in series.items():
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COLORS.get(name, "#fff")}" stroke-width="2"/>')
    lx = PAD + 8
    for name in series:
        out.append(f'<rect x="{lx}" y="8" width="10" height="3" fill="{COLORS.get(name, "#fff")}"/>'
                   f'<text x="{lx+14}" y="13" fill="#c9d1d9" font-size="10">{name}</text>')
        lx += 14 + 9 * len(name) + 18
    if points:
        out.append(f'<rect x="{lx}" y="8" width="10" height="3" fill="{COLORS["market"]}"/>'
                   f'<text x="{lx+14}" y="13" fill="#c9d1d9" font-size="10">market IV</text>')
    out.append("</svg>")
    return "".join(out)


def bar_chart(rows: list[tuple[str, float]], unit="vol pts") -> str:
    mx = max(v for _, v in rows) * 1.15
    out = ['<div class="bars">']
    for name, v in rows:
        out.append(f'<div class="bar-row"><span class="bar-name">{name}</span>'
                   f'<div class="bar-track"><div class="bar-fill" style="width:{v/mx*100:.1f}%"></div></div>'
                   f'<span class="bar-val">{v:.2f} {unit}</span></div>')
    out.append("</div>")
    return "".join(out)


def main() -> None:
    html = ["""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>StochVol Lab — Heston & SABR calibrated to the real NIFTY smile</title>
<style>
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;padding:32px}
h1{color:#f0f6fc;font-size:26px;margin:0 0 4px} h2{color:#f0f6fc;font-size:18px;margin:28px 0 8px}
.sub{color:#8b93a7;margin-bottom:8px} .card{background:#161b22;border:1px solid #232833;border-radius:10px;padding:18px;margin:14px 0}
table{border-collapse:collapse;font-size:13px} td,th{border-bottom:1px solid #232833;padding:5px 10px;text-align:right}
th{color:#8b93a7} td:first-child,th:first-child{text-align:left}
.bars{margin:6px 0} .bar-row{display:flex;align-items:center;gap:10px;margin:4px 0;font-size:13px}
.bar-name{width:170px;color:#c9d1d9} .bar-track{flex:1;background:#232833;border-radius:4px;height:14px}
.bar-fill{background:#f0b429;border-radius:4px;height:14px} .bar-val{width:110px;color:#8b93a7}
.kpi{display:flex;gap:14px;flex-wrap:wrap} .kpi div{background:#161b22;border:1px solid #232833;border-radius:10px;padding:12px 16px}
.kpi b{color:#f0b429;font-size:19px;display:block} .kpi span{color:#8b93a7;font-size:12px}
svg{width:100%;height:auto;background:#0d1117}
</style></head><body>
<h1>StochVol Lab</h1>
<div class="sub">Heston &amp; SABR, implemented from scratch and calibrated to the <b>real NIFTY smile of 3 Nov 2026</b>
(NSE bhavcopy, spot 23,810.85, OTM settlement IVs only). All numbers below are produced by <code>main.py</code>.</div>
"""]

    expiries = [e for e in RES if e.startswith("20")]
    # KPI strip: best model per slice
    html.append('<div class="kpi">')
    for exp in expiries:
        sb = RES[exp]["scoreboard"]
        best = min((r for r in sb if r["model"] != "Heston (Feller)"), key=lambda r: r["rmse_vs_market"])
        html.append(f'<div><b>{best["model"]} · {best["rmse_vs_market"]:.2f} vol pts</b>'
                    f'<span>best RMSE — {exp} ({RES[exp]["dte"]} DTE)</span></div>')
    h = RES[expiries[0]]["params"]["heston"]
    html.append(f'<div><b>{"Feller holds" if h["feller"] > 0 else "Feller violated"}</b>'
                f'<span>Heston 6-DTE calibration (2κθ−σ² = {h["feller"]:+.3f})</span></div>')
    dyn = RES["dynamics"]["summary"]
    html.append(f'<div><b>{dyn["delta_spread"]:.3f}</b><span>delta spread across smile models, spot −2% (Case C)</span></div>')
    html.append("</div>")

    for exp in expiries:
        d = RES[exp]
        series = {m: (d["log_moneyness"], [v * 100 for v in ivs])
                  for m, ivs in d["model_iv"].items() if m != "Heston (Feller)"}
        chart = line_chart(series, "log-moneyness log(K/F)", "implied vol (%)",
                           points=(d["log_moneyness"], [v * 100 for v in d["market_iv"]]))
        html.append(f'<div class="card"><h2>Real smile &amp; model fits — {exp} ({d["dte"]} DTE)</h2>{chart}</div>')
        rows = [(f'{r["model"]}', r["rmse_vs_market"]) for r in d["scoreboard"]]
        html.append(f'<div class="card"><h2>Fit RMSE vs market — {exp}</h2>{bar_chart(rows)}</div>')

    t = RES["dynamics"]["table"]
    ks = [r["log_moneyness"] for r in t]
    dh = [(r["iv_heston_after"] - r["iv_heston_before"]) * 100 for r in t]
    ds = [(r["iv_sabr_after"] - r["iv_sabr_before"]) * 100 for r in t]
    chart = line_chart({"Heston": (ks, dh), "SABR b=0.5": (ks, ds)},
                       "log-moneyness log(K/F) (pre-shock)", "IV change at fixed strike (vol pts)")
    html.append(f'<div class="card"><h2>Case C — smile dynamics, spot −2% ({RES["dynamics"]["slice"]})</h2>'
                f'{chart}<div class="sub">Sticky-strike is the zero line by construction: the same smile, three '
                f'different hedges. Old-ATM strike {dyn["atm_strike"]:.0f}: deltas sticky '
                f'{dyn["delta_after"]["sticky"]:.3f} · Heston {dyn["delta_after"]["heston"]:.3f} · '
                f'SABR {dyn["delta_after"]["sabr"]:.3f}.</div></div>')

    html.append('<div class="card"><h2>Reading this honestly</h2><div class="sub">'
                'Settlement IVs contain a parity-break step at the money (puts price ~5–8 vol points above calls at '
                'adjacent strikes on the weekly slices) — every smooth model shares that residual, and it is a '
                'property of the data, not of any one model. Deep-wing weekly IVs come from 2–6 point premiums. '
                'Full methodology, parameter tables and limitations: see README.md and docs/CASES.md in the repo.'
                '</div></div>')
    html.append("</body></html>")
    dest = ROOT / "docs" / "dashboard.html"
    dest.write_text("".join(html))
    print(f"Wrote {dest} ({dest.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
