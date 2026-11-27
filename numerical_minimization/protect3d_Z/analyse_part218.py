#!/usr/bin/env python3
"""part218 : chute libre +/-g (GRADA), houle wr=1,5,25. a/g de la demi-difference (partie impaire en g) par fenetres, acceleration locale, ajustement
x_odd(t) = 0.5 a t^2 + B (1 - cos(w t)) (mode interne mou ?), temoins g=0. Usage : python analyse_part218.py ../../resultats/part218"""
import json, os, sys, glob
import numpy as np
from scipy.optimize import least_squares
D = sys.argv[1] if len(sys.argv) > 1 else "../../resultats/part218"
G = 0.02
def load(tag):
    f = f"{D}/p218_{tag}/p218_{tag}_history.jsonl"
    if not os.path.exists(f): return None
    rows = [json.loads(l) for l in open(f) if l.strip()]; rows = [r for r in rows if "x4" in r]
    return np.array([r["t"] for r in rows]), np.array([r["x4"] for r in rows]), np.array([r.get("rho_min", np.nan) for r in rows])
def fit_osc(t, x):
    def res(p): return 0.5 * p[0] * t ** 2 + p[1] * (1 - np.cos(p[2] * t)) - x
    best = None
    for w0 in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
        r = least_squares(res, [x[-1] / 288, 0.0, w0])
        if best is None or r.cost < best.cost: best = r
    return best.x, np.sqrt(2 * best.cost / len(t))
out = {}
for wr in (1, 5, 25):
    for q in ("Q1", "Q2", "Q3"):
        p = load(f"{q}_wr{wr}_gp020"); m = load(f"{q}_wr{wr}_gm020")
        if p is None or m is None: continue
        t = p[0]; xo = (p[1] - m[1]) / 2; xe = (p[1] + m[1]) / 2 - (p[1][0] + m[1][0]) / 2
        row = dict(x_odd_end=float(xo[-1]), x_even_end=float(xe[-1]))
        for lo, hi in ((2, 10), (4, 14), (6, 18), (0, 24)):
            k = (t >= lo) & (t <= hi); row[f"a/g[{lo},{hi}]"] = float(abs(2 * np.polyfit(t[k], xo[k], 2)[0] / G))
        row["local"] = [round(float(abs(2 * np.polyfit(t[(t >= t0 - 3) & (t <= t0 + 3)], xo[(t >= t0 - 3) & (t <= t0 + 3)], 2)[0] / G)), 2) for t0 in (4, 8, 12, 16, 20)]
        pr, rms = fit_osc(t, -np.sign(xo[-1]) * np.abs(xo) if False else xo * np.sign(-xo[-1]) * -1 if False else xo)
        row["fit_osc(a,B,w)"] = [float(v) for v in pr]; row["fit_rms"] = float(rms)
        row["rho_min(t=0,24)"] = [float(p[2][0]), float(p[2][-1])]
        ij = f"{D}/p218_{q}_wr{wr}_gp020/inertie_p218_{q}_wr{wr}_gp020.json"
        if os.path.exists(ij):
            j = json.load(open(ij)); row["pred_a/g_xx"] = j["a_sur_g_pred_xx"]; row["M_xx/E"] = j["M_sur_E"][0]; row["E"] = j["E_moins_vide"]
        out[f"{q}_wr{wr}"] = row
for q in ("Q1", "Q2"):
    w = load(f"{q}_wr1_g000")
    if w is not None: out[f"temoin_{q}_g0"] = dict(derive_max=float(np.abs(w[1] - w[1][0]).max()))
print(json.dumps(out, indent=1))
json.dump(out, open(f"{D}/analyse_part218.json", "w"), indent=1)
