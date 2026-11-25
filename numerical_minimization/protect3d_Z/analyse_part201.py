"""part201 : analyse (lancer avec python -I depuis la racine du depot : python -I numerical_minimization/protect3d_Z/analyse_part201.py)
Lit resultats/part201/p201_*/ ; sort t_ann (premier t avec Qabs < 0,5), pic de Ekm, derive de Ekm+E2+E4+EV, separation des pics en x (same)."""
import json, os, numpy as np
from scipy import ndimage
R = "resultats/part201"; H = 0.5
def hist(tag):
    f = f"{R}/p201_{tag}/p201_{tag}_history.jsonl"
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else None
def t_ann(m):
    for r in m:
        if r["Qabs"] < 0.5: return r["t"]
    return None
def cons(r): return r["Ekm"] + r["E2"] + r["E4"] + r["EV"]
def sep_x(tag):
    f = f"{R}/p201_{tag}/p201_{tag}_proj.npy"
    if not os.path.exists(f): return None
    P = np.load(f).astype(np.float32); out = []
    for e in P:
        prof = ndimage.gaussian_filter1d(e.sum(1), 2.0 / H, mode="wrap"); x = (np.arange(len(prof)) - len(prof) / 2 + 0.5) * H
        pk = [i for i in range(len(prof)) if prof[i] > prof[i - 1] and prof[i] >= prof[(i + 1) % len(prof)] and prof[i] > 0.3 * prof.max()]
        pk = sorted(pk, key=lambda i: -prof[i])[:2]
        out.append(abs(x[pk[0]] - x[pk[1]]) if len(pk) == 2 else 0.0)
    return out
for tag in ("opp_kf0", "opp_kf0_dt5", "opp_kf1", "opp_kf1_dt5", "same_kf0", "same_kf0_dt5", "same_kf1", "same_kf1_dt5"):
    m = hist(tag)
    if not m: print(tag, ": pas de resultats"); continue
    c0 = cons(m[0]); drift = max(abs(cons(r) - c0) for r in m) / c0
    print(f"{tag}: t_fin={m[-1]['t']}  t_ann={t_ann(m)}  Ekm_max={max(r['Ekm'] for r in m):.0f}  derive max Econs/E0={drift:.2%}  Qabs_fin={m[-1]['Qabs']}  cos_min_run_min={min(r['cos_min_run'] for r in m):.2f}")
a, b, c = (t_ann(hist(t) or []) for t in ("opp_kf0", "opp_kf1", "opp_kf1_dt5"))
if a and b: print(f"rapport t_ann(KF1)/t_ann(KF0) = {b / a:.2f}   (P0 : t_ann(KF0) dans [5,1 ; 6,3] ; P1 : [1,3 ; 2,3])")
if b and c: print(f"controle dt : t_ann dt=0,005 / dt=0,0025 = {c / b:.3f}   (P4 : 1 +- 0,05)")
for tag in ("same_kf0", "same_kf1"):
    s = sep_x(tag)
    if s: print(tag, "separation des pics en x a t=0, 6, 12, 20 :", [round(s[int(t / 0.5)], 1) for t in (0, 6, 12, 20) if int(t / 0.5) < len(s)])
