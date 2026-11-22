"""Analyse part199 : x4(t) du noeud, ajustement x = x0 + v0 t + a t^2/2 pour t>=TMIN. usage : python analyse_part199.py [racine]"""
import json, sys, numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else '../../resultats/part199'
TMIN = 6.0
for tag, g in (('g0', 0.0), ('gp10', 0.01), ('gp05', 0.005), ('gm10', -0.01)):
    try: H = [json.loads(l) for l in open(f'{R}/p199_{tag}/p199_{tag}_history.jsonl')]
    except FileNotFoundError: print(tag, 'absent'); continue
    t = np.array([d['t'] for d in H]); x = np.array([d['x4'] for d in H]); y = np.array([d['y4'] for d in H]); z = np.array([d['z4'] for d in H])
    m = t >= TMIN
    c = np.polyfit(t[m], x[m], 2); a = 2 * c[0]
    print(f'{tag}: g={g:+.3f} t_max={t.max():.1f} x4(0)={x[0]:+.3f} x4(fin)={x[-1]:+.3f} dy={y[-1]-y[0]:+.3f} dz={z[-1]-z[0]:+.3f} | a(x4) = {a:+.5f}' + (f' | a/g = {a/g:+.3f} -> gamma = g/|a|-1 = {abs(g/a)-1:+.2f}' if g else ''))
