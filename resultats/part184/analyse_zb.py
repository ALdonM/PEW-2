#!/usr/bin/env python3
"""Recalcul independant depuis les journaux jsonl : charge totale (t=0), t_ann (Qabs<0.5 interpole),
qL/qR a des temps donnes, et bilan QPROF (qx, qfar, qtot, Eel) si present.
Usage: python3 -I analyse_zb.py fichier.log [...]"""
import json, sys, re, os

def load(p):
    rows = []
    for l in open(p):
        l = l.strip()
        if l.startswith('{'):
            try: rows.append(json.loads(l))
            except ValueError: pass
    return rows

def t_ann(rows, thr=0.5):
    for a, b in zip(rows, rows[1:]):
        if a['Qabs'] >= thr > b['Qabs']:
            return a['t'] + (a['Qabs'] - thr) / (a['Qabs'] - b['Qabs']) * (b['t'] - a['t'])
    return None

def at(rows, t):
    return min(rows, key=lambda r: abs(r['t'] - t))

for p in sys.argv[1:]:
    r = load(p)
    if not r: print(os.path.basename(p), 'VIDE'); continue
    r0 = r[0]; Qt = (abs(r0['qL']) + abs(r0['qR'])) / 2; Qs = r0['qL'] + r0['qR']
    ta = t_ann(r)
    s = f"{os.path.basename(p):28s} n={len(r):3d} tmax={r[-1]['t']:5.1f} |q|moy0={Qt:7.1f} qL+qR={Qs:7.1f} Qabs0={r0['Qabs']:.2f} t_ann={'%.2f'%ta if ta else 'non'}"
    if 'Econs' in r0: s += f" dEcons={r[-1]['Econs']-r0['Econs']:+.3f}"
    print(s)
    if 'qx' in r0:
        for t in (0, 5, 10, 15, 20):
            q = at(r, t)
            if abs(q['t'] - t) > .6: continue
            print(f"   t={q['t']:5.1f} qL={q['qL']:7.2f} qR={q['qR']:7.2f} qtot={q.get('qtot',float('nan')):8.2f} qfar={q.get('qfar',float('nan')):7.2f} Eel={q.get('Eel',float('nan')):9.2f} qx={q['qx']}")
