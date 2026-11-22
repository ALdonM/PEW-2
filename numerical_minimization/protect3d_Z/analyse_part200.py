"""Analyse part200. usage : python analyse_part200.py [racine]  (chute : x4(t) ; onde : retard de groupe vs part196b)"""
import json, sys, os, numpy as np
R = sys.argv[1] if len(sys.argv) > 1 else '../../resultats/part200'
def hist(tag): return [json.loads(l) for l in open(f'{R}/p200_{tag}/p200_{tag}_history.jsonl')]
print('=== chute libre (KF=1) ===')
for tag, g in (('f_g0', 0.0), ('f_gp10', 0.01), ('f_gp05', 0.005)):
    try: H = hist(tag)
    except FileNotFoundError: print(tag, 'absent'); continue
    t = np.array([d['t'] for d in H]); x = np.array([d['x4'] for d in H]); E = np.array([d.get('Ekm', 0) + d['E2'] + d['E4'] + d['EV'] for d in H])
    out = f'{tag}: g={g:+.3f} t_max={t.max():.1f} x4(fin)={x[-1]:+.3f} | derive E {E[-1]-E[0]:+.1f}'
    if g:
        for lo, hi in ((2, 12), (4, 16)):
            m = (t >= lo) & (t <= hi)
            if m.sum() >= 3: a = 2 * np.polyfit(t[m], x[m], 2)[0]; out += f' | a/g[{lo},{hi}]={a/g:+.2f} gamma=g/|a|-1={abs(g/a)-1:+.2f}'
    print(out)
print('=== retard d onde k=3 (rho_vac=1) ; reference part196b : 2,97/2,80/2,08/1,21/0,71 ===')
N = 96; h = 0.5; k = 3.0; xs = (np.arange(N) - N / 2 + .5) * h; x = xs[:, None, None]; Y, Zc = np.meshgrid(xs, xs, indexing='ij'); b = np.sqrt(Y**2 + Zc**2)
def Lf(f): return np.load(f).astype(float) if os.path.exists(f) else None
def Zs(p): return p[0] + 1j * p[1], p[2] + 1j * p[3]
def chi(P, Q):
    a1, a2 = Zs(P); b1, b2 = Zs(Q); return np.angle(np.conj(b1) * a1 + np.conj(b2) * a2)
kx = 2 * np.pi * np.fft.fftfreq(N, h)
def env(c):
    F = np.fft.fft(c, axis=0); F[~((kx > 2.0) & (kx < 4.0))] = 0; return np.abs(np.fft.ifft(F, axis=0))
K0 = Lf('../../resultats/part196b/p196b_r100_K0/p196b_r100_K0_champ_final.npy'); V0 = Lf('../../resultats/part196b/p196b_r100_Vw/p196b_r100_Vw_champ_final.npy')
vac = np.zeros_like(K0); vac[0] = 1.0; cV = chi(V0, vac)
for tag in ('w_Kw', 'w_Kb'):
    K = Lf(f'{R}/p200_{tag}/p200_{tag}_champ_final.npy')
    if K is None: print(tag, 'absent'); continue
    cK = chi(K, K0); print(tag)
    for a_, c_ in [(0, 3), (3, 5), (5, 7), (7, 9), (9, 11)]:
        m = (b >= a_) & (b < c_); eK, eV = env(cK)**2, env(cV)**2
        gK = (eK[:, m].mean(1) * xs).sum() / eK[:, m].mean(1).sum(); gV = (eV[:, m].mean(1) * xs).sum() / eV[:, m].mean(1).sum()
        win = ((x > 2.5) & (x < 14)).astype(float); zK = (cK * np.exp(-1j * k * x) * win)[:, m].sum(); zV = (cV * np.exp(-1j * k * x) * win)[:, m].sum()
        print(f'  b {a_}-{c_}: retard de groupe {gV-gK:+.2f} | amplitude relative {abs(zK)/abs(zV):.3f}')
