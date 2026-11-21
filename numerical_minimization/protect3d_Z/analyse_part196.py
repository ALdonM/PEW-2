"""Analyse part196 : retard de phase d'une onde libre de la porteuse. usage : python analyse_part196.py [racine]"""
import numpy as np, os, sys, json
R = sys.argv[1] if len(sys.argv) > 1 else '../../resultats/part196'
N = 96; h = 0.5; k = 3.0; xs = (np.arange(N) - N / 2 + .5) * h
x = xs[:, None, None]; Y, Zc = np.meshgrid(xs, xs, indexing='ij'); b = np.sqrt(Y**2 + Zc**2)
def L(t):
    f = f'{R}/p196_{t}/p196_{t}_champ_final.npy'
    return np.load(f).astype(float) if os.path.exists(f) else None
def Zs(p): return p[0] + 1j * p[1], p[2] + 1j * p[3]
def chi(P, Q):
    a1, a2 = Zs(P); b1, b2 = Zs(Q); return np.angle(np.conj(b1) * a1 + np.conj(b2) * a2)
bins = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 11)]
kx = 2 * np.pi * np.fft.fftfreq(N, h)
def env(c):
    F = np.fft.fft(c, axis=0); F[~((kx > 2.0) & (kx < 4.0))] = 0   # enveloppe du paquet (k entre 2 et 4)
    return np.abs(np.fft.ifft(F, axis=0))
for rn, rv in (('100', 1.0), ('050', 0.5)):
    K0 = L(f'r{rn}_K0')
    if K0 is None: print(rn, 'K0 absent'); continue
    vac = np.zeros_like(K0); vac[0] = rv
    rho = np.sqrt((K0**2).sum(0)) / rv; I = ((1 - rho) * h).sum(0)   # integrale de (1 - rho') dx sur toute la boite (colonnes y,z)
    for suf, lab in (('w', 'eps 0,05'), ('b', 'eps 0,025')):
        K = L(f'r{rn}_K{suf}'); V = L(f'r{rn}_V{suf}')
        if K is None or V is None: continue
        cK, cV = chi(K, K0), chi(V, vac); win = ((x > 2.5) & (x < 14)).astype(float)
        print(f'--- rho_vac={rv} {lab} (retard en longueur = arg/k ; positif = retard ; lambda/2pi = {1/k:.2f})')
        for a, c in bins:
            m = (b >= a) & (b < c)
            zK = (cK * np.exp(-1j * k * x) * win)[:, m].sum(); zV = (cV * np.exp(-1j * k * x) * win)[:, m].sum()
            d = np.angle(zK / zV) / k
            eK, eV = env(cK)**2, env(cV)**2; gK = (eK[:, m].mean(1) * xs).sum() / eK[:, m].mean(1).sum(); gV = (eV[:, m].mean(1) * xs).sum() / eV[:, m].mean(1).sum()
            print(f'  b {a}-{c}: retard de GROUPE (centre enveloppe V - K) {gV-gK:+.2f} | retard {d:+.3f} | amplitude relative {abs(zK)/abs(zV):.3f} | int(1-rho)dx = {I[m].mean():.3f}')
