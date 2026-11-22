"""Analyse part198 : retard de phase d'une onde libre de la porteuse. usage : python analyse_part198.py [racine]"""
import numpy as np, os, sys, json
R = sys.argv[1] if len(sys.argv) > 1 else '../../resultats/part198'
N = 96; h = 0.5; k = 3.0; xs = (np.arange(N) - N / 2 + .5) * h
x = xs[:, None, None]; Y, Zc = np.meshgrid(xs, xs, indexing='ij'); b = np.sqrt(Y**2 + Zc**2)
def L(t):
    old = t in ('r100_K0', 'r100_Vw')
    f = f"../../resultats/part196b/p196b_{t}/p196b_{t}_champ_final.npy" if old else f"{R}/p198_{t}/p198_{t}_champ_final.npy"
    return np.load(f).astype(float) if os.path.exists(f) else None
def Zs(p): return p[0] + 1j * p[1], p[2] + 1j * p[3]
def chi(P, Q):
    a1, a2 = Zs(P); b1, b2 = Zs(Q); return np.angle(np.conj(b1) * a1 + np.conj(b2) * a2)
bins = [(0, 3), (3, 5), (5, 7), (7, 9), (9, 11)]
kx = 2 * np.pi * np.fft.fftfreq(N, h)
def env(c):
    F = np.fft.fft(c, axis=0); F[~((kx > 2.0) & (kx < 4.0))] = 0   # enveloppe du paquet (k entre 2 et 4)
    return np.abs(np.fft.ifft(F, axis=0))
K0 = L('r100_K0'); V0 = L('r100_Vw'); rho = np.sqrt((K0**2).sum(0))
vac = np.zeros_like(K0); vac[0] = 1.0
cV = chi(V0, vac); win = ((x > 2.5) & (x < 14)).astype(float)
for tag, p in (('Kp05', 0.5), ('Kp1', 1.0)):
    K = L(f'r100_{tag}')
    if K is None: print(tag, 'absent'); continue
    cK = chi(K, K0); I = ((rho**(-(1+p)) - 1) * h).sum(0)
    print(f'--- p={p} : vitesse des ondes ~ rho^{1+p} ; prediction int(rho^-{1+p}-1)dx')
    for a_, c_ in bins:
        m = (b >= a_) & (b < c_)
        eK, eV = env(cK)**2, env(cV)**2; gK = (eK[:, m].mean(1) * xs).sum() / eK[:, m].mean(1).sum(); gV = (eV[:, m].mean(1) * xs).sum() / eV[:, m].mean(1).sum()
        zK = (cK * np.exp(-1j * k * x) * win)[:, m].sum(); zV = (cV * np.exp(-1j * k * x) * win)[:, m].sum()
        print(f'  b {a_}-{c_}: retard de GROUPE {gV-gK:+.2f} | prediction {I[m].mean():.2f} | amplitude relative {abs(zK)/abs(zV):.3f}')
