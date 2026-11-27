#!/usr/bin/env python3
"""part219 -- plus bas modes de fluctuation (valeurs propres de la Hessienne de E) autour d'un noeud relaxe du modele Z (L1, nu=0).
H = d2E/dphi2 (E = h^3 sum e), valeurs propres lam = (v^T H v)/(h^3 v^T v) (unites du champ) ; la dynamique lineaire est phi'' = -(1/2) H phi / h^3 => omega^2 = lam/2 (sans Faddeev temporel).
Methode : LOBPCG (vecteurs de depart : modes de translation d_x,d_y,d_z phi, rotation de phase i*phi, + aleatoires), preconditionneur FFT (laplacien de reseau + decalage).
Modes attendus ~ 0 : 3 translations, symetrie globale U(2) (4), rotations spatiales si le noeud n'est pas sphérique ; modes de vide de grande longueur d'onde ~ 2 k^2 (k = 2 pi / L). Une valeur < 0 = instabilite.
Usage : python hess_modes.py champ.npy --tag T --out T.json [--k 12] [--tmax 14000]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import zlat
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator, lobpcg

ap = argparse.ArgumentParser()
ap.add_argument("champ"); ap.add_argument("--h", type=float, default=0.5); ap.add_argument("--mu", type=float, default=0.375)
ap.add_argument("--e0", type=float, default=0.25); ap.add_argument("--d2", type=float, default=1e-3)
ap.add_argument("--k", type=int, default=12); ap.add_argument("--tmax", type=float, default=14000.0)
ap.add_argument("--tol", type=float, default=1e-5); ap.add_argument("--tag", default="noeud"); ap.add_argument("--out", default=None)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
phi0 = np.load(a.champ).astype(np.float64); shape = phi0.shape; N = shape[1]; h = a.h; n = phi0.size
cF = 1.0 / (2 * a.e0 ** 2)
Ej, gE, parts, qdens = zlat.make(h, a.mu, cF, a.d2)
P0 = jnp.asarray(phi0)
hvp = jax.jit(lambda v: jax.jvp(gE, (P0,), (v,))[1])
g0 = np.asarray(gE(P0)); print("|g|rms/h^3 =", float(np.sqrt(np.mean(g0 ** 2))) / h ** 3, flush=True)
h3 = h ** 3
def mv(x):
    return np.asarray(hvp(jnp.asarray(x.reshape(shape)))).ravel() / h3
A = LinearOperator((n, n), matvec=mv, matmat=lambda X: np.stack([mv(X[:, j]) for j in range(X.shape[1])], axis=1), dtype=np.float64)
k1 = 2 * np.pi * np.fft.fftfreq(N, d=h)
kx = (2 - 2 * np.cos(k1 * h)) / h ** 2
K2 = kx[:, None, None] + kx[None, :, None] + kx[None, None, :]
S = 0.3
def prec(X):
    X = np.asarray(X); out = np.empty_like(X)
    for j in range(X.shape[1]):
        f = np.fft.fftn(X[:, j].reshape(shape), axes=(1, 2, 3)) / (2 * K2[None] + S)
        out[:, j] = np.fft.ifftn(f, axes=(1, 2, 3)).real.ravel()
    return out
M = LinearOperator((n, n), matvec=lambda x: prec(x.reshape(-1, 1)).ravel(), matmat=prec, dtype=np.float64)
rng = np.random.default_rng(a.seed)
def cen(f, ax): return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * h)
tr = [np.stack([cen(phi0[c], ax) for c in range(4)]).ravel() for ax in (0, 1, 2)]
ph = np.stack([-phi0[1], phi0[0], -phi0[3], phi0[2]]).ravel()   # rotation de phase globale Z -> e^{i alpha} Z
X0 = np.stack(tr + [ph] + [rng.standard_normal(n) for _ in range(a.k - 4)], axis=1)
X0 /= np.linalg.norm(X0, axis=0)
t0 = time.time(); X = X0; hist = []
for rnd in range(1000):
    w, X, rn = lobpcg(A, X, M=M, largest=False, maxiter=8, tol=a.tol, retResidualNormsHistory=True)
    o = np.argsort(w); w = w[o]; X = X[:, o]
    res = np.array(rn[-1]) if len(rn) else np.zeros(len(w))
    print("ronde %d  t=%.0fs  lam =" % (rnd, time.time() - t0), np.round(w, 6).tolist(), " max_res=%.2e" % float(res.max()), flush=True)
    hist.append(w.tolist())
    if res.max() < a.tol * 10 or time.time() - t0 > a.tmax: break
# energie des modes dans le noeud (rayon) pour reconnaitre les modes localises
x1 = (np.arange(N) - N / 2 + 0.5) * h
R = np.sqrt(x1[:, None, None] ** 2 + x1[None, :, None] ** 2 + x1[None, None, :] ** 2)
loc = []
for j in range(X.shape[1]):
    v = X[:, j].reshape(shape); d = (v ** 2).sum(axis=0); loc.append(float((d * (R < 12)).sum() / d.sum()))
out = dict(tag=a.tag, N=N, h=h, k=a.k, lam=w.tolist(), residus=res.tolist(), fraction_dans_r12=loc, omega2_demi_lam=[x / 2 for x in w.tolist()],
           lam_vide_attendu_2k2=2 * (2 * np.pi / (N * h)) ** 2, histoire=hist, temps_s=time.time() - t0)
print(json.dumps({k: out[k] for k in ("tag", "lam", "residus", "fraction_dans_r12", "lam_vide_attendu_2k2")}, indent=1))
if a.out: json.dump(out, open(a.out, "w"), indent=1)
