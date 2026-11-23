"""part202 -- test de boost (modele L0 : NU=0, sans jauge). Donnee boostee a t=0 : phi_b(x) = phi_0(c + gamma (x-c), y, z), phi_dot_b = -gamma v d_x phi_0(.)  (solution exacte d'une theorie covariante).
Theorie covariante : H_b = gamma M, P = gamma M v, H_b^2 = M^2 + P^2. Usage : python test_boost202.py"""
import os, sys, numpy as np, jax, jax.numpy as jnp
from scipy.ndimage import map_coordinates
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lagcov
f0 = np.load('../../resultats/part179/wk_T0/wk_T0_champ_final.npy').astype(np.float64); N = f0.shape[1]; h = 0.5; c = (N - 1) / 2
print('champ', f0.shape, flush=True)
def sample(f, gam, deriv=False):
    ix = np.arange(N, dtype=np.float64); X, Y, Z = np.meshgrid(ix, ix, ix, indexing='ij'); Xs = c + gam * (X - c)
    out = np.stack([map_coordinates(f[k], [Xs, Y, Z], order=3, mode='grid-wrap') for k in range(4)])
    return out
def dx(f): return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * h)
for KF in (0.0, 1.0):
    S = lagcov.build(N, h=h, nu=0.0, g2=0.0, KF=KF); th = jnp.zeros((3, N, N, N)); w = jnp.zeros((3, N, N, N))
    ph0 = sample(f0, 1.0); M = float(S['energy'](jnp.array(ph0, dtype=jnp.float32), th))
    print(f"KF={KF:g} : M = E(statique, re-echantillonne gamma=1) = {M:.2f}", flush=True)
    for v in (0.05, 0.1, 0.2, 0.3, 0.5):
        g = 1 / np.sqrt(1 - v * v); phb = sample(f0, g);   # d/dx de phi0(gamma x) = gamma phi0' : phi_dot = -v d_x phi_b
        vel = -v * dx(phb)
        q = (jnp.array(phb, dtype=jnp.float32), th); vv = (jnp.array(vel, dtype=jnp.float32), w)
        Hb = float(S['hamil'](q, vv)); p = S['pfun'](q, vv)[0]; P = float(-jnp.sum(p * jnp.array(dx(phb), dtype=jnp.float32)))   # p = dK/dv contient deja h^3
        print(f"  v={v}: gamma={g:.4f}  H_b/M = {Hb / M:.4f} (covariant : {g:.4f})   P/(M v) = {P / (M * v):.4f} (covariant : {g:.4f})   H_b^2 - P^2 = {np.sqrt(max(Hb**2 - P**2, 0)) / M:.4f} M (covariant : 1)", flush=True)
