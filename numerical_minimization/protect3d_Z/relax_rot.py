#!/usr/bin/env python3
"""part221 -- noeud du modele Z (L1, nu>0) avec frequence interne omega : solution Z = (Z1, e^{i omega t} Z2).
Dynamique (dyn186c) : L = h^3 K(phi, phidot) - E(phi), K = sum v^2 + 0.5 KF cF sum_6 (q.v)^2 (terme de Faddeev temporel, KF=1) ; gap des ondes de Z2 : omega_gap^2 = 2 nu^2.
Reduction par symetrie (phi(t) = exp(omega t J) phi0, J = rotation de phase de Z2) : phi0 est un point critique de F = E - omega^2 h^3 K(phi0, J phi0) ; on le minimise (L-BFGS-B, float64),
en chainant plusieurs omega (homotopie). Sorties par omega : E statique, K, energie totale E + omega^2 h^3 K, charge N = 2 omega h^3 K, rho_min, deficit int(1-rho) (r<12 et total), degre.
Usage : python relax_rot.py --in champ.npy --N 96 --nu 0.12 --omegas 0.04,0.08,0.12 --tag T [--tmax 3600] [--maxit 3000]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import zlat
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
from zlat import fwd, nvec, cross

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--N", type=int, default=96); ap.add_argument("--h", type=float, default=0.5)
ap.add_argument("--mu", type=float, default=0.375); ap.add_argument("--e0", type=float, default=0.25); ap.add_argument("--d2", type=float, default=1e-3)
ap.add_argument("--nu", type=float, default=0.12); ap.add_argument("--KF", type=float, default=1.0)
ap.add_argument("--omegas", default="0.04,0.08,0.12"); ap.add_argument("--maxit", type=int, default=3000); ap.add_argument("--tmax", type=float, default=3600.0)
ap.add_argument("--tag", default="rot"); ap.add_argument("--outdir", default=".")
a = ap.parse_args()
h, cF = a.h, 1.0 / (2 * a.e0 ** 2)
_ed, _nu = zlat.energy_density, a.nu
def _ed_nu(phi, h_, mu, cF_, d2):
    g, f, p = _ed(phi, h_, mu, cF_, d2)
    return g, f, p + 2.0 * _nu ** 2 * (phi[2] ** 2 + phi[3] ** 2)
zlat.energy_density = _ed_nu
Ej, gE, parts, qdens = zlat.make(h, a.mu, cF, a.d2)

def qvecs(phi):
    n = nvec(phi); _, vjp = jax.vjp(nvec, phi); out = []
    for ax in (1, 2, 3):
        Df = fwd(n, ax, h); Db = (n - jnp.roll(n, 1, axis=ax)) / h
        for D in (Df, Db): out.append(vjp(cross(D, n))[0])
    return out
def kin(phi, v):
    return jnp.sum(v ** 2) + 0.5 * a.KF * cF * sum(jnp.sum(jnp.sum(q * v, axis=0) ** 2) for q in qvecs(phi))
def Jm(p): return jnp.stack([0 * p[0], 0 * p[1], -p[3], p[2]])

phi = np.load(a.inp).astype(np.float64); shape = phi.shape; N = shape[1]
x1 = (np.arange(N) - N / 2 + 0.5) * h
R = np.sqrt(x1[:, None, None] ** 2 + x1[None, :, None] ** 2 + x1[None, None, :] ** 2)
res = []
for w in [float(x) for x in a.omegas.split(",")]:
    def F(x):
        p = x.reshape(shape)
        return Ej(p) - w ** 2 * h ** 3 * kin(p, Jm(p))
    vg = jax.jit(jax.value_and_grad(F))
    def fg(x):
        e, g = vg(jnp.asarray(x.reshape(shape)))
        return float(e), np.asarray(g, dtype=np.float64).ravel()
    t0 = time.time(); it = [0]
    def cb(x):
        it[0] += 1
        if it[0] % 100 == 0:
            e, g = fg(x); print("omega %.3f it %d F=%.4f |g|rms=%.2e t=%.0fs" % (w, it[0], e, float(np.sqrt(np.mean(g ** 2))), time.time() - t0), flush=True)
        if time.time() - t0 > a.tmax: raise StopIteration
    r = minimize(fg, phi.ravel(), jac=True, method="L-BFGS-B", callback=cb, options=dict(maxiter=a.maxit, maxcor=20, gtol=1e-7, ftol=1e-14, maxfun=int(a.maxit * 1.3)))
    phi = r.x.reshape(shape); P = jnp.asarray(phi)
    E = float(Ej(P)); K = float(h ** 3 * kin(P, Jm(P))); g = np.asarray(vg(P)[1]); rho = np.sqrt((phi ** 2).sum(axis=0))
    vac = a.mu ** 2 * (1.0 / (1.0 + a.d2) - 1.0) * N ** 3 * h ** 3
    d = 1 - rho
    out = dict(omega=w, nit=int(r.nit), msg=str(r.message), gradrms=float(np.sqrt(np.mean(g ** 2))), E_stat_moins_vide=E - vac, K=K, Ecin=w ** 2 * K, E_tot=E - vac + w ** 2 * K, charge_N=2 * w * K,
               rho_min=float(rho.min()), deficit_r12=float(h ** 3 * d[R < 12].sum()), deficit_tot=float(h ** 3 * d.sum()), degre_Q=float(np.asarray(qdens(P)).sum() * h ** 3),
               E2_E4_EV=[float(x) for x in np.asarray(parts(P))], temps_s=time.time() - t0)
    print(json.dumps(out), flush=True); res.append(out)
    np.save(os.path.join(a.outdir, "%s_w%03d.npy" % (a.tag, int(round(w * 1000)))), phi.astype(np.float32))
    json.dump(res, open(os.path.join(a.outdir, a.tag + "_mesures.json"), "w"), indent=1)
