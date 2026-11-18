#!/usr/bin/env python3
"""
halo_relax.py -- relaxe un noeud du modele Z dans une boite plus grande (part187)

Part d'un champ phi0 (4, n0, n0, n0) (par ex. resultats/part179/wk_T0/wk_T0_champ_final.npy, float16 accepte), le place au centre
d'un reseau N^3 rempli du vide (1,0,0,0), puis minimise l'energie statique du modele Z (zlat.py) par L-BFGS-B, en float32 pour
l'energie/gradient et float64 pour le vecteur de L-BFGS. Ecrit un point de reprise tous les --save iterations.

Usage (depuis numerical_minimization/protect3d_Z) :
  python halo_relax.py --in ../../resultats/part179/wk_T0/wk_T0_champ_final.npy --N 96 --h 0.5 --mu 0.375 --e0 0.25 --d2 1e-3 \
         --maxit 4000 --save 500 --out relaxe_N96.npy | tee relaxe_N96.log

Remarque : le noeud de wk_T0 n'est pas au viriel de Derrick (E4/(E2+3EV) = 1,30 > 1 : l'energie baisserait si le noeud grossissait).
Dans une boite plus grande le minimiseur peut donc le faire grossir ; la taille obtenue fait partie du resultat (voir halo_mesure.py :
rayon median et rms de l'energie).
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from zlat import make   # (zlat desactive x64 a l'import : on le reactive plus bas si --x64)


def embed_centre(phi, N):
    n0 = phi.shape[1]
    out = np.zeros((4, N, N, N), dtype=np.float32)
    out[0] = 1.0
    s = (N - n0) // 2
    out[:, s:s + n0, s:s + n0, s:s + n0] = phi.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.375)
    ap.add_argument("--e0", type=float, default=0.25)
    ap.add_argument("--d2", type=float, default=1e-3)
    ap.add_argument("--maxit", type=int, default=4000)
    ap.add_argument("--save", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-degree", action="store_true")
    ap.add_argument("--x64", type=int, default=1, help="1 : energie et gradient en float64 (recommande : en float32 L-BFGS-B s'arrete sur le bruit d'arrondi, ~1e-7 E)")
    a = ap.parse_args()
    jax.config.update("jax_enable_x64", bool(a.x64))
    fl = jnp.float64 if a.x64 else jnp.float32

    phi0 = np.load(a.inp)
    if phi0.shape[1] == a.N:
        start = phi0.astype(np.float32)  # float16 -> float32 sans perte
    else:
        start = embed_centre(phi0, a.N)
    shape = start.shape
    cF = 1.0 / (2 * a.e0 ** 2)
    Ej, gE, parts, qdens = make(a.h, a.mu, cF, a.d2)

    def fg(x):
        p = jnp.array(x.reshape(shape), dtype=fl)
        return float(Ej(p)), np.array(gE(p), dtype=np.float64).ravel()

    k = [0]
    t0 = time.time()

    def measure(p, it):
        pa = np.array(parts(p))
        q = float(np.array(qdens(p)).sum() * a.h ** 3)
        rho_min = float(np.sqrt(np.sum(np.array(p) ** 2, axis=0)).min())
        e, g = fg(np.array(p, dtype=np.float64).ravel())
        print("it %5d  E=%.4f  E2=%.3f E4=%.3f EV=%.3f  E4/(E2+3EV)=%.3f  Q=%.3f  rho_min=%.3f  |g|rms=%.2e  t=%.0fs"
              % (it, e, pa[0], pa[1], pa[2], pa[1] / (pa[0] + 3 * pa[2]), q, rho_min, float(np.sqrt(np.mean(g ** 2))), time.time() - t0), flush=True)

    def cb(x):
        k[0] += 1
        if k[0] % 100 == 0:
            p = jnp.array(x.reshape(shape), dtype=fl)
            measure(p, k[0])
        if a.save > 0 and k[0] % a.save == 0:
            np.save(a.out, x.reshape(shape))

    measure(jnp.array(start, dtype=fl), 0)
    res = minimize(fg, start.astype(np.float64).ravel(), jac=True, method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=a.maxit, maxcor=20, gtol=1e-7, ftol=1e-14, maxfun=int(a.maxit * 1.3)))
    p = res.x.reshape(shape)
    np.save(a.out, p)
    print("FIN", res.message, "nit", res.nit, flush=True)
    measure(jnp.array(p, dtype=fl), res.nit)
    if not a.no_degree:
        try:
            import degre
            print("degre exact", degre.degree(p), flush=True)
        except Exception as ex:  # le degre est un controle, pas un prerequis
            print("degre non calcule :", ex, flush=True)


if __name__ == "__main__":
    main()
