#!/usr/bin/env python3
"""part217 -- inertie (tenseur M_ij) et couplage au fond d'un noeud relaxe du modele Z (L1, nu=0) avec terme de Faddeev temporel KF=KT=1.
Lagrangien dynamique (dyn186c) : L = |dphi/dt|^2 + KF*cF*sum_i (n.dt n x d_i n)^2/... - E_stat ; pour un noeud translate a la vitesse v le long de x :
  (1/2) M_xx v^2 = v^2 (m_x + F_x),  m_x = int |d_x phi|^2,  F_x = cF int [(n.d_x n x d_y n)^2 + (n.d_x n x d_z n)^2]   =>  M_xx = 2 (m_x + F_x)   (audit part200)
Couplage modele A (E2 et EV ~ a^2, E4 fixe) : F = - g dE/dlna = -g * 2 (E2 + EV) => a/g = 2(E2+EV)/M_xx.
Usage : python inertie_noeud.py champ.npy --tag T --out T.json   (EV compte au-dessus du vide de boite)."""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import zlat
jax.config.update("jax_enable_x64", True)
from zlat import fwd, nvec, cross

ap = argparse.ArgumentParser()
ap.add_argument("champ"); ap.add_argument("--h", type=float, default=0.5); ap.add_argument("--mu", type=float, default=0.375)
ap.add_argument("--e0", type=float, default=0.25); ap.add_argument("--d2", type=float, default=1e-3)
ap.add_argument("--tag", default="noeud"); ap.add_argument("--out", default=None)
a = ap.parse_args()
phi = jnp.asarray(np.load(a.champ).astype(np.float64)); h = a.h; cF = 1.0 / (2 * a.e0 ** 2)
g, f, p = zlat.energy_density(phi, h, a.mu, cF, a.d2)
E2, E4, EV = float(h**3 * g.sum()), float(h**3 * f.sum()), float(h**3 * p.sum())
vac = a.mu ** 2 * (1.0 / (1.0 + a.d2) - 1.0) * phi.shape[1] ** 3 * h ** 3
EVp = EV - vac
m = [float(h**3 * jnp.sum(fwd(phi, ax, h) ** 2)) for ax in (1, 2, 3)]
n = nvec(phi)
Df = [fwd(n, ax, h) for ax in (1, 2, 3)]; Db = [(n - jnp.roll(n, 1, axis=ax)) / h for ax in (1, 2, 3)]
def Fij(i, j):
    t = 0.0
    for Di in (Df[i], Db[i]):
        for Dj in (Df[j], Db[j]):
            t = t + 0.25 * jnp.sum(n * cross(Di, Dj), axis=0) ** 2
    return float(cF * h**3 * jnp.sum(t))
F12, F13, F23 = Fij(0, 1), Fij(0, 2), Fij(1, 2)
Fx, Fy, Fz = F12 + F13, F12 + F23, F13 + F23
M = [2 * (m[0] + Fx), 2 * (m[1] + Fy), 2 * (m[2] + Fz)]
Etot = E2 + E4 + EVp
out = dict(tag=a.tag, E_moins_vide=Etot, E2=E2, E4=E4, EV_moins_vide=EVp, m=m, F=[Fx, Fy, Fz], F_pairs=[F12, F13, F23], E4_check=F12 + F13 + F23,
           M_xx_yy_zz=M, M_sur_E=[x / Etot for x in M], couplage_modeleA_2_E2_EV=2 * (E2 + EVp), couplage_sur_E=2 * (E2 + EVp) / Etot,
           a_sur_g_pred_xx=2 * (E2 + EVp) / M[0], a_sur_g_pred_zz=2 * (E2 + EVp) / M[2], couplage_E2seul_sur_M_xx=2 * E2 / M[0])
print(json.dumps(out, indent=1))
if a.out: json.dump(out, open(a.out, "w"), indent=1)
