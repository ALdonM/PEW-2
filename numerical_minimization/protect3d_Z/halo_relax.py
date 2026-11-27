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


def make_slab(h, mu, cF, d2, N, ep):
    """Energie, gradient, parties et charge calcules par tranches d'epaisseur ep le long de l'axe 1 (halo de 1 maille, periodique).
    Necessaire pour N>=192 : jax compile l'energie complete avec ~500 tableaux N^3 intermediaires (29 Go a N=192, OOM)."""
    import zlat
    assert N % ep == 0, "N doit etre multiple de --slab"
    ns = N // ep

    def idx(i):
        return (np.arange(i * ep - 1, (i + 1) * ep + 1)) % N

    def E_pad(pp):
        g, f, p = zlat.energy_density(pp, h, mu, cF, d2)
        return h ** 3 * jnp.sum((g + f + p)[1:-1])

    def parts_pad(pp):
        g, f, p = zlat.energy_density(pp, h, mu, cF, d2)
        return h ** 3 * jnp.array([jnp.sum(g[1:-1]), jnp.sum(f[1:-1]), jnp.sum(p[1:-1])])

    def q_pad(pp):
        z = pp / jnp.sqrt(jnp.sum(pp ** 2, axis=0))
        M = jnp.stack([z] + [zlat.cen(z, ax, h) for ax in (1, 2, 3)])
        M = jnp.moveaxis(M, (0, 1), (-2, -1))
        return jnp.sum(jnp.linalg.det(M)[1:-1]) / (2 * np.pi ** 2)

    vg_pad, gE_pad, E_j, parts_j, q_j = jax.jit(jax.value_and_grad(E_pad)), jax.jit(jax.grad(E_pad)), jax.jit(E_pad), jax.jit(parts_pad), jax.jit(q_pad)

    def pad(p, i):
        return jnp.asarray(np.asarray(p)[:, idx(i)])

    def Ej(p):
        return sum(float(E_j(pad(p, i))) for i in range(ns))

    def gE(p):
        g = np.zeros(np.asarray(p).shape, dtype=np.float64)
        for i in range(ns):
            g[:, idx(i)] += np.asarray(gE_pad(pad(p, i)))
        return g

    def vg(p):
        g = np.zeros(np.asarray(p).shape, dtype=np.float64)
        e = 0.0
        for i in range(ns):
            ei, gi = vg_pad(pad(p, i))
            e += float(ei)
            g[:, idx(i)] += np.asarray(gi)
        return e, g

    def parts(p):
        return sum(np.asarray(parts_j(pad(p, i))) for i in range(ns))

    def qdens(p):
        return np.array([sum(float(q_j(pad(p, i))) for i in range(ns))])  # sum(det)/(2 pi^2) ; le appelant multiplie par h^3

    Ej.vg = vg
    return Ej, gE, parts, qdens


def tourne(phi, axe, deg):
    """Rotation spatiale (rotation des coordonnees seulement, indices internes du doublet inchanges)
    du champ phi (4,N,N,N) autour de son centre, par interpolation cubique. Donne une autre
    configuration de meme charge de Hopf (rotation continue) ; sert a tester la memoire de l'axe initial."""
    from scipy.ndimage import affine_transform
    ax = np.asarray(axe, dtype=float); ax /= np.linalg.norm(ax)
    t = np.radians(deg); K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)
    M = R.T  # coordonnee d'entree = R^-1 (x - c) + c
    c = (np.array(phi.shape[1:]) - 1) / 2.0
    off = c - M @ c
    out = np.stack([affine_transform(phi[i].astype(np.float64), M, offset=off, order=3, mode="nearest") for i in range(4)])
    out[:, 0, :, :] = out[:, :, 0, :] = out[:, :, :, 0] = 0  # bords : retour au vide (1,0,0,0) ci-dessous
    return out.astype(np.float32)


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
    ap.add_argument("--nu", type=float, default=0.0, help="masse des modes charges : + 2 nu^2 |Z2|^2 (part184b) ; 0 = modele Z pur")
    ap.add_argument("--a", type=float, default=1.0, help="amplitude de fond a (part216) : E2 et EV multiplies par a^2 (champ Z' = Z/a, vide (1,0,0,0)), E4 (Faddeev) inchange ; a=1 : modele usuel")
    ap.add_argument("--wr", type=float, default=1.0, help="raideur de la houle (part218) : |grad rho|^2 multiplie par wr dans E2 (la partie angulaire reste a 1) ; wr=1 : modele usuel")
    ap.add_argument("--slab", type=int, default=0, help="epaisseur de tranche (N multiple) : energie/gradient par tranches, obligatoire pour N>=192 (RAM)")
    ap.add_argument("--maxcor", type=int, default=20, help="memoire L-BFGS-B (nb de paires s,y) ; 6 pour N>=192 (RAM)")
    ap.add_argument("--tmax", type=float, default=0.0, help="duree max de la minimisation en secondes (arret propre, le champ est sauvegarde)")
    ap.add_argument("--taper", default="", help="r0,r1 (unites physiques) : ecart au vide multiplie par un cos^2 de 1 (r<r0) a 0 (r>r1) apres l'insertion ; sert a agrandir la boite depuis un champ deja relaxe")
    ap.add_argument("--rot", default="", help="ax,ay,az,angle_deg : rotation spatiale du noeud avant relaxation (test de memoire de l'axe)")
    ap.add_argument("--free", default="", help="fichier .npy (N,N,N) : 1 = site libre, 0 = site fige (gradient annule : le champ y reste exactement a sa valeur de depart) ; sert aux paires a cœurs figes (part187f)")
    ap.add_argument("--x64", type=int, default=1, help="1 : energie et gradient en float64 (recommande : en float32 L-BFGS-B s'arrete sur le bruit d'arrondi, ~1e-7 E)")
    a = ap.parse_args()
    jax.config.update("jax_enable_x64", bool(a.x64))
    fl = jnp.float64 if a.x64 else jnp.float32

    phi0 = np.load(a.inp)
    if phi0.shape[1] == a.N:
        start = phi0.astype(np.float32)  # float16 -> float32 sans perte
    else:
        start = embed_centre(phi0, a.N)
    if a.rot:
        v = [float(x) for x in a.rot.split(",")]
        start = tourne(start, v[:3], v[3])
        start[:, 0, :, :] = start[:, :, 0, :] = start[:, :, :, 0] = 0
        start[0, 0, :, :] = start[0, :, 0, :] = start[0, :, :, 0] = 1.0  # vide sur les faces
        print("depart tourne :", a.rot, flush=True)
    if a.taper:
        r0, r1 = (float(x) for x in a.taper.split(","))
        Nn = start.shape[1]
        x1 = (np.arange(Nn) - Nn / 2 + 0.5) * a.h
        R = np.sqrt(x1[:, None, None] ** 2 + x1[None, :, None] ** 2 + x1[None, None, :] ** 2)
        w = np.where(R <= r0, 1.0, np.where(R >= r1, 0.0, np.cos(0.5 * np.pi * (R - r0) / (r1 - r0)) ** 2)).astype(np.float32)
        vac = np.zeros_like(start); vac[0] = 1.0
        start = (vac + (start - vac) * w).astype(np.float32)
        print("taper cos^2 :", r0, r1, flush=True)
    shape = start.shape
    cF = 1.0 / (2 * a.e0 ** 2)
    if a.nu > 0:
        import zlat as _z
        _ed, _nu = _z.energy_density, a.nu

        def _ed_nu(phi, h, mu, cF, d2):
            g, f, p = _ed(phi, h, mu, cF, d2)
            return g, f, p + 2.0 * _nu ** 2 * (phi[2] ** 2 + phi[3] ** 2)
        _z.energy_density = _ed_nu
    if a.wr != 1.0:
        import zlat as _z3
        _ed3, _wr3 = _z3.energy_density, a.wr

        def _ed_wr(phi, h, mu, cF, d2):
            g, f, p = _ed3(phi, h, mu, cF, d2)
            rho = jnp.sqrt(jnp.sum(phi ** 2, axis=0))
            gr = sum(_z3.fwd(rho, ax, h) ** 2 for ax in (0, 1, 2))
            return g + (_wr3 - 1.0) * gr, f, p
        _z3.energy_density = _ed_wr
        print("raideur de la houle wr =", a.wr, flush=True)
    if a.a != 1.0:
        import zlat as _z2
        _ed2, _a2 = _z2.energy_density, a.a

        def _ed_a(phi, h, mu, cF, d2):
            g, f, p = _ed2(phi, h, mu, cF, d2)
            return _a2 ** 2 * g, f, _a2 ** 2 * p
        _z2.energy_density = _ed_a
        print("amplitude de fond a =", a.a, flush=True)
    if a.slab > 0:
        Ej, gE, parts, qdens = make_slab(a.h, a.mu, cF, a.d2, shape[1], a.slab)
    else:
        Ej, gE, parts, qdens = make(a.h, a.mu, cF, a.d2)

    libre = np.load(a.free).astype(np.float64) if a.free else None
    if libre is not None:
        assert libre.shape == shape[1:], (libre.shape, shape)
        print("sites figes :", int((libre == 0).sum()), "sur", libre.size, flush=True)

    def fg(x):
        p = jnp.array(x.reshape(shape), dtype=fl)
        if hasattr(Ej, "vg"):
            e, g = Ej.vg(p)
            g = np.array(g, dtype=np.float64)
        else:
            e, g = float(Ej(p)), np.array(gE(p), dtype=np.float64)
        if libre is not None:
            g = g.reshape(shape) * libre[None]
        return float(e), g.ravel()

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
        if a.tmax and time.time() - t0 > a.tmax:
            np.save(a.out, x.reshape(shape))
            print("arret sur --tmax a l'iteration", k[0], flush=True)
            raise StopIteration
        if k[0] % 100 == 0:
            p = jnp.array(x.reshape(shape), dtype=fl)
            measure(p, k[0])
        if a.save > 0 and k[0] % a.save == 0:
            np.save(a.out, x.reshape(shape))

    measure(jnp.array(start, dtype=fl), 0)
    res = minimize(fg, start.astype(np.float64).ravel(), jac=True, method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=a.maxit, maxcor=a.maxcor, gtol=1e-7, ftol=1e-14, maxfun=int(a.maxit * 1.3)))
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
