#!/usr/bin/env python3
"""
halo_mesure.py -- halo d'un noeud du modele Z (part187)

Mesure, sur un champ phi (4, N, N, N) float, centre sur un noeud isole :
  1. l'energie cumulee E(<r) / Etot (total, gradient E2, Faddeev E4, potentiel EV) ;
  2. les profils spheriques de 1-rho, de Theta = |grad zeta|^2, de la densite d'energie, du courant de phase |j| ;
  3. les pentes locales (log-log) de ces profils par fenetres, et un ajustement A r^-p + B (B = fond d'images) ;
  4. les multipoles angulaires (l = 1..4) de 1-rho et de la densite d'energie sur chaque coquille ;
  5. le tenseur d'inertie de l'energie (axe, rapport d'axes) et le dipole de courant D(<R), flux de courant par plan.

Meme fonctionnelle que protect3d_Z/zlat.py (Z : e = |grad Z|^2 + cF (n.Dn x Dn)^2 + mu^2 (rho^2 + 1/(rho^2+d2) - 2), cF = 1/(2 e0^2)),
reecrite en numpy float64 (aucune dependance JAX). Le script verifie les totaux E2, E4, EV contre ceux du journal si on les donne
(--ref E2,E4,EV).

Usage : python halo_mesure.py champ.npy --h 0.5 --mu 0.375 --e0 0.25 --d2 1e-3 --tag wk_T0 --out halo_wk_T0.json [--ref 319.47,527.94,28.88]
Convention de maillage : x_i = (i - N/2 + 0.5) h (comme zlat.py) ; le centre du noeud est recalcule (barycentre periodique de l'energie).
"""
import argparse, json, sys
import numpy as np
try:
    from scipy.special import sph_harm_y
except ImportError:  # scipy < 1.15
    from scipy.special import sph_harm

    def sph_harm_y(n, m, theta, phi):
        return sph_harm(m, n, phi, theta)


# ----------------------------------------------------------------------------- fonctionnelle
def fwd(f, ax, h):
    return (np.roll(f, -1, axis=ax) - f) / h


def cen(f, ax, h):
    return (np.roll(f, -1, axis=ax) - np.roll(f, 1, axis=ax)) / (2 * h)


def densities(phi, h, mu, e0, d2):
    """Retourne e2 (|grad Z|^2), e4 (Faddeev), eV (potentiel), rho, Theta (=|grad zeta|^2, differences avant), j (courant de phase, diff. centrees)."""
    phi = phi.astype(np.float64)
    cF = 1.0 / (2 * e0 ** 2)
    e2 = sum(np.sum(fwd(phi, ax, h) ** 2, axis=0) for ax in (1, 2, 3))
    r2 = np.sum(phi ** 2, axis=0)
    a, b, c, d = phi
    n = np.stack([2 * (a * c + b * d), 2 * (a * d - b * c), a * a + b * b - c * c - d * d]) / r2
    Df = [fwd(n, ax, h) for ax in (1, 2, 3)]
    Db = [(n - np.roll(n, 1, axis=ax)) / h for ax in (1, 2, 3)]
    e4 = 0.0
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        for Di in (Df[i], Db[i]):
            for Dj in (Df[j], Db[j]):
                cr = np.cross(Di, Dj, axisa=0, axisb=0, axisc=0)
                e4 = e4 + 0.25 * np.sum(n * cr, axis=0) ** 2
    e4 = cF * e4
    eV = mu ** 2 * (r2 + 1.0 / (r2 + d2) - 2.0)
    rho = np.sqrt(r2)
    zeta = phi / rho
    theta = sum(np.sum(fwd(zeta, ax, h) ** 2, axis=0) for ax in (1, 2, 3))
    j = np.stack([a * cen(b, ax, h) - b * cen(a, ax, h) + c * cen(d, ax, h) - d * cen(c, ax, h) for ax in (0, 1, 2)])
    return e2, e4, eV, rho, theta, j


# ----------------------------------------------------------------------------- geometrie
def centre_periodique(e, h):
    """Barycentre periodique de e (coordonnees physiques, convention x_i = (i - N/2 + 0.5) h)."""
    N = e.shape[0]
    L = N * h
    x = (np.arange(N) - N / 2 + 0.5) * h
    c = []
    for ax in range(3):
        w = e.sum(axis=tuple(k for k in range(3) if k != ax))
        ang = 2 * np.pi * x / L
        m = np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum())
        c.append(m * L / (2 * np.pi))
    return np.array(c)


def coords(N, h, c):
    L = N * h
    x = (np.arange(N) - N / 2 + 0.5) * h
    dx = [((x - c[i] + L / 2) % L) - L / 2 for i in range(3)]
    X, Y, Z = np.meshgrid(*dx, indexing="ij")
    return X, Y, Z


def shell_mean(f, r, h, rmax):
    nb = int(np.floor(rmax / h))
    idx = np.minimum((r / h).astype(int), nb)  # nb = "au-dela"
    s = np.bincount(idx.ravel(), weights=f.ravel(), minlength=nb + 1)[:nb]
    n = np.bincount(idx.ravel(), minlength=nb + 1)[:nb]
    rc = (np.arange(nb) + 0.5) * h
    # rayon moyen reel de la coquille (plus precis que le milieu)
    rm = np.bincount(idx.ravel(), weights=r.ravel(), minlength=nb + 1)[:nb] / np.maximum(n, 1)
    return rm, s / np.maximum(n, 1), n


def cumul(e, r, h, rmax, etot):
    nb = int(np.floor(rmax / h))
    idx = np.minimum((r / h).astype(int), nb)
    s = np.bincount(idx.ravel(), weights=e.ravel() * h ** 3, minlength=nb + 1)[:nb]
    return (np.arange(1, nb + 1)) * h, np.cumsum(s) / etot


# ----------------------------------------------------------------------------- pentes et ajustements
def pente_locale(rm, f, r1, r2):
    m = (rm >= r1) & (rm <= r2) & (f > 0)
    if m.sum() < 3:
        return None
    p = np.polyfit(np.log(rm[m]), np.log(f[m]), 1)
    return float(p[0])


# ----------------------------------------------------------------------------- multipoles
def multipoles_coquille(f, X, Y, Z, r, r1, r2, lmax=4):
    """Pour la coquille r1<=r<r2 : amplitude rms de la partie l de f (l=1..lmax), rapportee a la moyenne de f sur la coquille.
    Harmoniques spheriques reelles orthonormees (sur la sphere unite), moindres carres sur les points de la coquille."""
    m = (r >= r1) & (r < r2)
    x, y, z, v = X[m], Y[m], Z[m], f[m]
    rr = np.sqrt(x * x + y * y + z * z)
    th = np.arccos(np.clip(z / rr, -1, 1))
    ph = np.arctan2(y, x)
    cols, ls = [], []
    for l in range(0, lmax + 1):
        for mm in range(-l, l + 1):
            Y_lm = sph_harm_y(l, abs(mm), th, ph)
            if mm == 0:
                col = Y_lm.real
            elif mm > 0:
                col = np.sqrt(2) * (-1) ** mm * Y_lm.real
            else:
                col = np.sqrt(2) * (-1) ** mm * Y_lm.imag
            cols.append(col)
            ls.append(l)
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    ls = np.array(ls)
    mean = float(v.mean())
    out = {"n_pts": int(m.sum()), "moyenne": mean}
    for l in range(1, lmax + 1):
        out[f"rms_l{l}"] = float(np.sqrt(np.sum(coef[ls == l] ** 2) / (4 * np.pi)))
    return out


# ----------------------------------------------------------------------------- programme
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("champ")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.375)
    ap.add_argument("--e0", type=float, default=0.25)
    ap.add_argument("--d2", type=float, default=1e-3)
    ap.add_argument("--tag", default="noeud")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None, help="E2,E4,EV de reference (journal)")
    ap.add_argument("--rcore", type=float, default=8.0, help="rayon pour le tenseur d'inertie")
    a = ap.parse_args()

    phi = np.load(a.champ)
    dtype0 = str(phi.dtype)
    N = phi.shape[1]
    h = a.h
    L = N * h
    rmax = L / 2
    e2, e4, eV, rho, theta, j = densities(phi, h, a.mu, a.e0, a.d2)
    # Le regulateur d2 donne au vide (rho=1) une densite de potentiel -mu^2 d2/(1+d2) non nulle : on la retranche pour les
    # fractions d'energie (sinon -4,6 sur 876 pour L=32, soit ~5 % de l'energie au-dela de r=10). Les totaux "bruts" restent ceux du journal.
    e_vac = a.mu ** 2 * (1.0 / (1.0 + a.d2) - 1.0)
    eV_brut = eV
    eV = eV - e_vac
    et = e2 + e4 + eV
    tot = {k: float(v.sum() * h ** 3) for k, v in dict(E2=e2, E4=e4, EV=eV, Etot=et).items()}
    tot_brut = {"EV_brut": float(eV_brut.sum() * h ** 3), "Etot_brut": float((e2 + e4 + eV_brut).sum() * h ** 3), "e_vide_par_volume": e_vac}
    res = {"tag": a.tag, "dtype": dtype0, "N": N, "h": h, "L": L, "totaux_au_dessus_du_vide": tot, "totaux_bruts": tot_brut}
    res["int_Theta"] = float(theta.sum() * h ** 3)
    res["int_rho2_Theta"] = float((rho ** 2 * theta).sum() * h ** 3)
    res["rho_min"] = float(rho.min())
    if a.ref:
        ref = [float(t) for t in a.ref.split(",")]
        brut = {"E2": tot["E2"], "E4": tot["E4"], "EV": tot_brut["EV_brut"]}
        res["ecart_relatif_vs_ref"] = {k: float(brut[k] / r - 1) for k, r in zip(("E2", "E4", "EV"), ref)}

    c = centre_periodique(et, h)
    res["centre"] = [float(t) for t in c]
    X, Y, Z = coords(N, h, c)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # 1. energie cumulee
    rc, cum_t = cumul(et, r, h, rmax, tot["Etot"])
    _, cum_2 = cumul(e2, r, h, rmax, tot["Etot"])
    _, cum_4 = cumul(e4, r, h, rmax, tot["Etot"])
    _, cum_V = cumul(eV, r, h, rmax, tot["Etot"])
    res["cumul"] = {"r": rc.tolist(), "total": cum_t.tolist(), "E2": cum_2.tolist(), "E4": cum_4.tolist(), "EV": cum_V.tolist()}
    res["rayon_median_energie"] = float(np.interp(0.5, cum_t, rc))
    ms = r < rmax
    res["rayon_rms_energie_dans_sphere_L2"] = float(np.sqrt((et[ms] * r[ms] ** 2).sum() / et[ms].sum()))
    res["fraction_hors_r"] = {f"{R:g}": float(1 - cum_t[int(round(R / h)) - 1]) for R in (2, 3, 4, 6, 8, 10, 12, 14) if R < rmax}

    # 2. profils spheriques
    prof = {}
    jm = j.mean(axis=(1, 2, 3))                      # courant moyen = flux total / L^2 (enroulement de la boite)
    jc = j - jm[:, None, None, None]
    for nom, f in (("un_moins_rho", 1 - rho), ("Theta", theta), ("e_tot", et), ("e2", e2), ("e4", e4), ("eV", eV),
                   ("abs_j", np.sqrt(np.sum(j ** 2, axis=0))), ("abs_j_sans_moyenne", np.sqrt(np.sum(jc ** 2, axis=0)))):
        rm, fm, n = shell_mean(f, r, h, rmax)
        prof[nom] = {"r": rm.tolist(), "val": fm.tolist()}
    res["profils"] = prof

    # 3. pentes
    fenetres = [(3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12), (12, 14), (14, 16)]
    pentes = {}
    for nom in prof:
        rm = np.array(prof[nom]["r"]); fm = np.array(prof[nom]["val"])
        pentes[nom] = {f"{r1:g}-{r2:g}": pente_locale(rm, fm, r1, r2) for (r1, r2) in fenetres if r2 <= rmax}
    res["pentes_locales"] = pentes
    # pente / r : constante pour une exponentielle (-k), decroissante pour une loi de puissance (-p/r)
    res["pente_sur_r"] = {nom: {k: (None if v is None else v / (0.5 * sum(float(t) for t in k.split("-")))) for k, v in pentes[nom].items()}
                          for nom in ("un_moins_rho", "Theta", "e_tot")}

    # image : axes de la boite contre diagonales, a meme rayon
    def axes_vs_diag(f, r1, r2):
        m = (r >= r1) & (r < r2)
        u = np.stack([np.abs(X), np.abs(Y), np.abs(Z)]) / np.maximum(r, 1e-9)
        axe = m & (u.max(axis=0) > 0.94)             # moins de ~20 deg d'un axe de la boite
        diag = m & (u.min(axis=0) > 0.45)            # pres des diagonales (1,1,1)/sqrt3 = 0.577
        if axe.sum() < 20 or diag.sum() < 20:
            return None
        return float(f[axe].mean() / f[diag].mean())
    res["rapport_axes_sur_diagonales_1moinsrho"] = {f"{r1:g}-{r1 + 2:g}": axes_vs_diag(1 - rho, r1, r1 + 2) for r1 in (4, 6, 8, 10, 12, 14) if r1 + 2 <= rmax}

    # 4. multipoles
    mp = {}
    for nom, f in (("un_moins_rho", 1 - rho), ("e_tot", et)):
        mp[nom] = {}
        for r1 in (2, 4, 6, 8, 10, 12, 14):
            if r1 + 2 <= rmax:
                mp[nom][f"{r1:g}-{r1 + 2:g}"] = multipoles_coquille(f, X, Y, Z, r, r1, r1 + 2)
    res["multipoles"] = mp

    # 5. inertie de l'energie et dipole de courant
    m = r < a.rcore
    w = et[m] * h ** 3
    xs = np.stack([X[m], Y[m], Z[m]])
    I = (xs[:, None, :] * xs[None, :, :] * w).sum(axis=2) / w.sum()
    ev, evec = np.linalg.eigh(I)
    # axe unique = vecteur propre dont la valeur propre est la plus eloignee des deux autres (oblate : 1 petite ; prolate : 1 grande)
    ecart = [abs(ev[k] - np.mean([ev[q] for q in range(3) if q != k])) for k in range(3)]
    ku = int(np.argmax(ecart))
    axe_u = evec[:, ku]
    forme = "oblate (aplati le long de l'axe)" if ev[ku] < np.mean([ev[q] for q in range(3) if q != ku]) else "prolate (allonge le long de l'axe)"
    res["inertie_energie"] = {"rayon": a.rcore, "valeurs_propres": ev.tolist(), "axe_unique": axe_u.tolist(), "forme": forme,
                              "rapport_max_min": float(ev[2] / ev[0])}
    evec_u = axe_u
    Dtot = (j.sum(axis=(1, 2, 3)) * h ** 3)
    res["D_boite"] = Dtot.tolist()
    res["D_dans_r"] = {f"{R:g}": (np.array([j[k][r < R].sum() * h ** 3 for k in range(3)])).tolist() for R in (4, 6, 8, 10, 12, 14) if R < rmax}
    flux = {}
    for ax, nom in enumerate(("x", "y", "z")):
        fl = j[ax].sum(axis=tuple(k for k in range(3) if k != ax)) * h ** 2
        flux[nom] = {"min": float(fl.min()), "max": float(fl.max()), "moyenne": float(fl.mean())}
    res["flux_courant_par_plan"] = flux
    ang = None
    Dn = np.linalg.norm(Dtot)
    if Dn > 0:
        ang = float(np.degrees(np.arccos(min(1.0, abs(np.dot(Dtot / Dn, evec_u))))))
    res["angle_D_axe_unique_deg"] = ang
    res["courant_moyen"] = jm.tolist()
    res["dipole_sans_moyenne_dans_r"] = {f"{R:g}": (np.array([jc[k][r < R].sum() * h ** 3 for k in range(3)])).tolist() for R in (4, 6, 8, 10, 12, 14) if R < rmax}

    # ---- resume imprime
    print(f"== {a.tag} : N={N} h={h} L={L} dtype={dtype0}")
    print("totaux au-dessus du vide", {k: round(v, 3) for k, v in tot.items()}, "| bruts", {k: round(v, 3) for k, v in tot_brut.items()}, "| int Theta %.1f  int rho^2 Theta %.1f  rho_min %.3f" % (res["int_Theta"], res["int_rho2_Theta"], res["rho_min"]))
    if "ecart_relatif_vs_ref" in res:
        print("ecart vs reference", {k: "%+.2e" % v for k, v in res["ecart_relatif_vs_ref"].items()})
    print("centre", np.round(c, 3))
    print("rayon median de l'energie %.2f | rayon rms (sphere r<L/2) %.2f" % (res["rayon_median_energie"], res["rayon_rms_energie_dans_sphere_L2"]))
    print("fraction d'energie HORS de r :", {k: round(v, 4) for k, v in res["fraction_hors_r"].items()})
    for nom in ("un_moins_rho", "Theta", "e_tot", "abs_j", "abs_j_sans_moyenne"):
        print("pentes", nom, {k: (None if v is None else round(v, 2)) for k, v in pentes[nom].items()})
    for nom in ("un_moins_rho", "Theta", "e_tot"):
        print("pente/r", nom, {k: (None if v is None else round(v, 3)) for k, v in res["pente_sur_r"][nom].items()})
    print("axes/diagonales (1-rho)", {k: (None if v is None else round(v, 3)) for k, v in res["rapport_axes_sur_diagonales_1moinsrho"].items()})
    for nom in mp:
        for k, v in mp[nom].items():
            print("multipoles", nom, k, " ".join("l%d=%.3f" % (l, v[f"rms_l{l}"] / abs(v["moyenne"])) for l in (1, 2, 3, 4)))
    print("inertie energie (r<%g) : vp" % a.rcore, np.round(ev, 3), "rapport max/min %.3f" % res["inertie_energie"]["rapport_max_min"], res["inertie_energie"]["forme"], "axe unique", np.round(axe_u, 3))
    print("D_boite", np.round(Dtot, 1), "| angle D / axe unique : %s deg" % (None if ang is None else round(ang, 1)), "| courant moyen", np.round(jm, 5))
    print("D sans moyenne, dans r<R :", {k: np.round(v, 1).tolist() for k, v in res["dipole_sans_moyenne_dans_r"].items()})
    print("flux de courant par plan", {k: (round(v["min"], 2), round(v["max"], 2)) for k, v in flux.items()})

    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)
        print("ecrit", a.out)


if __name__ == "__main__":
    sys.exit(main())
