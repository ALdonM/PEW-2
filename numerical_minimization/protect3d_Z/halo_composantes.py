#!/usr/bin/env python3
"""part187c -- ventilation de l'energie du halo par composante, a grand r.

Lit un champ relaxe (4,N,N,N) et mesure, par coquilles spheriques autour du centre d'energie :
  e2 = |grad Z|^2 (separe en e_rho = |grad rho|^2 et e_ang = rho^2 Theta, Theta = |grad zeta|^2),
  e4 (Faddeev), eV (potentiel au-dessus du vide), e_j = |j|^2/rho^2 (partie « courant de phase » de e_ang : borne
  inferieure, Cauchy-Schwarz), |j| (courant de phase).
Sorties : moyennes par coquille, pentes log-log locales, part de chaque composante dans E(>R), ajustement en loi de
puissance de |j| et de e_j (A r^-p) et de exponentielle (A exp(-k r)) sur une fenetre [r1, r2] ; les deux ajustements sont
compares par leur residu relatif (le meilleur des deux est signale, SANS le conclure : 3 boites, une fenetre).
Usage : python halo_composantes.py champ.npy --h 0.5 --tag L64 --out comp_L64.json [--fen 10,18]
"""
import argparse, json
import numpy as np
from halo_mesure import densities, fwd, centre_periodique, coords, shell_mean, cumul, pente_locale


def ajuste(rm, f, r1, r2):
    m = (rm >= r1) & (rm <= r2) & (f > 0)
    if m.sum() < 4:
        return None
    x, y = rm[m], np.log(f[m])
    # puissance : y = ln A - p ln r ; exponentielle : y = ln A - k r
    out = {}
    for nom, xx in (("puissance", np.log(x)), ("exponentielle", x)):
        c = np.polyfit(xx, y, 1)
        res = float(np.sqrt(np.mean((y - np.polyval(c, xx)) ** 2)))  # ecart rms sur ln f
        out[nom] = {"exposant_ou_k": float(-c[0]), "ln_A": float(c[1]), "residu_rms_lnf": res}
    out["meilleur"] = min(("puissance", "exponentielle"), key=lambda k: out[k]["residu_rms_lnf"])
    out["fenetre"] = [float(r1), float(r2)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("champ")
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.375)
    ap.add_argument("--e0", type=float, default=0.25)
    ap.add_argument("--d2", type=float, default=1e-3)
    ap.add_argument("--nu", type=float, default=0.0)
    ap.add_argument("--tag", default="noeud")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fen", default="", help="r1,r2 de la fenetre d'ajustement (defaut : 0.19 L a 0.31 L)")
    a = ap.parse_args()

    phi = np.load(a.champ).astype(np.float64)
    N = phi.shape[1]; h = a.h; L = N * h; rmax = L / 2
    e2, e4, eV, rho, theta, j = densities(phi, h, a.mu, a.e0, a.d2, a.nu)
    eV = eV - a.mu ** 2 * (1.0 / (1.0 + a.d2) - 1.0)
    et = e2 + e4 + eV
    c = centre_periodique(et, h)
    X, Y, Z = coords(N, h, c)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    e_rho = sum(fwd(rho, ax, h) ** 2 for ax in (0, 1, 2))
    # base S^2 de la fibration de Hopf : |grad zeta|^2 = (1/4)|grad n|^2 + (Im zeta^+ grad zeta)^2  (exact dans le continu)
    A_, B_, C_, D_ = phi
    nn = np.stack([2 * (A_ * C_ + B_ * D_), 2 * (A_ * D_ - B_ * C_), A_ * A_ + B_ * B_ - C_ * C_ - D_ * D_]) / rho ** 2
    e_n = 0.25 * rho ** 2 * sum(np.sum(fwd(nn, ax, h) ** 2, axis=0) for ax in (1, 2, 3))
    e_ang = rho ** 2 * theta
    jn = np.sqrt((j ** 2).sum(0))
    e_j = jn ** 2 / rho ** 2
    comp = {"e_tot": et, "e2": e2, "e4": e4, "eV": eV, "e_rho": e_rho, "e_ang": e_ang, "e_n": e_n, "e_j": e_j, "abs_j": jn, "Theta": theta}
    tot = float(et.sum() * h ** 3)
    res = {"tag": a.tag, "N": N, "h": h, "L": L, "centre": c.tolist(), "E_tot": tot,
           "totaux": {k: float(v.sum() * h ** 3) for k, v in comp.items() if k not in ("abs_j", "Theta")},
           "ctrl_e_ang_moins_e_n_e_j": float((e_ang - e_n - e_j).sum() * h ** 3)}
    r1, r2 = (float(x) for x in a.fen.split(",")) if a.fen else (0.19 * L, 0.31 * L)

    prof = {}
    for k, f in comp.items():
        rm, m, n = shell_mean(f, r, h, rmax)
        prof[k] = (rm, m)
    res["profils"] = {k: {"r": [round(float(x), 3) for x in rm[::2]], "moy": [float(f"{x:.4e}") for x in m[::2]]} for k, (rm, m) in prof.items()}

    wins = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 15), (15, 17), (17, 19), (19, 21), (21, 23)]
    wins = [w for w in wins if w[1] <= rmax]
    res["pentes_locales"] = {k: {f"{w[0]}-{w[1]}": pente_locale(rm, np.abs(m), *w) for w in wins} for k, (rm, m) in prof.items()}
    res["pente_sur_r"] = {k: {f"{w[0]}-{w[1]}": (None if pente_locale(rm, np.abs(m), *w) is None else pente_locale(rm, np.abs(m), *w) / (0.5 * (w[0] + w[1])))
                              for w in wins} for k, (rm, m) in prof.items() if k in ("Theta", "e_ang", "e_rho")}

    # part de chaque composante dans l'energie hors de R
    res["energie_hors_de_R"] = {}
    for R in [6, 8, 10, 12, 14, 16, 20]:
        if R >= rmax:
            continue
        mask = r > R
        res["energie_hors_de_R"][str(R)] = {k: float(comp[k][mask].sum() * h ** 3) for k in ("e_tot", "e2", "e4", "eV", "e_rho", "e_ang", "e_n", "e_j")}
    # exposant local de E(>R) (energie totale)
    ER = {R: v["e_tot"] for R, v in ((int(k), v) for k, v in res["energie_hors_de_R"].items())}
    ks = sorted(ER)
    res["pente_E_hors_de_R"] = {f"{x}-{y}": float(np.log(ER[y] / ER[x]) / np.log(y / x)) for x, y in zip(ks, ks[1:]) if ER[x] > 0 and ER[y] > 0}

    res["ajustements"] = {k: ajuste(*prof[k], r1, r2) for k in ("abs_j", "e_j", "e_n", "e_tot", "Theta", "e_rho")}
    json.dump(res, open(a.out, "w") if a.out else open("/dev/stdout", "w"), indent=1)

    # resume lisible
    print(f"== {a.tag}: N={N} L={L} Etot={tot:.2f} fenetre d'ajustement {r1:.1f}-{r2:.1f}")
    print("part de e_j / e_tot, e_ang / e_tot, e_rho / e_tot, e4 / e_tot, eV / e_tot dans E(>R):")
    for R, v in res["energie_hors_de_R"].items():
        t = v["e_tot"]
        print(f"  R={R:>2}  E(>R)={t:8.2f}  e_j {v['e_j']/t:6.1%}  e_n {v['e_n']/t:6.1%}  e_ang {v['e_ang']/t:6.1%}  e_rho {v['e_rho']/t:6.1%}  e4 {v['e4']/t:6.1%}  eV {v['eV']/t:6.1%}")
    print("pente log-log de E(>R):", {k: round(v, 2) for k, v in res["pente_E_hors_de_R"].items()})
    for k in ("abs_j", "e_j", "e_n", "e_tot", "Theta"):
        aj = res["ajustements"][k]
        if aj:
            print(f"  ajustement {k:6s}: puissance p={aj['puissance']['exposant_ou_k']:.2f} (res {aj['puissance']['residu_rms_lnf']:.3f}) | exp k={aj['exponentielle']['exposant_ou_k']:.3f} (res {aj['exponentielle']['residu_rms_lnf']:.3f}) -> {aj['meilleur']}")
    print("pentes locales abs_j :", {k: (None if v is None else round(v, 2)) for k, v in res["pentes_locales"]["abs_j"].items()})
    print("pentes locales e_j   :", {k: (None if v is None else round(v, 2)) for k, v in res["pentes_locales"]["e_j"].items()})
    print("pentes locales e_n   :", {k: (None if v is None else round(v, 2)) for k, v in res["pentes_locales"]["e_n"].items()})
    print("pentes locales Theta :", {k: (None if v is None else round(v, 2)) for k, v in res["pentes_locales"]["Theta"].items()})


if __name__ == "__main__":
    main()
