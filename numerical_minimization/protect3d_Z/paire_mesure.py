"""part187f -- energie au-dessus du vide d'un champ (paire ou noeud isole), charge de Hopf par degre exact, E_int = E(paire) - 2 E(seul).
usage : python paire_mesure.py paire.npy --seul noeud.npy --nu 0.12 --h 0.5 --tag T --out T.json"""
import argparse, json, numpy as np
from halo_mesure import densities


def total(phi, h, mu, e0, d2, nu):
    e2, e4, eV, rho, Th, j = densities(phi, h, mu, e0, d2, nu)
    evac = mu**2 * (1.0 / (1.0 + d2) - 1.0)
    return dict(E2=float(e2.sum() * h**3), E4=float(np.sum(e4) * h**3), EV=float((eV - evac).sum() * h**3), rho_min=float(np.min(rho)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('champ'); ap.add_argument('--seul', required=True)
    ap.add_argument('--h', type=float, default=0.5); ap.add_argument('--mu', type=float, default=0.375)
    ap.add_argument('--e0', type=float, default=0.25); ap.add_argument('--d2', type=float, default=1e-3)
    ap.add_argument('--nu', type=float, default=0.0); ap.add_argument('--tag', default=''); ap.add_argument('--out', default='')
    a = ap.parse_args()
    P = total(np.load(a.champ), a.h, a.mu, a.e0, a.d2, a.nu)
    S = total(np.load(a.seul), a.h, a.mu, a.e0, a.d2, a.nu)
    Ep = P['E2'] + P['E4'] + P['EV']; Es = S['E2'] + S['E4'] + S['EV']
    try:
        import degre
        Q = float(degre.degree(np.load(a.champ)))
    except Exception as ex:
        Q = None
    r = dict(tag=a.tag, E_paire=Ep, E_seul=Es, E_int=Ep - 2 * Es, Q=Q, paire=P, seul=S)
    print('%s : E_paire=%.3f  E_seul=%.3f  E_int=%.3f  Q=%s  rho_min=%.3f' % (a.tag, Ep, Es, Ep - 2 * Es, Q, P['rho_min']))
    if a.out: json.dump(r, open(a.out, 'w'), indent=1)
