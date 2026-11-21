"""part187f -- depart d'une paire de noeuds a cœurs figes, a partir d'un noeud isole deja relaxe (meme boite N^3, meme h).
Produit : depart_<tag>.npy (4,N,N,N) float32 et libre_<tag>.npy (N,N,N) : 0 dans les boules de rayon Rc autour des deux centres (cœurs figes
a la valeur du noeud isole, sans contamination par l'autre noeud), 1 ailleurs.
Convention de jauge.py (part184) : produit de quaternions A*B ; Q oppose = miroir de B (z -> -z pour --dir z ; x -> -x pour x et diag).
Les decalages sont des nombres entiers de mailles : separation d = 2 s h (z, x) ou 2 sqrt2 s h (diag = (1,1,0)).
usage : python paire_depart.py noeud.npy --h 0.5 --dir z|x|diag --s 24 --charge opp|same --rc 5 --tag T"""
import argparse, numpy as np


def qprod(A, B):
    a0, a1, a2, a3 = A; b0, b1, b2, b3 = B
    return np.stack([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2, a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0])


def construire(k1, h, direction, s, charge, rc):
    N = k1.shape[1]
    k1 = k1.astype(np.float32)
    def deplace(f, sh):   # translation entiere de sh mailles le long de la direction
        if direction == 'z': return np.roll(f, sh, axis=3)
        if direction == 'x': return np.roll(f, sh, axis=1)
        return np.roll(np.roll(f, sh, axis=1), sh, axis=2)   # diag (1,1,0)
    Bsrc = k1
    if charge == 'opp':
        Bsrc = np.ascontiguousarray(k1[:, :, :, ::-1]) if direction == 'z' else np.ascontiguousarray(k1[:, ::-1])
    A = deplace(k1, -s); B = deplace(Bsrc, +s)
    p = qprod(A, B)
    x1 = (np.arange(N) - N / 2 + 0.5) * h                       # meme grille que halo_relax (maille centree)
    X, Y, Z = np.meshgrid(x1, x1, x1, indexing='ij')
    def centre(sgn):
        c = sgn * s * h
        return (0.0, 0.0, c) if direction == 'z' else ((c, 0.0, 0.0) if direction == 'x' else (c, c, 0.0))
    libre = np.ones((N, N, N), dtype=np.float32)
    for sgn, champ in ((-1, A), (+1, B)):
        cx, cy, cz = centre(sgn)
        # distances periodiques a la boite
        L = N * h
        dx = (X - cx + L / 2) % L - L / 2; dy = (Y - cy + L / 2) % L - L / 2; dz = (Z - cz + L / 2) % L - L / 2
        m = (dx**2 + dy**2 + dz**2) <= rc**2
        p[:, m] = champ[:, m]
        libre[m] = 0.0
    return p.astype(np.float32), libre


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('noeud'); ap.add_argument('--h', type=float, default=0.5)
    ap.add_argument('--dir', choices=['z', 'x', 'diag'], required=True)
    ap.add_argument('--s', type=int, required=True); ap.add_argument('--charge', choices=['opp', 'same'], required=True)
    ap.add_argument('--rc', type=float, default=5.0); ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    k1 = np.load(a.noeud)
    p, libre = construire(k1, a.h, a.dir, a.s, a.charge, a.rc)
    d = (2 if a.dir != 'diag' else 2 * 2 ** 0.5) * a.s * a.h
    print('paire %s %s : s=%d mailles, separation d=%.2f, %d sites figes sur %d' % (a.dir, a.charge, a.s, d, int((libre == 0).sum()), libre.size))
    np.save('depart_%s.npy' % a.tag, p); np.save('libre_%s.npy' % a.tag, libre)
