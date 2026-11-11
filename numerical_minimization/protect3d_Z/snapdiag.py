"""part183 -- diagnostic 3D reduit d'un instantane du modele Z (pour ne pas dependre des artefacts).
Usage : python snapdiag.py H out.jsonl snap1.npy [snap2.npy ...]   (phi : (4,N,N,N), float16/32 ; L = N*H, periodique)
Pour chaque instantane : densite g = |grad phi|^2 (4 composantes, differences centrees), lissee (sigma = 1 maille) ;
 - pic maximal ; second pic = maximum local le plus eleve a distance >= DMIN du premier (image periodique la plus proche) ;
 - vecteur de separation (dx,dy,dz), composante selon x et module perpendiculaire a x ;
 - hauteur du second pic / premier ; valeurs propres du tenseur d'inertie de g sur {g > 0.25 gmax} (axes de l'objet).
Sortie : une ligne JSON par instantane (petite : quelques centaines d'octets).
"""
import sys, json, re
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

h = float(sys.argv[1]); out = sys.argv[2]; files = sys.argv[3:]
DMIN = 3.0   # unites physiques (= 4 mailles a h = 0,5)

def wrap(d, L): return d - L * np.round(d / L)

with open(out, "a") as fo:
    for f in files:
        phi = np.load(f).astype(np.float32); N = phi.shape[1]; L = N * h
        g = sum(((np.roll(phi, -1, axis=ax) - np.roll(phi, 1, axis=ax)) / (2 * h))**2 for ax in (1, 2, 3)).sum(0)
        import os; SIG = float(os.environ.get("SIG", "2.0")); gs = gaussian_filter(g, sigma=SIG, mode="wrap")
        ip = np.unravel_index(np.argmax(gs), gs.shape); g1 = float(gs[ip])
        p1 = np.array(ip) * h
        # maxima locaux (fenetre 5 mailles, periodique), tries
        lm = (gs == maximum_filter(gs, size=9, mode="wrap")) & (gs > 0.05 * g1)
        cand = np.argwhere(lm)
        best = None
        for c in cand:
            p = c * h; d = wrap(p - p1, L)
            if np.linalg.norm(d) >= DMIN and (best is None or gs[tuple(c)] > best[0]): best = (float(gs[tuple(c)]), d, p)
        rec = dict(file=f.split("/")[-1], t=int(re.search(r"_t(\d+)", f).group(1)) if re.search(r"_t(\d+)", f) else None,
                   gmax=round(g1, 3), p1=[round(float(x), 2) for x in p1], nloc=int(len(cand)))
        if best is not None:
            d = best[1]
            rec.update(g2_over_g1=round(best[0] / g1, 3), sep=[round(float(x), 2) for x in d], sep_x=round(float(d[0]), 2),
                       sep_perp_x=round(float(np.hypot(d[1], d[2])), 2), sep_norm=round(float(np.linalg.norm(d)), 2))
        # inertie de g sur l'objet (autour du pic max, periodique)
        mask = gs > 0.5 * g1
        idx = np.argwhere(mask); w = gs[mask]
        dd = wrap(idx * h - p1, L); mu = (w[:, None] * dd).sum(0) / w.sum(); dd = dd - mu
        C = (w[:, None, None] * dd[:, :, None] * dd[:, None, :]).sum(0) / w.sum()
        ev, evec = np.linalg.eigh(C)
        rec.update(inertia_eig=[round(float(x), 2) for x in ev], long_axis=[round(float(x), 2) for x in evec[:, 2]],
                   ncells=int(mask.sum()))
        fo.write(json.dumps(rec) + "\n"); fo.flush(); print(json.dumps(rec), flush=True)
