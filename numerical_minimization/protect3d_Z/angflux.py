"""part183 -- test 'deux photons dos a dos' sur une annihilation nœud + antinœud, a partir de proj.npy (densite d'energie integree selon z).
Usage : python angflux.py H history.jsonl proj.npy [R1 R2]
Pour chaque mesure : centre = centre du plus gros amas (history) ; energie dans la couronne R1<r<R2 (unites physiques, defaut 10-28,
< L/2 pour eviter les images periodiques) repartie en 16 secteurs d'angle dans le plan xy ;
 harmoniques a1 (dipole), a2 (quadrupole = deux lobes opposes) normalises par la moyenne, angle de l'axe du quadrupole,
 part de l'energie de la couronne contenue dans les deux secteurs opposes les plus charges (isotrope : 2/16 = 0,125).
LIMITE : la projection selon z masque des lobes dirigés selon z ; la densite d'energie de couronne contient aussi le bruit de fond.
On soustrait la mediane de la couronne a la premiere mesure (fond) pour ne garder que l'excedent.
"""
import sys, json
import numpy as np
h = float(sys.argv[1]); hist = [json.loads(l) for l in open(sys.argv[2])]; P = np.load(sys.argv[3]).astype(np.float64)
R1 = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0; R2 = float(sys.argv[5]) if len(sys.argv) > 5 else 28.0
N = P.shape[1]; L = N * h
x = (np.arange(N) + 0.0) * h
X, Y = np.meshgrid(x, x, indexing="ij")
meas = hist[-len(P):]   # une mesure de proj par mesure d'historique (les dernieres)
bg = None
for k, m in enumerate(meas):
    cx, cy = (m['centres'][0][0], m['centres'][0][1]) if m['centres'] else (L / 2, L / 2)
    dx = (X - cx + L / 2) % L - L / 2; dy = (Y - cy + L / 2) % L - L / 2
    r = np.hypot(dx, dy); ang = np.arctan2(dy, dx); ann = (r > R1) & (r < R2)
    E = P[k]
    if bg is None: bg = np.median(E[ann])
    ex = np.clip(E - bg, 0, None)
    sect = ((ang + np.pi) / (2 * np.pi) * 16).astype(int) % 16
    hs = np.array([ex[ann & (sect == s)].sum() for s in range(16)])
    tot = hs.sum()
    if tot <= 0: print(m['t'], "excedent nul"); continue
    th = (np.arange(16) + 0.5) * 2 * np.pi / 16 - np.pi
    a1 = np.abs((hs * np.exp(1j * th)).sum()) / tot; z2 = (hs * np.exp(2j * th)).sum() / tot
    opp = hs + np.roll(hs, 8); j = int(np.argmax(opp[:8])); two = opp[j] / tot
    print(json.dumps(dict(t=m['t'], Qabs=m['Qabs'], excedent=round(float(tot), 2), a1=round(float(a1), 3), a2=round(float(abs(z2)), 3),
                          axe_quadrupole_deg=round(float(np.degrees(np.angle(z2) / 2)), 1), part_2_lobes_opposes=round(float(two), 3),
                          iso=0.125)))
