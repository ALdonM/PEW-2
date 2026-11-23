"""part201 -- depart d'une paire de noeuds wk_T0 (produit quaternionique, orientation beta0) dans un reseau N^3 (N=96, h=0,5).
Usage (depuis protect3d_Z) : python make_pair201.py SORTIE.npy SEP_MAILLES OPP    (OPP=1 : noeud + anti-noeud par reflexion x->-x ; OPP=0 : memes chiralites)
Meme construction que pairs_Z/make_pair.py (part181), mais N=96 au lieu de 128."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pairs_Z'))
import pair
out, sep, opp = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
knot = np.load('../../resultats/part179/wk_T0/wk_T0_champ_final.npy').astype(np.float64)
N = 96; c = (N - 1) / 2
A = pair.embed(knot, N, (c - sep / 2, c, c))
kB = knot[:, ::-1, :, :] if opp else knot
B = pair.embed(kB, N, (c + sep / 2, c, c))
P = pair.qmul(A, B)
np.save(out, P.astype(np.float32)); print('paire ecrite', out, P.shape, 'sep', sep, 'opp', opp)
