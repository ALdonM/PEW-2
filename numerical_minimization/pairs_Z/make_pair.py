"""part181 -- fabrique le champ initial d'une paire de noeuds (produit quaternionique). Usage : make_pair.py SORTIE.npy SEP_MAILLES ORIENT OPP
ORIENT in {beta0, pi_z, pi_x, pi2_x} ; OPP = 0 (meme chiralite, Q=-2) ou 1 (noeud + anti-noeud par reflexion x->-x, Q=0)."""
import sys, numpy as np
import pair
out, sep, orient, opp = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
G = {'beta0': None, 'pi_z': pair.unit_q([0, 0, 1], np.pi), 'pi_x': pair.unit_q([1, 0, 0], np.pi), 'pi2_x': pair.unit_q([1, 0, 0], np.pi / 2)}[orient]
knot = np.load('../../resultats/part179/wk_T0/wk_T0_champ_final.npy').astype(np.float64)
N = 128
c = (N - 1) / 2
A = pair.embed(knot, N, (c - sep / 2, c, c))
kB = knot[:, ::-1, :, :] if opp else knot          # reflexion x -> -x : inverse le degre
B = pair.embed(kB, N, (c + sep / 2, c, c))
if G is not None: B = pair.conj_by(B, G)
P = pair.qmul(A, B)
np.save(out, P.astype(np.float32))
print('paire ecrite', out, P.shape, 'sep', sep, orient, 'opp', opp)
