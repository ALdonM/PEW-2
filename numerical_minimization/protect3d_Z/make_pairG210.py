"""part210 -- paire jaugee N=64, h=0.5 : noeud (L0, relaxe) + anti-noeud, separation SEP mailles. Usage : python make_pairG210.py SORTIE.npy SEP OPP"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pairs_Z'))
import pair
out, sep, opp = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
knot = np.load('../../resultats/part203/knot_L32_h0.5_newton.npy').astype(np.float64); N = 64; c = (N - 1) / 2
A = pair.embed(knot, N, (c - sep / 2, c, c)); B = pair.embed(knot[:, ::-1, :, :] if opp else knot, N, (c + sep / 2, c, c))
np.save(out, pair.qmul(A, B)); print('paire', out, sep, opp)
