"""part210 -- depart d'une paire de noeuds relaxes a h=0,4 (N=120, L=48), meme construction que make_pair201.py.
Usage (depuis protect3d_Z) : python make_pair210.py SORTIE.npy SEP_MAILLES OPP   (sep 30 mailles = 12 unites)"""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pairs_Z'))
import pair
out, sep, opp = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
knot = np.load('../../resultats/part210/knot_L32_h0.4_newton_nu0.npy').astype(np.float64)   # noeud L0 (nu=0) relaxe a h=0,4, boite L=32 (N=80)
N = 120; c = (N - 1) / 2
A = pair.embed(knot, N, (c - sep / 2, c, c)); kB = knot[:, ::-1, :, :] if opp else knot
B = pair.embed(kB, N, (c + sep / 2, c, c)); P = pair.qmul(A, B)
np.save(out, P.astype(np.float32)); print('paire ecrite', out, P.shape, 'sep', sep, 'opp', opp)
