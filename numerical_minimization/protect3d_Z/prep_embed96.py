"""Plonge le noeud part179 (64^3, L=32) au centre d'une boite 96^3 (L=48) de vide (1,0,0,0) ; sortie float32."""
import numpy as np, sys
P = np.load(sys.argv[1]).astype(np.float64); N = int(sys.argv[3]); out = np.zeros((4, N, N, N)); out[0] = 1.0
a = (N - P.shape[1]) // 2; out[:, a:a+P.shape[1], a:a+P.shape[2], a:a+P.shape[3]] = P
np.save(sys.argv[2], out.astype(np.float32)); print('embed', out.shape)
