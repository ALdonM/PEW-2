#!/usr/bin/env python3
"""part213 : champ initial de charge de Hopf Q = a*b (modele Z), boite periodique N^3, pas h.
Z = (Z1^a, Z2^b)/norme, avec Z1 = cos f + i sin f cos(th), Z2 = sin f sin(th) e^{i phi} (R^3 -> S^3 par f(r)), f(r) = pi * (1 - tanh((r-R)/w))/2 (f=pi au centre, 0 loin : vide (1,0,0,0)).
usage : python make_QH.py a b N h R w out.npy [sx sy sz]   (sx,sy,sz : etirement de l'ansatz selon x,y,z ; defaut 1 1 1)"""
import sys, numpy as np
a, b, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); h, R, w = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]); out = sys.argv[7]
x = (np.arange(N) - (N - 1) / 2) * h
sx, sy, sz = (float(v) for v in sys.argv[8:11]) if len(sys.argv) >= 11 else (1.0, 1.0, 1.0)
X, Y, Z = np.meshgrid(x / sx, x / sy, x / sz, indexing='ij'); r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-12
f = np.pi * (1 - np.tanh((r - R) / w)) / 2
Z1 = np.cos(f) + 1j * np.sin(f) * Z / r
Z2 = np.sin(f) * (X + 1j * Y) / r
W1, W2 = Z1 ** a, Z2 ** b
n = np.sqrt(np.abs(W1) ** 2 + np.abs(W2) ** 2)
# normalisation a |Z|=1 (le relaxeur ajuste l'amplitude) ; au vide W1=1, W2=0
W1, W2 = W1 / n, W2 / n
phi = np.stack([W1.real, W1.imag, W2.real, W2.imag]).astype(np.float32)
np.save(out, phi); print('ecrit', out, phi.shape, 'min|Z|', float(np.sqrt((phi**2).sum(0)).min()))
