#!/usr/bin/env python3
"""Tests de halo_mesure.py sur des champs analytiques (verite connue). Usage : python test_halo.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import halo_mesure as H

N, h = 96, 0.5
L = N * h
c = np.zeros(3)
X, Y, Z = H.coords(N, h, c)
r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
ok = True


def check(nom, val, ref, tol):
    global ok
    bon = abs(val - ref) <= tol
    ok &= bon
    print(("OK  " if bon else "FAIL"), nom, "mesure %.4f  attendu %.4f  (tol %.3f)" % (val, ref, tol))


# 1. pente d'une loi de puissance
rr = np.maximum(r, 0.25)
for p in (2.0, 4.0, 6.0):
    f = rr ** (-p)
    rm, fm, _ = H.shell_mean(f, r, h, L / 2)
    check("pente r^-%g (fenetre 6-9)" % p, H.pente_locale(rm, fm, 6, 9), -p, 0.05)

# 2. pente d'une exponentielle : -k r
k = 0.4
f = np.exp(-k * r)
rm, fm, _ = H.shell_mean(f, r, h, L / 2)
check("pente exp(-0.4 r) a r=8 (fenetre 7-9)", H.pente_locale(rm, fm, 7, 9), -k * 8, 0.1)

# 3. multipole l=2 : f = 1 + a P2(cos theta), rms_l2 / moyenne = a / sqrt(5)
a = 0.2
P2 = (3 * Z ** 2 / np.maximum(r ** 2, 1e-9) - 1) / 2
f = 1 + a * P2
m = H.multipoles_coquille(f, X, Y, Z, r, 6, 8)
check("rms_l2 / moyenne", m["rms_l2"] / m["moyenne"], a / np.sqrt(5), 0.003)
check("rms_l1 (doit etre ~0)", m["rms_l1"] / m["moyenne"], 0.0, 0.003)
check("rms_l3 (doit etre ~0)", m["rms_l3"] / m["moyenne"], 0.0, 0.003)

# 4. multipole l=1 : f = 1 + b cos theta, rms_l1/moyenne = b / sqrt(3)
b = 0.3
f = 1 + b * X / np.maximum(r, 1e-9)
m = H.multipoles_coquille(f, X, Y, Z, r, 6, 8)
check("rms_l1 / moyenne (dipole selon x)", m["rms_l1"] / m["moyenne"], b / np.sqrt(3), 0.003)

# 5. energie cumulee : densite gaussienne e = exp(-r^2/s^2), E(<R)/E(tot)
s = 4.0
e = np.exp(-r ** 2 / s ** 2)
etot = e.sum() * h ** 3
rc, cum = H.cumul(e, r, h, L / 2, etot)
from scipy.special import erf
R = 6.0
exact = (erf(R / s) - 2 * R / (s * np.sqrt(np.pi)) * np.exp(-R ** 2 / s ** 2)) / 1.0  # fraction de la masse totale (infinie)
i = int(round(R / h)) - 1
check("E(<6)/Etot pour une gaussienne s=4", cum[i], exact, 0.01)

# 6. centre periodique : un blob decale de (3, -2, 5), traverse le bord
cc = np.array([3.0, -2.0, 5.0])
Xs, Ys, Zs = H.coords(N, h, cc)
e = np.exp(-(Xs ** 2 + Ys ** 2 + Zs ** 2) / 4.0)
cm = H.centre_periodique(e, h)
check("centre periodique (x)", cm[0], cc[0], 0.05)
check("centre periodique (y)", cm[1], cc[1], 0.05)
check("centre periodique (z)", cm[2], cc[2], 0.05)

# 7. fonctionnelle : le vide (1,0,0,0) a une energie nulle ; un hopfion de zlat est un champ de degre 1 d'energie finie positive
vide = np.zeros((4, 32, 32, 32)); vide[0] = 1.0
e2, e4, eV, rho, th, j = H.densities(vide, 0.5, 0.375, 0.25, 1e-3)
e_vac = 0.375 ** 2 * (1.0 / (1.0 + 1e-3) - 1.0)
check("densite d'energie du vide moins e_vide (doit etre 0)", float(np.abs(e2 + e4 + eV - e_vac).max()), 0.0, 1e-9)

print("\nTOUS LES TESTS PASSENT" if ok else "\nECHEC")
sys.exit(0 if ok else 1)
