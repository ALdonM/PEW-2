# part209 — optimisation de l'acceleration de lagcov (jauge + KF), benchmark local N=64, h=0.5 (24/11/2026)
Cout (s, sandbox) : dK/dq 0.255 ; dE/dq 0.545 ; jvp(dp/dq) 0.125 ; Mv 0.053.
CG non preconditionne (M = matrice de masse, mal conditionnee : bloc E_a ~ 1/g2, bloc Faddeev) : 45 it (tol 1e-4), 77 (1e-6), 109 (1e-8), 186 (1e-13).
CG preconditionne (bloc 4x4 exact par site pour phi, diagonale pour theta ; M_phiphi local, M_thetatheta diagonal par lien) : 9 it (1e-4, err 6e-4), 13 it (1e-6, err 1.7e-5), 19 it (1e-8, err 1.2e-6).
=> ~6x moins d'iterations a precision egale ; erreur 1e-5 sur a avec 13 it (tol 1e-6). Depart a chaud (a precedent) : gain attendu supplementaire.
Note : dyn186c (sans jauge) resout deja M a = rhs par solve 4x4 local exact (une seule evaluation de gradient par pas) ; la jauge ajoute un couplage local phi-theta (E_a (n3-1)) traite par PCG.
A faire : integrateur saut-de-mouton avec acc PCG a depart a chaud ; test de tolerance en dt (0.005 ... 0.04) ; amortissement de l'eponge compatible Gauss (jauge transverse) ; GPU (JAX).
