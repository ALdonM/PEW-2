# part209 — optimisation de l'acceleration de lagcov (jauge + KF), benchmark local N=64, h=0.5 (24/11/2026)
Cout (s, sandbox) : dK/dq 0.255 ; dE/dq 0.545 ; jvp(dp/dq) 0.125 ; Mv 0.053.
CG non preconditionne (M = matrice de masse, mal conditionnee : bloc E_a ~ 1/g2, bloc Faddeev) : 45 it (tol 1e-4), 77 (1e-6), 109 (1e-8), 186 (1e-13).
CG preconditionne (bloc 4x4 exact par site pour phi, diagonale pour theta ; M_phiphi local, M_thetatheta diagonal par lien) : 9 it (1e-4, err 6e-4), 13 it (1e-6, err 1.7e-5), 19 it (1e-8, err 1.2e-6).
=> ~6x moins d'iterations a precision egale ; erreur 1e-5 sur a avec 13 it (tol 1e-6). Depart a chaud (a precedent) : gain attendu supplementaire.
Note : dyn186c (sans jauge) resout deja M a = rhs par solve 4x4 local exact (une seule evaluation de gradient par pas) ; la jauge ajoute un couplage local phi-theta (E_a (n3-1)) traite par PCG.
A faire : integrateur saut-de-mouton avec acc PCG a depart a chaud ; test de tolerance en dt (0.005 ... 0.04) ; amortissement de l'eponge compatible Gauss (jauge transverse) ; GPU (JAX).

## Test de tolerance en dt (dt_test209.py ; N=64, h=0.5, g2=0.02, nu=0.12, KF=1, noeud boost v=0.1, T=2, boite periodique, CG preconditionne a depart a chaud tol 1e-8, preconditionneur reconstruit tous les 5 pas)
Reference dt=0.005 (le run dt=0.0025 a ete interrompu par un redemarrage de l'environnement a t~1,0).
dt | ||phi-phi_ref||/||phi_ref-vide|| | dH/H final | CG/pas | duree (local)
0.04 | 1.3e-5 | -4.8e-6 | 14.8 | 112 s
0.02 | 5.8e-6 | +5.8e-7 | 14.3 | 199 s
0.01 | 2.7e-6 | +9.6e-7 | 13.2 | 375 s
0.005 | (ref)  | +7.1e-7 | 12.6 | 720 s
=> sur un noeud lisse qui se deplace, dt=0.04 est precis a 1e-5 (8x moins de pas que dt=0.005). L'erreur ne decroit qu'en ~dt^1 (plancher de tolerance CG/preconditionneur probable).
Limites : champ sans rayonnement ni annihilation (modes courts), sans eponge ; max|Gauss|=5e-2 (donnees de depart non coherentes avec Gauss, theta=0).
Cout : ~2,2 s/pas a N=64 en local, soit ~1 h pour T=24 a dt=0.04 (600 pas) a N=64.
