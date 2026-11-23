# part201 — le terme de Faddeev temporel (KF) change-t-il la dynamique d'une paire de nœuds ?

Statut : [D], prédictions écrites AVANT les résultats (23/11/2026, 10 h 50).

## Question
Sans KF, l'inertie d'un nœud en translation vaut ≈ 199 pour E ≈ 788 (part199) ; avec KF, M ≈ 878 (part200). Les paires de part181/part184 (modèle L0/L1) ont été simulées sans KF. Si KF change fortement l'annihilation ou la répulsion, ces résultats dynamiques sont à refaire. Sinon une note suffit.

## Dispositif
- Code : `dyn186c.py` (KF intégré, audité en part200 contre l'Euler-Lagrange exact, 1,8·10⁻⁶). Modèle L0 : NU=0, sans jauge, CP=0, poids = 1. Ce n'est PAS L1 ni L2.
- Départ identique pour KF=0 et KF=1 (`make_pair201.py` : produit quaternionique de deux nœuds wk_T0, orientation beta0, séparation 24 mailles = 12 unités, N=96, h=0,5, vitesse nulle). Vérifié : Q = −0,001 / Qabs = 1,825 (opp) et Q = −1,958 (same), comme part181.
- Éponge sphérique SPR0=18 → SPRB=24 ; T = 24 ; mesure toutes les 0,5.
- Jobs : opp_kf0, opp_kf1 (dt 0,0025), opp_kf1_dt5 (dt 0,005 : contrôle d'ordre de l'intégrateur), same_kf0, same_kf1.
- Référence : part181 (N=128, L=64, sans éponge, t0=10) : opp s24 passe Qabs < 0,5 à t − t0 ≈ 5,7 (Qabs 1,825 → 0,43 en 5,5) ; same s24 : séparation des pics 10,5 → ~32 (répulsion, non quantifiée).

## Observables
- t_ann = premier t où Qabs < 0,5 (opp). Énergie cinétique pic (Ekm).
- same : séparation des deux pics de la projection d'énergie (proj.npy) le long de x à t = 6, 12, 20.
- Conservation : Ekm + E2 + E4 + EV (Etot de l'historique n'est pas l'énergie conservée avec KF).

## Prédictions pré-enregistrées
- P0 (contrôle) : opp_kf0 reproduit part181 : t_ann ∈ [5,1 ; 6,3] (±10 %). Sinon, le dispositif (N=96, éponge) n'est pas comparable et rien d'autre n'est interprétable.
- P1 : t_ann(KF=1)/t_ann(KF=0) ∈ [1,3 ; 2,3]. Raisonnement : un nœud rigide donne √(878/199) = 2,1 ; une partie du mouvement initial est de la relaxation interne du produit de deux nœuds, pas de la translation, donc je m'attends à moins que 2,1. Estimation centrale 1,7. Réfutée si < 1,2 (KF négligeable pour l'annihilation) ou > 2,5.
- P2 : l'annihilation a lieu avec KF=1 avant T=24 (Qabs < 0,5). Incertain : une inertie plus grande ne supprime pas l'attraction, mais la paire pourrait rebondir ou se disperser autrement.
- P3 : same : séparation parcourue à t=12 avec KF=1 / avec KF=0 ∈ [0,2 ; 0,5] (force identique, inertie ~4× plus grande : 0,23 si rigide ; moins de contraste si une partie du mouvement est interne).
- P4 : contrôle dt : t_ann(KF=1, dt=0,005) = t_ann(KF=1, dt=0,0025) à ±5 %. Sinon l'intégrateur d'ordre 1 de KF est insuffisant pour cette dynamique.
- P5 : dérive de Ekm+E2+E4+EV < 1 % sur KF=1 (perte seulement, comme en part200).

## À ne pas conclure d'avance
Aucune identification physique. Une seule réalisation, un seul h, un seul départ, une seule orientation (beta0). Si KF change t_ann de plus de 30 %, les résultats de part181/184 sont « sans inertie quartique » et à refaire ; sinon ils restent valables à cette précision pour cette orientation seulement.
