# part176 — protection de la charge de Hopf sur réseau (modèle Z)

Note de cadrage : part175 (barrière d'admissibilité, 7 directions d'arête de Kuhn).
dyn.py = dynamique de part174 sans biais, plus la barrière P = EPSB · Σ B(cos θ), B = (c*−c)²/c sous c*.
Témoins BARRIER=0. Départs admissibles uniquement (chaos ξ0 = 10 graines 1, 2, 4, 11 ; ξ0 = 8 graine 4), nœud ±1 (produit de quaternions).
Job derive_* : MODE none, sans friction, pour la dérive d'énergie avec barrière.
Critères (part175 §3) : degré exact, cos_min et minimum courant, énergie de barrière < 1 %, deg_bloc = deg, taille du nœud.
