# README part186c — trempe à la Kibble : des nœuds naissent-ils quand l'horizon grandit ? — STATUT : lancé le 18/11/2026

Demande (M. Aldon, 18/11) : coller à la cosmologie standard pour le début ; seule la porteuse (espace-temps par vitesse de phase) s'ajoute. Question : le temps qui se déploie joue-t-il un rôle dans la création de nœuds / anti-nœuds ?

Pourquoi pas un bain chaud : essai préalable (dyn186c, N=16) : sous bruit de Langevin, ρ gonfle comme √(T)/μ et raidit la direction n (ordre 0,97 dès t=4). Le potentiel μ²ρ² n'a pas de phase symétrique à haute température, d'où l'ordre « avant » le refroidissement de part179 §4. Le désordre est donc imposé comme condition initiale (Kibble), pas thermalisé.
Protocole : n aléatoire (amplitude ρ = 1 partout), longueur de corrélation XI0 = τ0 (≈ horizon), expansion radiative a = √(τ/τ0), friction de Hubble 1,5/τ, aucun bain. Boîte 64³, h = 0,5 (L = 32), e0 = 0,25, μ = 0,375 (nœud R ≈ 5). Balayage τ0 ∈ {2, 4, 8} (T_END = 8 τ0, a_max ≈ 2,8), 2 graines chacun, + témoin sans expansion (a = 1, sans friction, XI0 = 4).
Critères écrits avant les résultats :
1. Nœud « né » : compte_noeuds avec order ≥ 0,85 ET cos_min > 0 (leçon part179 §4), charge ±1 conservée ≥ 10 unités de temps, deg_bloc cohérent.
2. Nombre de nœuds + anti-nœuds en fin de run, en fonction de τ0 : comparer à (L / 2τ0)³ (volumes d'horizon dans la boîte). Si le temps joue un rôle (Kibble), le nombre doit décroître avec τ0 ; le témoin dit si l'expansion compte.
3. Charge nette vs √N_tot : un écart ≫ √N est un biais (artefact de réseau ou de boîte) puisqu'aucun terme brisant la symétrie n'est présent.
4. Perte de charge : cos_min au moment de la perte (saut de réseau si < 0), comme part186b.
5. Résultat négatif possible et informatif : aucun nœud persistant → la friction de Hubble seule ne suffit pas à les produire dans ce modèle.
Limites : un seul h (convergence en h non établie, part180b) ; a_max ≈ 2,8 imposé par la taille de boîte (rayon comobile du nœud = a·R) ; amplitude ρ figée à 1 au départ ; statistiques faibles (2 graines) ; a(t) imposé, la porteuse n'est pas dynamique.
