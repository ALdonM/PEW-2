# README part186d — trempe à la Kibble avec ρ aléatoire, boîte plus grande, maillage plus fin — STATUT : lancé le 18/11/2026

Suite de part186c (résultat négatif : n aléatoire, ρ = 1 figé, 64³, h = 0,5, aucun nœud persistant).
Question (M. Aldon) : si ρ est aléatoire au départ, les zones où ρ est petit servent-elles de germes aux nœuds ?
Remarque de cadrage : Z a 4 composantes réelles pour 3 dimensions d'espace, donc les zéros de ρ sont de codimension 4 : génériquement absents de l'espace à un instant donné, isolés dans l'espace-temps (événements). Le terme 1/(ρ²+d2) les interdit de toute façon (ρ_min ≈ 0,25–0,3 sur réseau). Le test regarde donc si des régions de ρ petit favorisent un changement de charge, pas des zéros exacts.
Protocole : INIT chaos (ρ et direction aléatoires, RMS ρ = 1), corrélation XI0 = τ0, a = √(τ/τ0), friction de Hubble 1,5/τ, sans bain, μ = 0,375, e0 = 0,25. Boîte L = 38,4 (contre 32).
Runs : A = N 96, h 0,4 : τ0 ∈ {4, 8} × 2 graines (T_END = 8 τ0) + témoin sans expansion (τ0 = 4) ; B = N 128, h 0,3 (même boîte) : τ0 = 8, 1 graine (convergence en h, grossière : la réalisation du champ aléatoire diffère de celle de A).
Critères écrits avant les résultats :
1. Nœud « né » : compte_noeuds avec order ≥ 0,85 ET cos_min > 0 ; charge ±1 conservée ≥ 10 unités de temps.
2. Comparaison avec part186c (ρ = 1) : plus de nœuds persistants ? Compter n+ et n− en fin de run et leur durée de vie ; charge nette vs √N_tot.
3. Si une charge apparaît : ρ_min et cos_min au moment du changement (germe de ρ petit ou saut de réseau).
4. Convergence : même résultat qualitatif (aucun / quelques nœuds) entre h = 0,4 et h = 0,3 ; sinon la perte ou la naissance est un effet de réseau.
5. Résultat négatif possible et informatif.
Limites : mesures toutes les 1,0 unité (un événement de charge plus court est invisible) ; 1 seule graine pour B ; a_max ≈ 3 ; a(t) imposé ; un seul modèle (Z sans terme de brisure de symétrie nœud/anti-nœud).
