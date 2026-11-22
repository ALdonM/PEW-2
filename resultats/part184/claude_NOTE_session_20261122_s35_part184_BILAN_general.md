# part184 — bilan général des tests de paires de nœuds (modèle Z, L1) [D]

Session s35, 22/11/2026. Synthèse faite à partir des en-têtes et tableaux des notes part184d à ze8 (lecture rapide, pas de recalcul). Modèle Z classique ; aucune identification à une particule (discipline D). Lagrangien simulé : L1 = |∂Z1|² + |DZ2|² + cF ΣΩ² + μ²(ρ²+1/(ρ²+d²)−2) + 2ν²|Z2|² + (1/2g²)F² (μ=0,375, ν=0,12, g²=0,02, e0=0,25, d²=1e-3). Ce n'est PAS le Lagrangien « milieu fixé » L2 (dyn186c, part189-198) : aucun résultat ci-dessous n'en dit quoi que ce soit (voir claude_NOTE_session_20261122_synthese_Lagrangiens_code.md).

## Établi
- Une paire de nœuds de charge de Hopf opposée s'annihile en t ≈ 15 à 26 selon l'orientation et g².
- Rayonnement : anneau équatorial avec quadrupôle. Les « deux lobes » ne sont pas établis (d, e).
- Le retard « diagonal » de e venait d'une erreur de miroir dans la construction de B (f), pas du réseau. Effet de J : ~1 %, simple retard (f).
- Rotation interne ω : retard d'annihilation croissant plus vite que ω² (exposant local ≈ 2,6). Spin-spin et somme des carrés rejetés ; (L_A+L_B)² le plus proche ; J_net n'est pas le moteur (g, h, i). Voir réserves.
- Nœud stationnaire : sans jauge, quasi stationnaire en rotation (xy). Avec jauge, seulement après relaxation de l'équilibre magnétostatique (énergie −16 %) (zza).
- Méthode : absorbeur sphérique réfléchit ≤ 0,4 % (cube périodique : 8 % matière, 20 % jauge) ; jauge transverse préserve Gauss (ze). N=96 → 128 : t_ann −0,5 à −1,8.
- Relaxation avec jauge N=128 : converge à ω=0,10 et 0,14 ; dérive à 0,19 (charge 392 → 481 sur 150 unités).
- ze8 (ω=0,19, z Jo, N=128) : t_ann décroît de façon lisse avec la charge relaxée q0 (341→18,54 ; 344,5→17,64 ; 346,9→16,99 ; 348,5→16,65 ; 349,5→16,34 ; + 336→20,25, 350→16,03). Pas de seuil. Le « creux » de z Jo à ω=0,19 est un effet de durée de relaxation (donc de charge).

## Fragile / à ne pas citer
- Séries part184o à w (ω=0,25) : au-dessus du seuil de masse √2ν ≈ 0,17, nœuds non stationnaires ; celles à g²>0 partaient hors équilibre magnétostatique. Lois de retard en ω et rejets de modèles à refaire.
- Hypothèse « ω>ν fait gonfler Z2 » : réfutée par ω=0,14 > ν qui converge. Seuil éventuel entre 0,14 et 0,19 (inférence, mode mou, non testée).
- Une réalisation par point ; bruit ≈ 0,15 estimé par des cas voisins.
- Résolution h jamais testée (seulement taille de boîte).
- Piège de nommage : x_par=Jo, x_anti=Js ; z_par=Js, z_anti=Jo.

## À faire (par priorité)
1. t_ann à q0 égal pour ω = 0,10 / 0,14 / 0,19 (paires stationnaires avec équilibre magnétostatique, N=128) : le retard dû au spin est-il un effet de ω ou de charge ?
2. Convergence en h (0,5 contre 0,4) sur une paire de référence.
3. Dépendance en g² et en ν sur cette paire.
4. Stabilité dynamique d'un nœud isolé (perturbation à ω=0,10 et 0,14, absorbeur). Sans ce test : états stationnaires, pas démontrés stables.
5. Code unifié L0/L1/L2 : ajouter ν à dyn186c.py, vérifier E=931,32 à ρ_vac=1, CP=0.
