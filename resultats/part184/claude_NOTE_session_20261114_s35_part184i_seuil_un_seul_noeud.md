# NOTE — session 35 (14–15/11/2026) — part184i : loi du retard en ω, un seul nœud tournant

**Statut : AUDITÉ par un agent isolé** (recalcul depuis les journaux, lecture de `jauge.py`). Rien n'est publié ; rien n'est identifié à une particule (option (D)). Complète les notes part184g et part184h.
Run GitHub 43610438117 ; données `resultats/part184/gh_part184i/` ; workflow `pew-part184i-seuil-omega.yml` ; `jauge.py` : `SJ=one` (s2=0, seul le nœud A tourne), `CANCEL_PZ=1`. Paire sur x, sép. 20, N=96, T=38, g²=0,02, ν=0,12.
Cas : R1 ω=0,25, R2 ω=0,15, R3 ω=0,1 (spins parallèles, deux nœuds) ; R4 ω=0,2, R5 ω=0,3 (un seul nœud tournant).

## Résultats (audités)
- **Retard d'annihilation** (Qabs<0,5, interpolé ; indépendant de la méthode d'interpolation à 0,02 près ; bruit entre réalisations ≈ 0,15 d'après P3/P2/Q3/Q4 à 24,56–24,71) : ω=0 : 24,71 ; 0,1 : 25,72 (+1,01) ; 0,15 : 27,26 (+2,55) ; 0,2 : 30,06 (+5,35) ; 0,25 : 36,54 (+11,8 ; Qabs final 0,222, donc franchi à 1,5 de la fin) ; 0,3 : jamais (Qabs 1,945 à t=38).
- **Loi en ω :** retard/ω² = 101, 113, 134, 189 : **ω² pur rejeté**, exposant local ≈ 2,6 ; croissance plus rapide que ω² : retenu. Une divergence à ω_c (≈0,35–0,38 selon la forme) **n'est pas démontrée** : 4 points, forme choisie, et Q2 (ω=0,3) ne contraint rien (retard extrapolé ≈33 > T=38).
- **Un seul nœud tournant vs deux (même Lz total) :** Lz initial linéaire en ω (−2215·ω pour deux nœuds) ; R3=R4=−221,51 ; R2=R5=−332,26. Retards : R4 0,91 vs R3 1,01 (rapport 0,90) ; R5 2,04 vs R2 2,55 (0,80). B immobile dans R4/R5 (s2=0 ; boost nul). qL/qR initiaux : R4 111,5/0,1 (J_net ≈ 111,6) ; R3 55,7/−55,7 (≈0).
- **Discrimination :** **spin–spin (∝ L_A·L_B) rejeté** (R4 ≈ 0 attendu, 0,91 observé) ; **somme des carrés (L_A²+L_B²) rejetée** (R4 ≈ 2×R3 attendu) ; **le carré du moment cinétique total (L_A+L_B)² est le plus proche** ; **J_net n'est pas le moteur** (R4 : J_net ≈ 111, même retard que R3 : J_net ≈ 0 ; Q1 et R4 : même qL ≈ 111,5, retards 5,35 et 0,91).

## Réserves
- Écarts de 10–20 % entre un et deux nœuds, toujours dans le même sens (un nœud est un peu moins efficace) ; une dépendance faible en plus du total n'est pas exclue.
- Une seule réalisation par cas.
- **L n'est pas conservé** (L+Lg en z : R3 −95 à t=20 contre −221,5 à t=0 ; R2 −172 contre −332,3) : la corrélation avec L_total ne prouve pas un rôle de la **conservation** du moment cinétique.
- Les séparations à t=20 vont de ±6,0 (ω=0) à ±9,1 (ω=0,25) : le retard correspond à un rapprochement ralenti, avec un repoussement qui croît avec ω.

## Bilan
Le retard de l'annihilation Q/anti-Q dépend du **moment cinétique spatial total** de la paire (et non du produit des spins, ni de la somme de leurs carrés, ni de J_net), croît plus vite que ω², et empêche l'annihilation dans la fenêtre pour ω=0,3 (Lz total ≈ −664 avec deux nœuds). Une paire de moment cinétique total nul annihile normalement. Cela va dans le sens de la conjecture de M. Aldon (rotation opposée requise), au sens du moment cinétique spatial. Le mécanisme (conservation, répulsion centrifuge/effective, autre) reste non isolé.

## À faire
1. Corriger/valider la conservation de L et P (Poynting sur réseau, boîte périodique) ; 2. ω=0,27 / 0,28 et T plus long pour le seuil ; 3. répliques (décalage de grille) pour une barre d'erreur ; 4. test de résolution h ; 5. moment cinétique total fixé et réparti différemment (un nœud à ω fort contre deux) à séparation variée ; 6. moment cinétique emporté par le rayonnement après annihilation (bilan à reconstruire).
