# README part188b — « iceberg dans la houle », houle courte (régime d'optique géométrique) — STATUT : lancé le 19/11/2026

Suite de part188 (k = 1, λ ≈ 6,3, comparable à la taille du nœud ≈ 5 : test non concluant, diffraction + non-linéarité).
Ici : k = 3 (λ ≈ 2,1 ; vitesse de groupe réseau ≈ 0,73 pour la phase, ≈ 0,97 pour l'amplitude ω = √(k²+4μ²)), σ = 4, ε = 0,05 (et 0,025 pour la linéarité), départ x = −10, t = 22. Après la traversée (t ≈ 14), le paquet a parcouru ≈ 6 : échelle de Fresnel √(λL) ≈ 3,5, nettement sous la taille du nœud. K0 (nœud seul) réutilisé de part188 (même code, même durée).
Runs : Kp3, Vp3 (phase, ε 0,05), Kp3b, Vp3b (phase, ε 0,025), Ka3, Va3 (amplitude, ε 0,05).
Mesure : Δx(b) = centre de |S|² (nœud) − (vide) par colonne transverse, avec S_K = φ(K+houle) − φ(K0), S_V = φ(V+houle) − vide ; analyse retenue pour 6 ≤ b ≤ 13 : au-delà de la coquille diffusée (rayon ≈ 6) et avant les images périodiques (b > 14).
Critères écrits avant : (1) linéarité : Δx(b) à ε = 0,05 et 0,025 compatibles à ≲ 20 % ; (2) phase : si Δx(b ≥ 6) ≈ 0 dans le bruit (écart-type par bin) → pas de retard de type gravité, la queue de part188 était de la diffraction ; si Δx ≠ 0 et robuste à ε : ajuster b⁻ⁿ et exponentielle sur 6–13, comparer à l'amplitude ; (3) amplitude : longueur de décroissance vs ≈ 2 (part188) ; (4) un décalage < 3 × l'écart-type du bin n'est pas rapporté.
Limites : une boîte L = 32 (images), un nœud, un h (k = 3 → ≈ 4 mailles par longueur d'onde : dispersion de réseau, annulée en partie par la comparaison K − V), centre d'énergie ≠ retard de phase pur.
