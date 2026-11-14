# NOTE — session 35 (14/11/2026) — part184f : diagonale avec miroir le long de l'axe de la paire

**Statut : AUDITÉ le 14/11 par un agent isolé** (recalcul depuis les journaux, test du code du miroir). Rien n'est publié ; rien n'est identifié à une particule (option (D)). Corrige la note part184e (§B).
Run GitHub 43508451147 ; données `resultats/part184/gh_part184f_T45/` (M1 J opposées, M2 J identiques, M3 sans J ; ω=0,08 ou 0) ; code `jauge.py` option `MIRROR=axis`.

## Construction
B = image de A par la réflexion (x,y)→(−y,−x) (plan perpendiculaire à l'axe de la paire), au lieu de x→−x. Vérifié : le code réalise bien cette réflexion ; Q=0 à t=0 ; Qabs initial 1,954 (K : 1,955) ; xL/xR = −9,96/+9,88 ; qL/qR cohérents.

## Résultats
- **Temps de passage Qabs<0,5 :** M1 19,48 ; M2 19,25 ; M3 19,22 ; K1 19,91 ; K3 19,69 ; (anciens, réflexion x) H3 24,25 ; H1 29,37. **Le retard de la diagonale disparaît** avec le bon miroir. Il venait de la construction de B (phase relative A/B), pas du réseau ni de J. (Inférence : le miroir est le seul paramètre changé, mais aucun run ne change B seul.)
- **Effet de J opposées résiduel :** +0,26 (M1−M3), +0,23 (K1−K3), soit ~1,2 %, de même signe dans les deux orientations ; constant à ±0,03 sur les seuils 1,0/0,5/0,2 : **simple retard temporel**. J identiques : +0,03 (nul). Il explique le déficit de couronne de M1 à t=30 (−2,9 % ; K : −3,3 %), disparu à t=36 (+0,2–0,4 %).
- **Rayonnement :** énergie de couronne à t=36 et 45 identique à ≤0,24 % entre M1, M2, M3. Par secteur, M1 s'écarte de 21–25 % sur les petits secteurs à t=30–36 (retard), M2/M3 ≤3,6 %. Donc pas de dépendance en J de la dynamique intégrée ; les secteurs à t=30–36 gardent la trace du retard.
- **Direction :** maxima de sect16 dans les secteurs 5/6/14, perpendiculaires à la diagonale, mais alternant à quelques % près : ne pas sur-interpréter.
- **Forme :** à t=45, polar8 et azi12_eq (anneau équatorial à quadrupôle) comme dans K à ~3 % ; à t=30–36 les profils M et K diffèrent légèrement.
- **M plus rapide que K de ~0,4 (≈2 %)** : séparation 19,84 vs 19,99 (−0,75 %), Etot initial +2, anisotropie du réseau possibles ; non démontré (exigerait t ∝ séparation^≈3).

## Bilan sur la question de M. Aldon (annihilation en deux « photons-ressort » et rotation ω opposée)
Le sens de J (opposé / identique / absent) **ne change ni l'issue ni le rayonnement intégré** de l'annihilation Q/anti-Q ; il décale seulement le temps d'annihilation de ~1 % pour J opposées. Le critère « ω opposée requise » n'est **pas confirmé** dans ce régime (ω=0,08, g²=0,005, J interne, charge non liée au nœud). Pas de deux lobes localisés : anneau équatorial à quadrupôle, direction fixée par la géométrie de la paire.

## À faire
Séparation variée (test de t ∝ sép^n) ; ω et g² plus grands ; moment cinétique spatial ; résolution h ; énergie conservée complète ; répliques.
