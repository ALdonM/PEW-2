# NOTE — session 35 (13/11/2026) — part184d : lobes de rayonnement, paires Q opposées, séparation 20, T=30

**Statut : AUDITÉ le 13/11 par un agent isolé** (recalcul depuis les journaux G1/G3, lecture de `jauge.py`). Rien n'est publié ; rien n'est identifié à une particule (option (D)).
Données : `resultats/part184/gh_sep20_T30_lobes/{G1_Qopp_Jopp_T30,G3_Qopp_sansJ_T30}.log` et `*_proj.npy` (run GitHub 43432102200, commit 92f71d5/3779d24).

## Verdict
Le test attendu (« excédent d'énergie en deux lobes opposés égaux, direction non imposée, dépendance en J ») est **NON établi**. Seul tient : deux zones ±y d'énergies égales à ~3 % près, qui émergent après t≈20. Ce qu'on voit à T=30 est un front d'énergie qui entre dans la couronne, pas un lobe en régime établi.

## Résultats (chiffres audités)
- **Repérage.** `sect=int((ang+π)/2π·16)` : le secteur 0 commence à −180°. Secteurs 3-4 = −y, 11-12 = +y, 7-8 = +x, 15-0 = −x. (Erreur de repérage corrigée : « 0 et 8 = ±x » est faux.)
- **t<18 : couronne dominée par les queues des nœuds** (x=±10). À t=0, énergie ±x 222 contre 1,3 sur ±y. Minimum d'anisotropie à t≈20 (|c2| min ≈ 1,6 ; y⁻=26,3, x⁺=30,7).
- **t>20 : montée de ±y** : y⁻+y⁺ = 41 (t=20), 80 (24), 140 (26), 203 (28), 240 (30). Le total de la couronne passe de 217 (t≈22) à 420 (t=30) : montée générale, diagonales comprises (104→156), pas une structure localisée.
- **Égalité des lobes** : y⁻/y⁺ = +2,8 % (G1), +2,7 % (G3) à t=30 ; biais systématique y⁻>y⁺ (2–11 % selon t), déjà présent à t=0 : artefact de réseau (grille décalée d'une demi-maille).
- **Dépendance en J** : G1 (J opp.) vs G3 (sans J) : écart de lobe 5,6 %, total couronne 421 vs 437 (4 %). Formes et vitesses semblables. « Indépendant de J » est trop fort : pas de réplique, donc pas de barre d'erreur ; on peut seulement dire « J change ~5 %, même forme ».
- **Vitesse** : pic radial de l'énergie projetée (secteur ±y) de r≈2,3 (t=20) à 9,3–10,3 (t=30) ; pente 0,63 (G1) / 0,71 (G3), **< c** ; le pic est encore sous le bord interne de la couronne (r=12). Le front arrive encore.
- **Direction non démontrée spontanée** : ±y est à la fois perpendiculaire à l'axe de la paire et axe du réseau cubique ; de plus la projection est sommée sur z, un disque équatorial axisymétrique autour de x donnerait aussi deux lobes ±y.
- **Réserves** : e2 non défini positif (valeurs ≈ −0,4 pour r>14 à t=0) ; couronne 12<r<22 proche du bord périodique (r=24) ; runs T24 sans sect16.

## À faire pour trancher
1. T ≥ 45 et boîte plus grande (ou N plus fin) ; 2. paire orientée sur la diagonale (x=y) ; 3. coupe z=0 au lieu de la somme sur z, profil en angle azimutal autour de l'axe de la paire ; 4. répliques (décalage de grille/résolution) pour une barre d'erreur ; 5. énergie conservée complète (masse et jauge incluses) ; 6. comparer h=1/3 et h=0,5 (part182/part183, non fait).
