# NOTE — session 35 (13/11/2026) — part184e : paire Q opposées, axe x vs diagonale, N=112, T=45

**Statut : AUDITÉ le 13/11 par un agent isolé** (recalcul depuis les journaux et les `*_proj.npy`, lecture de `jauge.py`). Rien n'est publié ; rien n'est identifié à une particule (option (D)).
Run GitHub 43477509324 ; données `resultats/part184/gh_part184e_T45/` ; code `jauge.py` (ORIENT, SEPD ; champs `sect16_z0`, `polar8`, `azi12_eq`) ; workflow `pew-part184e-diag-T45.yml`. Complète la note part184d (T=30).

## Cas
K1/K3 : paire sur x, J opposées / sans J. H1/H3 : paire sur la diagonale (1,1,0), SEPD=14 (séparation 19,8). N=112 (boîte ±28), h=0,5, T=45, ν=0,12, g²=0,005, ω=0,08 ou 0.

## Résultats
- **(A) Boîte :** K1/K3 reproduisent G1/G3 (N=96) : Qabs −3 % à t=18, −6 % à t=20–22, −3 % à t=24 ; au plancher (Qabs≈0,07) écarts ±5–9 % = bruit. Seule la taille de boîte est testée (h fixe) : la résolution ne l'est pas.
- **(B) Temps de passage Qabs<0,5 :** K3 19,7 ; K1 19,9 ; H3 24,3 ; H1 29,4. Effet de J : +0,2 sur x, +5 en diagonale. **Non interprétable comme anisotropie du réseau** : dans `embed`, B est toujours réfléchi x→−x, y compris en mode diag, donc B n'est pas l'image miroir de A par rapport à la paire. L'écart vaut une rotation de 90° autour de z, équivalente (hopfion p2,p3 ∝ x,y) à une rotation interne de π/2 dans le plan (2,3), générateur de J : la phase relative A/B diffère entre K et H (produit de quaternions non commutatif). Avec J opposées la phase relative dérive de 2·ω·t ≈ 3,2 rad à t=20. Autres écarts : séparation 19,8 vs 20 ; Etot initial +21 en H. **Conclusion : l'effet de J mesuré n'est pas établi comme intrinsèque ; il dépend de la phase relative initiale.**
- **(C) Direction des maxima de sect16 :** K → ±y (secteurs 3-4, 11-12) ; H → perpendiculaire à la diagonale (secteurs 5-6, 13-14). L'orientation suit l'axe de la paire, pas le réseau. Réserves : résolution 22,5° ; pour H1 prendre t≥33 ; le pic est celui d'un anneau équatorial projeté sur z, **pas une preuve de deux lobes localisés**.
- **(D) Forme :** énergie de couronne maximale à l'équateur de la paire (×3 à ×9 vs pôles). Azimut autour de l'axe non plat : modulation jusqu'à +43/−45 % (K t=39), +37/−34 % (K t=45), +35/−24 % (H1 t=36), période 180°, identique K1/K3, alternant ±y/±z. Image : **anneau équatorial avec quadrupôle**, pas un disque parfait ni deux lobes. L'axe z est privilégié par construction (hopfion axisymétrique autour de z, générateur de J = rotation autour de z ; excès ±z ×1,34–1,47 déjà à t=0).
- **(E) Propagation :** pic de couronne ≈ r 9 pendant 8–10 unités après l'annihilation, puis vitesse 0,8–1,07, compatible avec c=1 ; front à mi-hauteur 0,90–0,97 ; fit de H1 sur 8 points seulement. L'origine extrapolée coïncide avec Qabs≈0,5.

## Verdict sur le test de la direction
La direction du rayonnement **n'est pas fixée par le réseau** (elle tourne avec la paire). Elle reste **imposée par la géométrie de la paire** (perpendiculaire à son axe) et par l'axe z du nœud ; elle n'est pas spontanée. Le critère « annihilation → deux lobes opposés dont la direction n'est pas imposée » n'est **pas confirmé** : on observe un anneau équatorial à quadrupôle.

## À faire
1. Contrôle H avec **miroir le long de l'axe de la paire** (reproduire exactement la phase relative de K) ; 2. varier la phase relative initiale A/B en x pour tester l'effet de J ; 3. test de résolution (h) ; 4. orientation de l'axe du nœud (z) différente de la normale de la paire ; 5. énergie conservée complète ; 6. répliques.
