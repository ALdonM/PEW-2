# part177 — refroidissement d'un chaos chaud : des nœuds naissent-ils d'eux-mêmes ? (aucun nœud injecté)

Idée (M. Aldon, 06/11) : suivre le déroulé cosmologique — chaos chaud sans matière stable, refroidissement, puis seuil où des nœuds deviennent viables.
Départ : chaos chaud (ξ0 = 2, RMS ϱ² = 2), 64³, h = 0,5, e0 = 0,25 (nœud R ≈ 10 mailles), μ = 0,375, sans barrière.
Refroidissement : friction constante G0 = 0,02 / 0,1 / 0,5 (vitesse de refroidissement, à la Kibble–Zurek) ou expansion (MODE rad). t = 10 → 60.
Mesures : énergie, ordre, degré total, degré par blocs, comptage nœuds/anti-nœuds (préimages de l'antipode du vide moyen, par région retournée).

Critères écrits avant les résultats :
1. Le comptage n'a de sens que si order ≳ 0,85 ; avant, seul le degré total est rapporté.
2. Un nœud « né » doit : être résolu (deg_bloc cohérent), garder sa charge ≥ 10 unités de temps, occuper ≥ ~10 mailles.
3. Nœuds et anti-nœuds doivent naître en nombre égal en moyenne (pas de biais dans l'équation) ; un excès systématique serait un artefact.
4. Résultat négatif possible et informatif : aucun nœud ne survit au refroidissement pour aucune vitesse.
