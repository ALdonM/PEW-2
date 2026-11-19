# Note part184x-y : nœuds relaxés dans le référentiel tournant, avec et sans champ de jauge

Session s35, 2026-11-18/19. Deux audits isolés (agents séparés, recalcul depuis les journaux) avant écriture ; leurs corrections sont intégrées. Modèle Z classique, aucune identification à une particule (discipline D). Données : `resultats/part184/gh_part184x/` (relaxations R000/R010/R014 et 12 runs X_*, g²=0,02) et `gh_part184y/` (8 runs Y0_*, g²=0, mêmes nœuds réutilisés par artefact). N=96, séparation 20, T=60.

## 1. Méthode nouvelle

- **Seuil de masse.** Les fluctuations de Z2 ont une masse √2·ν ≈ 0,17. Un nœud qui tourne plus vite ne peut pas être stationnaire. Toutes les séries précédentes (part184o à w) utilisaient ω=0,25, au-dessus du seuil, avec une vitesse imposée et masquée.
- **Relaxation dans le référentiel tournant** (option OMREL de jauge.py) : on minimise V − ω²I (I = h³∑|Z2|², force centrifuge ω²Z2), à g²=0, pour ω=0, 0,10, 0,14, 120 unités. rms_force final 2,2e-5 / 3,3e-5 / 6,4e-5, encore décroissant ; I non convergé à ω=0,14 (+2,2 % sur les 20 dernières unités). J = 2ωI = 374 et 670.
- **Vitesse initiale** (option VPROD) : dérivée exacte du produit de quaternions quand A tourne à s1 et B à s2 (rotation globale de la phase de Z2, sans masque, sans annulation d'impulsion).
- L'énergie est invariante par rotation globale de la phase de Z2 (vérifié dans le code), donc un état relaxé dans le référentiel tournant est une solution stationnaire exacte de la dynamique sans jauge.

## 2. Établi

- **Sans jauge (g²=0), les nœuds en rotation sont quasi stationnaires** : sur 60 unités, Ekin constant à 1 % (18,72→18,59 ; 46,88→46,13), Lz à ±0,6 % (oscillation, pas dérive), charge dans la boule de rayon 8 à 1,5 %. |Z2|² saute de +2 % avant t=10 puis plafonne (ajustement rapide de la queue non convergée).
- **Avec jauge (g²=0,02), ils ne le sont pas** : Ekin 18,7→45 en 10 unités (ω=0,10), Lz −373 → −242 à t=10 (minimum −226), Lz+Lg non conservé non plus. La cause est le champ de jauge, absent de la relaxation ; le mécanisme n'est pas identifié : le champ de jauge démarre aussi à ω=0 sans charge (courants Z2 statiques non équilibrés avec θ=0), donc un simple décalage électrostatique de fréquence ne suffit pas.
- **Un nœud en rotation porte une impulsion axiale intrinsèque** (Pz = −2ω∫|Z2|²∂z arg Z2 ; 20 à 37 par nœud), sans translation correspondante (le nœud seul ne se déplace pas dans le sens de son Pz ; il n'y a pas d'invariance de Lorentz dans ce modèle). L'ancienne annulation de Pz ajoutait donc un boost à un état en rotation.

## 3. Retards à ω=0,14 (t_ann par rapport à ω=0, même orientation, même g²)

| Configuration (charges J à t=0) | g²=0 | g²=0,02 |
|---|---|---|
| côte à côte, spins parallèles (charges opposées) | −0,37 | +6,89 |
| côte à côte, spins opposés (charges égales) | −0,12 | −0,29 |
| sur l'axe, spins parallèles (charges égales) | −2,98 | −2,93 |
| sur l'axe, spins opposés (charges opposées) | +10,05 | +7,77 |
| référence ω=0 (côte à côte / sur l'axe) | 15,07 / 20,91 | 22,88 / 19,90 |

À ω=0,10 et g²=0,02 : +2,19 / −0,25 / −1,26 / +16,09. Les références ω=0 reproduisent exactement les runs antérieurs. Econs conservé à ±0,02 partout. Seuil sur Qabs : retards stables à <0,5.

- **Côte à côte, spins parallèles** : le retard (+6,9) disparaît sans jauge (−0,4). La jauge retarde toutes les paires côte à côte (+7,8 sur la référence, +7,6 sur les spins opposés), et ajoute un surplus de +7,3 propre à cette configuration. Formulation retenue : surplus propre aux charges opposées côte à côte, en plus d'un retard de jauge commun d'environ +7,7. Je retire « couplage jauge×rotation ».
- **Sur l'axe, spins parallèles** : accélération inchangée par la jauge dans ce run (−2,98 contre −2,93).
- **Sur l'axe, spins opposés** : retard présent sans jauge (+10). L'écart avec g²=0,02 (2,3) est dans le bruit (±50 % de sensibilité aux conditions initiales vu en part184u) : je retire « la jauge le modifie ».
- **Lecture par charges ou par spin** : dans chaque orientation, les deux étiquettes sont en bijection, rien ne les départage. À g²=0 la charge n'exerce aucune force à longue portée, donc le motif sans jauge (côte à côte ≈0 ; axe +10 / −3) n'est pas un effet de charge. La lecture par charges est plus compacte pour l'effet de la jauge, mais n'est pas une explication.

## 4. Limites

Une réalisation par point ; profil différent entre ω=0 (R000) et ω=0,14 (R014 : |Z2|² +55 %, 62 % hors du rayon 8), donc « retard dû à la rotation » mélange rotation et profil (il manque une référence non tournante au profil R014) ; recouvrement initial fort (+16 à +22 % de |Z2|² de la paire) ; queue à ω=0,14 décroissant sur ~10 unités contre 28 jusqu'à l'image ; Lz non conservé dans la géométrie côte à côte (−22 % avant annihilation à g²=0) ; relaxation R014 non convergée ; « parallèle » et « opposé » ne désignent pas le même signe des charges selon l'orientation.

## 5. Bilan provisoire du mécanisme (part184j à y)

Deux effets distincts. (1) Côte à côte, un retard qui n'existe qu'avec le champ de jauge, propre aux paires de charges opposées. (2) Sur l'axe, un retard des spins opposés et une accélération des spins parallèles qui existent sans jauge (secteur scalaire). Aucun n'est un dipôle pur ; le second ne vient pas de l'énergie statique à ω fixé ; aucun mécanisme n'est encore isolé.

## 6. Suite

Relaxer avec le champ de jauge inclus (nœud chargé stationnaire à g²>0) ; référence non tournante au même profil ; relaxations plus longues ; quelques réalisations ; potentiel statique à charge fixée avec ces nœuds.
