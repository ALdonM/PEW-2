# part184zc : courant de charge au plan médian (rejeu des paires zb, N=96)

Run 44638869727 (6 paires, succès). Journaux : `gh_part184zc/`. Code : `jauge.py` avec `QCUR=1` (commit b2b250b), vérificateur `/home/claude/tools/chk_cur.py` (non versionné). Audit isolé par un agent séparé : tout confirmé sauf deux points sur le profil radial (corrigés ci-dessous).

## Diagnostic
Courant de liaison exact : q̇(x) = (1/2h³) Σ_a [gt_a(x−a) − gt_a(x)], gt = ∂E/∂θ. Flux de charge vers la droite à travers le plan S=0 : Imid ; liaison périodique de bord de boîte : Iwrap ; dqL/dt = Imid + Iwrap. Lignes `CUR` toutes les 0,1 unité.
Vérifications : continuité ponctuelle ≤ 3e-6 (pour |q̇| de 0,1 à 0,9 en valeur max) ; delta qL = ∫(Imid+Iwrap) à moins de 0,011 sur toutes les fenêtres [1,5] à [20,30] pour les 6 paires. t_ann identiques à zb (0,00 d'écart, calcul déterministe : ce n'est pas un test indépendant).
Étiquettes : Js = J de même signe, Jo = J opposé, mesurés par JA·JB, pour x comme pour z.

## Résultats (flux intégré de 0 à t_ann)

| paire | q0 | t_ann | ∫Imid | /q0 | instant où 0,5·q0 a traversé | |Imid| max à |
|---|---|---|---|---|---|---|
| ZC0 xJo ω0,10 | 168 | 14,95 | −231 | −1,37 | 7,8 | 16,0 |
| ZC0 zJo ω0,10 | 181 | 26,31 | −419 | −2,32 | 13,9 | 24,4 |
| ZCQ xJo ω0,14 | 172 | 24,32 | −222 | −1,29 | 9,0 | 24,8 |
| ZCQ zJo ω0,14 | 185 | 25,76 | −308 | −1,67 | 13,3 | 23,7 |
| ZC0 zJs ω0,10 | 207 | 19,38 | +4,5 | +0,02 | — | — |
| ZCQ zJs ω0,14 | 223 | 18,10 | +1,0 | 0,00 | — | — |

1. **Seules les paires J opposé échangent de la charge à travers le plan.** Les contrôles J même signe n'en échangent pas (≤2 % de q0).
2. **Le flux dépasse q0 de 30 à 130 % avant t_ann** : qL ne s'arrête pas à 0, il change de signe et continue. La charge ne s'annule pas simplement au contact : les deux nœuds se la transfèrent en traversant le plan, au moins jusqu'à t_ann.
3. **Le flux est maximal près de t_ann** (à 0,5 à 2,1 unité de t_ann), alors que la moitié de q0 a déjà traversé à ~0,4 à 0,55 t_ann. Le passage de qL par 0 précède donc t_ann de 2 à 7 unités (note zb) mais le courant ne s'arrête pas : t_ann n'est pas l'instant où la charge « a fini de passer ».
4. **Où passe le flux (profil radial, `jrad`)** : sans jauge, 95 à 97 % du flux traverse à r<12 (bins de 2, concentrés à r de 2 à 8). **Avec jauge, 34 à 39 % du flux passe à r≥12, mais ce chiffre est peu fiable** : `jrad` n'est écrit qu'aux unités de temps et le flux oscille plus vite ; intégré au pas de 1, il donne −154 (xJo) et −249 (zJo) au lieu de ∫Imid = −222 et −308, soit 31 % et 19 % d'écart (contre ≤7 % sans jauge). Les fractions (−0,37 et −0,33) que j'avais d'abord lues comme des sommes sont des parts, pas des flux. À refaire avec `jrad` au pas de 0,1.
5. **Bord de boîte** : la part du flux qui passe par la liaison périodique (Iwrap) vaut 11 % (xJo sans jauge), 8 % (xJo jauge), 1,7 à 1,8 % (zJo). En x, la charge atteint donc le bord de boîte : cela rejoint la réserve sur qfar à ω0,19. Le run zd (N=128) doit trancher.

## Limites
Une réalisation par point ; q0 non identique d'une paire à l'autre (lire les ratios, pas les valeurs absolues) ; interprétation du flux à grand rayon avec jauge non tranchée (courant de matière ou rééquilibrage du champ électrique ?) ; la définition de qL (moitié S<0) suppose le plan médian comme seule frontière.
Les champs `Imid`, `Iwrap`, `cont`, `jrad` et la ligne `CUR` ne sont pas inscrits au dictionnaire des variables : à décider.

## En cours
Run zd 44638873798 : ω0,19 avec jauge, relaxation prolongée (N=96, +150) et boîte 128 (relaxations puis 12 paires). Étiquettes Js/Jo. Plusieurs heures (jusqu'à ~12 h).
