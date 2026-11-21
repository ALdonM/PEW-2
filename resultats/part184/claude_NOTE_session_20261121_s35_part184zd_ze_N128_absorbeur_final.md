# part184zd (N=128) et part184ze (absorbeur) : résultats et audit isolé (21/11/2026)

Statut : [D] numérique, modèle d'essai Z, aucune identification physique.

## 1. Effet de boîte sans jauge, omega 0,14 (t_ann)
Références N=96 : gh_part184y/Y0_*_om014 (pas zb ; ZBQ/ZCQ sont AVEC jauge). Nommage inversé selon l'axe : x_par = Jo, x_anti = Js ; z_par = Js, z_anti = Jo.
| Paire | N=96 (Y0) | N=128 (ZD0128) | écart |
|---|---|---|---|
| x Jo | 14,69 | 14,17 | -0,52 |
| x Js | 14,95 | 14,45 | -0,50 |
| z Js | 17,94 | 17,40 | -0,54 |
| z Jo | 30,96 | 30,37 | -0,59 |
Raccourcissement systématique de 0,5 à 0,6 (3 %). Charges q0 non identiques (N96 -> N128 : x Jo +4,6 %, x Js -3,5 %, z Jo +3 %, z Js 0 %).

## 2. Avec jauge, omega 0,19
| Paire | N=96 | N=128 | q0 N96 / N128 |
|---|---|---|---|
| x Jo | 24,81 | 24,57 | 300 / 303 |
| x Js | 22,76 | 21,49 | 568 / 432 |
| z Jo | 15,90 | 20,25 | 332 / 336 |
| z Js | 16,36 | 16,36 | 595 / 467 |
A charge égale seulement pour les Jo ; pour les Js la charge initiale diffère de 21 à 24 %.
z Jo : forte dépendance en N (+27 %) et en durée de relaxation (ZBG 18,86 contre ZDG96 15,90 au même N). [CORRIGÉ dans la note ze2 : l'audit avait mélangé les séries sans jauge (26,31 et 30,37) et avec jauge (20,25). Voir claude_NOTE_session_20261121_s35_part184ze2_N128_omega010_jauge_zJo.md.] On ne peut PAS conclure à un artefact de boîte.

## 3. Absorbeur (jauge.py : SPM, SPG, SPRS, SPROT), paire x Jo, omega 0,10, sans jauge
Référence cube périodique ZC0_xJo_om10 (= ZB0_x_par_om10) : t_ann 14,95, qL(10) = 21,94.
- ZE0 (amortit toute la vitesse) : 14,91 (sp18), 14,88 (sp16) ; qL(10) = 1,5 / -9,2 ; qtot(10) = 0,5 : le spin des nœuds est freiné.
- ZE1 (préserve la rotation de phase de Z2) : 14,95 / 14,96 ; qL(10) = 18,4 / 18,3 ; qtot = 0.
- Énergie : perte avant t=15 de 0,5 à 1,2 % (Econs), 3 % (Etot). Restant à t=60 : ZE1 25 à 27 % (Econs) ; ZE0 8 % (sp16) et 12 % (sp18).
Portée : une paire, un omega, sans jauge, un N, un axe. « t_ann indépendant du bouclage » n'est établi que dans ce cas. Que qL(10) bouge de 22 à 1,5 sans changer t_ann suggère que t_ann est fixé par l'approche initiale.
Non testé : jauge avec absorbeur (loi de Gauss), paires z, Js. Le champ rayonné EM revient plus (20 % contre 8 % pour la matière dans le test linéaire) : à absorber aussi.

## Audit isolé
Agent séparé : tous les chiffres recalculés CONFIRMÉS ; journaux complets (61 lignes), Qabs0 = 1,94 à 1,95, qL+qR = 0 pour les Jo sans jauge (< 0,4 % avec jauge). Corrections reprises ci-dessus.
