# part184ze2 : N=128, omega 0,10, avec jauge, paires z (21/11/2026)

Statut : [D] numérique, modèle d'essai Z.

## Résultats (ZDG128_zJo_om10, ZDG128_zJs_om10 ; relaxation en 4 étapes de 25 unités depuis le noeud N=128 sans jauge ; 100 unités au total)
| Paire (z, avec jauge, omega 0,10) | N=96 (ZA_z_*_om10) | N=128 | écart | q0 N96 / N128 |
|---|---|---|---|---|
| z Jo | 21,78 | 20,77 | -1,01 | 113,5 / 113,0 |
| z Js | 19,15 | 18,66 | -0,49 | 126,0 / 117,8 |
Avec jauge à omega 0,10, la boîte 128 raccourcit t_ann de 0,5 à 1,0, comme sans jauge (0,5 à 0,6). Charge égale pour z Jo (0,4 % d'écart), pas pour z Js (-6,5 %).

## Série z Jo avec jauge en omega
- N=96 : 21,78 (0,10) ; 25,76 (0,14) ; 18,86 puis 15,90 (0,19, relaxation de 100 puis 250 unités).
- N=128 : 20,77 (0,10) ; pas de point à 0,14 ; 20,25 (0,19).
Le maximum à 0,14 reste non vérifié à N=128. À 0,19 l'écart N96 -> N128 est de +4,4 (et de +1,4 contre la première relaxation) alors qu'à 0,10 il est de -1,0 : le point 0,19 est le plus sensible à la boîte et à la durée de relaxation. Avec seulement deux points à N=128, on ne peut conclure ni à la non-monotonie ni à son absence.

## Correction d'une erreur de mon audit précédent
La note zd/ze disait que la non-monotonie de z Jo persiste à N=128 avec 26,31 / 30,37 / 20,25. Ces valeurs mélangent deux séries : 26,31 (ZB0, N96) et 30,37 (ZD0128) sont SANS jauge, 20,25 est AVEC jauge. Série sans jauge z Jo : 26,31 (0,10, N96), 30,96 (0,14, N96), 30,37 (0,14, N128) : croissante. Corrigé.

## Suivi
Pour trancher : une paire z Jo avec jauge à omega 0,14 en N=128, et omega 0,19 en N=128 avec une relaxation plus longue.
