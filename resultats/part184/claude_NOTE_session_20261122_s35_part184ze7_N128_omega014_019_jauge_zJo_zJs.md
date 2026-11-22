# part184ze7 : N=128, avec jauge, omega 0,14 et 0,19 (relaxation prolongée), paires z (22/11/2026)

Statut : [D] numérique, modèle d'essai Z. Run GitHub 44959317019 (chaîne A : omega 0,14, relaxation 4 x 25 unités depuis le noeud N=128 sans jauge ; chaîne B : omega 0,19, +150 unités en 6 étapes après les 100 unités de zd ; puis paires z). Journaux : gh_part184ze/ZDG128_z*_om14.log et *_om19_long.log.

## t_ann (avec jauge)
| Paire | omega | N=96 | N=128 | q0 N96 / N128 |
|---|---|---|---|---|
| z Jo | 0,10 | 21,78 | 20,77 | 113,5 / 113,0 |
| z Jo | 0,14 | 25,76 | 23,93 | 185,0 / 181,8 |
| z Jo | 0,19 (100 unités) | 18,86 (ZBG) | 20,25 | 326 / 336 |
| z Jo | 0,19 (250 unités) | 15,90 | 16,03 | 332 / 350 |
| z Js | 0,10 | 19,15 | 18,66 | 126,0 / 117,8 |
| z Js | 0,14 | 18,10 | 17,79 | 223,2 / 199,8 |
| z Js | 0,19 (100 unités) | 16,57 (ZBG) | 16,36 | 518 / 467 |
| z Js | 0,19 (250 unités) | 16,36 | 16,12 | 595 / 641 |

## Lecture
- z Jo avec jauge : la non-monotonie en omega (maximum à 0,14 puis creux à 0,19) existe aux deux tailles de boîte : N=96 : 21,8 / 25,8 / 15,9 ; N=128 : 20,8 / 23,9 / 16,0. Ce n'est pas un artefact de boîte (écart N96 -> N128 de 0,1 à 1,8).
- Le creux à omega 0,19 dépend de la durée de relaxation : t_ann passe de 20,25 (100 unités) à 16,03 (250 unités) à N=128, et de 18,86 à 15,90 à N=96.
- Charge : elle monte encore avec la durée de relaxation (z Jo 336 -> 350 ; z Js 467 -> 641 à N=128), donc la relaxation à omega 0,19 n'est pas convergée à 250 unités. z Js n'est pas à charge égale entre N=96 et N=128 (-10 % à omega 0,14 ; +8 % à 0,19 après 250 unités).
- z Js reste toujours plus rapide que z Jo (16,1 contre 16,0 à omega 0,19 : égalité à la précision des données ; ailleurs 1 à 6 unités de moins).

## Limites
Une orientation (z), un seul N supplémentaire, relaxation à omega 0,19 non convergée, charges initiales différentes pour les Js. Le creux à omega 0,19 n'est qu'à moitié établi : les deux relaxations N=96 et N=128 de 250 unités le donnent à 16,0 mais la charge n'a pas cessé de croître.
