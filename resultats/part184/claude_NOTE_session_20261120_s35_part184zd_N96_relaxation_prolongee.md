# part184zd (branche N=96) : ω0,19 avec jauge, relaxation prolongée de 100 à 250 unités

Run 44638873798, jobs `relax96b` + 4 paires ZDG96 (succès). Journaux : `gh_part184zd/`. Branche N=128 encore en cours (note séparée à venir). Pas encore audité par un agent séparé : chiffres issus de `analyse_zb.py` et d'un calcul de flux identique à celui de la note zc ; à auditer avec la branche 128.
Étiquettes : Js = J même signe, Jo = J opposé (mesurés par JA·JB).

## La relaxation à ω0,19 n'est pas convergée
Journal `relax_G019b_N96.log` (reprise à 100, +150 unités) : charge totale 406,8 → 447,9 (+10 %), Z2 2768 → 3032, force rms 8,3e-5 → 1,7e-5. La croissance ralentit (≈ +4,7 puis +1,4 par 10 unités) mais n'est pas arrêtée : la charge d'un état « relaxé » dépend encore de la durée. Les charges de paires à ω0,19 ne sont donc pas des valeurs d'équilibre.

## Effet sur les paires (ω0,19 avec jauge ; relaxation 100 → 250)

| config | q0 (100 → 250) | t_ann (100 → 250) |
|---|---|---|
| z Js | 518 → 595 | 16,57 → 16,36 |
| z Jo | 326 → 332 | 18,86 → **15,90** |
| x Js | 498 → 568 | 22,63 → 22,76 |
| x Jo | 292 → 300 | 25,27 → 24,81 |

- t_ann bouge peu (≤0,5) en x et en z Js. **z Jo avec jauge descend à 15,90, soit encore plus loin de la tendance sans jauge** (22,7 → 26,3 → 31,0 pour ω 0,077 / 0,10 / 0,14). Avec jauge, z Jo vaut maintenant 21,8 (ω0,10) → 25,8 (ω0,14) → 15,9 (ω0,19) : la non-monotonie n'est pas due à une relaxation trop courte (au moins jusqu'à 250).
- Le flux de charge au plan médian jusqu'à t_ann (Imid, trapèzes sur `CUR`) : Jo −1,10·q0 (x) et −0,87·q0 (z) ; Js ≈ 0 (+0,3 et +1,7). Le flux Jo reste de signe et d'ordre de grandeur comme dans zc, mais en z Jo il est inférieur à q0 (−0,87), contrairement à zc (−1,7 à −2,3·q0) : à ω0,19, t_ann tombe avant que q0 ait fini de traverser.
- Part du flux au bord périodique (Iwrap/Imid) : 12 % (x Jo), 4 % (z Jo).
- Charge à r>22 (qfar) : 23 % (x Js) et 20 % (z Js) de la charge totale à t=0 ; nulle pour les Jo (|qtot|<5, donc la fraction n'a pas de sens). Les paires Js à ω0,19 mettent donc une part importante de leur charge près du bord de boîte N=96 : à lever avec N=128.
- |L| à t=0 : x Jo 1136, x Js 29, z Js 1244, z Jo 1 (même structure J/L qu'avant).

## Limites
Une réalisation par point ; relaxation non convergée (voir ci-dessus) ; q0 diffère d'une paire à l'autre (z Js et x Js ont ~2× la charge des Jo) ; réseau 96 seul. Question ouverte : est-ce la boîte ou une convergence plus lente encore qui contrôle ω0,19 ? Réponse attendue de la branche 128.
