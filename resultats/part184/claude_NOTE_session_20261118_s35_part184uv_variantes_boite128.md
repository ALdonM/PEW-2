# Note part184u-v : variantes de conditions initiales (N=96) et boîte N=128

Session s35, 2026-11-17/18. Audit isolé (agent séparé, recalcul depuis les journaux) avant écriture ; ses corrections sont intégrées. Modèle Z classique, aucune identification à une particule (discipline D). Données : `resultats/part184/gh_part184u_a/` (A*, N=96) et `gh_part184v_b/` (B*, N=128). Le volet N=128 de part184u a échoué (relaxation au-delà de la limite de 6 h des jobs GitHub ; dossier gh_part184u_b jamais créé) ; il a été refait en part184v avec une relaxation en deux étapes de 60 unités (option RELIN ajoutée à jauge.py ; le redémarrage remet v=0, donc 2×60 n'équivaut pas à 120 unités continues). Suite des notes part184jo à t.

## 1. Variantes sur l'axe, spins antiparallèles, Pz annulé (N=96)

| Run | g² | séparation (t=0) | ω | t_ann |
|---|---|---|---|---|
| A2 | 0,02 | 19,06 | 0,25 | 26,16 |
| S_z_opp_cancel (réf.) | 0,02 | 20,05 | 0,25 | 40,32 |
| A3 | 0,02 | 21,03 | 0,25 | 41,79 |
| A4 | 0,02 | 20,05 | 0,24 | 42,55 |
| A5 | 0,02 | 20,05 | 0,26 | 38,54 |
| A6 | 0,01 | 19,06 | 0,25 | 22,96 |
| A1 | 0,01 | 20,05 | 0,25 | 54,63 |
| A7 | 0,01 | 21,03 | 0,25 | 50,50 |
| A8 | 0,015 | 20,05 | 0,25 | 45,53 |

Tous confirmés ; Econs constant à 1e-5 ; Pz initial nul. A1 reproduit exactement T_z_opp_g001_cancel jusqu'à t=45 (le nœud N=96 se reproduit à l'identique). À g²=0,01 la paire annihile bien, à t=54,6 (non annihilée à T=45 en part184t).

- **t_ann est mal conditionné** : ±1 unité de séparation (±5 %) donne 26 / 40 / 42 (g²=0,02) et 23 / 55 / 51 (g²=0,01), non monotone ; l'ordre en g² s'inverse à séparation 19. À séparation 19 la paire est capturée dès la première approche, sans rebond ; à 20 et 21, rebond puis capture. La sensibilité vient de la barrière, pas du seuil sur Qabs (retards stables à <0,2 selon le seuil).
- **ω** : 0,24 / 0,25 / 0,26 → 42,6 / 40,3 / 38,5 : dépendance lisse et décroissante (plus de rotation, moins de retard, dans cette plage étroite).
- Pas de référence ω=0 à séparation 19/21 ni à g²=0,015 : aucun retard n'est calculable pour A2, A3, A6, A7, A8. Le « retard antiparallèle sur l'axe » n'est défini qu'à ±50 % près ; je retire tout chiffre unique pour lui.

## 2. Boîte N=128 (distance à l'image 44 au lieu de 28)

| Configuration (Pz annulé) | retard N=96 | retard N=128 |
|---|---|---|
| axe, parallèles (g²=0,02) | +6,32 | +6,08 |
| côte à côte, parallèles | +6,85 | +6,73 |
| côte à côte, antiparallèles | +0,34 | +0,53 |
| axe, antiparallèles (g²=0,02) | +20,42 | +17,28 |
| axe, antiparallèles (g²=0,01) | +33,81 | +32,29 |

Références ω=0 : z 19,90→19,37, x 22,88→22,24, z (g²=0,01) 20,82→20,28 (−3 %). Econs conservé à 1e-5.

- Retards parallèles stables (−3,8 % en z, −1,8 % en x). Antiparallèles côte à côte : nuls dans les deux cas (l'écart de 0,19 porte sur un effet nul).
- Antiparallèles sur l'axe : −15 % à g²=0,02, −4,5 % à g²=0,01. Je retire l'argument « dans la dispersion des variantes » comme preuve de robustesse : la dispersion de séparation ±1 est une perturbation plus forte qu'un changement de boîte, et B2 (36,65) sort de la plage des variantes en ω (38,5 à 42,6).
- **Boîte et nœud ne sont pas séparables** : le nœud remplit la boîte (NB=N). Les cœurs sont quasi identiques (énergie dans la boule de rayon 8 : 767,5 contre 766,1), l'écart d'énergie vient de la queue au-delà de r=22, que la boîte contrôle. Formulation retenue : **les retards survivent au passage de 28 à 44, même signe et même ordre ; boîte et queue du nœud ont changé ensemble**. Je retire « images périodiques exclues ».
- Convergence de la relaxation N=128 : **non établie** (aucun journal de relaxation archivé, ni pour N=128 ni pour N=96 à 120 unités).

## 3. Rebond sur l'axe (antiparallèles)

| Run | premier minimum de xR−xL | maximum suivant | rebond |
|---|---|---|---|
| A1 (g²=0,01, sép. 20) | 17,69 (t=18) | 19,27 (t=37) | +1,58 |
| A3 (g²=0,02, sép. 21) | 19,65 (t=19) | 19,92 (t=25) | +0,27 |
| A7 (g²=0,01, sép. 21) | 19,61 (t=18) | 20,17 (t=28) | +0,56 |
| S_z_opp_cancel (N=96) | 17,87 (t=17) | 18,79 (t=27) | +0,92 |
| B2 (N=128) | 17,48 (t=17) | 18,08 (t=24) | +0,60 |
| B8 (N=128, g²=0,01) | 17,25 (t=18) | 18,72 (t=37) | +1,47 |

Le rebond existe à N=128, plus faible (−35 % à g²=0,02, −7 % à g²=0,01). Il n'apparaît dans aucun run ω=0 ni dans aucun run à spins parallèles. Ce n'est pas une barrière à distance fixe : le point de retour dépend de la séparation initiale (19,6 à séparation 21 ; 17,7 à 20 ; aucun à 19).

## 4. Limites

Une seule réalisation par point, alors que t_ann est très sensible aux conditions initiales. Dans les runs sur l'axe antiparallèles, le centre de masse dérive de −1 à −1,6 le long de z (pareil à N=96, N=128 et g²=0). Pz initial nul mais Pz matière oscille ensuite (±13), en échange avec le champ de jauge. B7 et B8 à T=60 (B8 annihile à 52,6, marge faible).

## 5. Bilan

Établi : le retard des spins parallèles (~+6 à +7 en z et x, Pz annulé, g²=0,02) est robuste à la boîte. Les antiparallèles côte à côte ne retardent pas. Les antiparallèles sur l'axe passent par un rebond puis une capture, avec un retard grand mais mal défini (dépend fortement de la séparation initiale). Mécanisme toujours non isolé. Suite utile : plonger le nœud N=96 dans N=128 (NB=96) pour séparer boîte et nœud ; archiver les journaux de relaxation ; utiliser une métrique indépendante du seuil (énergie d'interaction en fonction de la distance, ou potentiel effectif extrait de xR−xL(t)).
