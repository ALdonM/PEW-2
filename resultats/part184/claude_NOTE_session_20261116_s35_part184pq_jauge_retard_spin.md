# Note part184p-q : rôle du champ de jauge dans le retard d'annihilation lié au spin

Session s35, 2026-11-16. Audit isolé (agent séparé, recalcul depuis les journaux) avant écriture ; ses corrections sont intégrées. Modèle Z classique, aucune identification à une particule (discipline D). Données : `resultats/part184/gh_part184p/` (Y_*) et `gh_part184q/` (Z_*). Nœud relaxé en N=96 (RELT=120), NB=96, SEPC=20, ν=0,12, T=45, une réalisation par point. Suite de la note part184jo.

## 1. Établi

**(a) Le retard se joue pendant l'approche.** Distance xR−xL à t=15 (part184o) : 15,4 (ω=0), 15,8 (0,10), 16,9 (0,20), 17,6 (0,25 parallèles). Une fois les nœuds proches, l'annihilation est rapide. Les spins parallèles affaiblissent l'attraction à longue distance.

**(b) Temps d'annihilation t_ann (Qabs<0,5 interpolé ; incertitude ~0,1 à 0,3) :**

| g² | ω=0 | ω=0,25 opposés | ω=0,25 parallèles | retard (parallèles) |
|---|---|---|---|---|
| 0 | 15,07 | 15,49 | 16,12 | +1,05 |
| 0,01 | 19,92 | | 23,68 | +3,76 |
| 0,02 | 22,88 | | 32,25 | +9,37 |
| 0,04 | 27,76 | | (instable) | |

- Le retard dû au spin est multiplié par ~9 entre g²=0 et g²=0,02. Rapport t_ann(parallèle)/t_ann(ω=0) : 1,07 / 1,19 / 1,41.
- Le champ de jauge ralentit aussi l'approche sans rotation : 15,07 → 19,92 → 22,88 → 27,76 (sous-linéaire).
- Décomposition du retard total 17,18 (32,25 contre 15,07) : jauge seule 7,81 (45 %), spin seul à g²=0 : 1,05 (6 %), terme croisé jauge×spin 8,32 (48 %). L'effet du spin est donc un **couplage non additif jauge×spin**. Je retire « le champ de jauge explique ~90 % du retard » : les deux références (15,07 et 22,88) diffèrent.

**(c) Runs invalides, à ne jamais citer :** les 3 runs g²=0,1 (Y_g01_*) et Z_g004_om025par : Econs saute (×2 à ×3 ; variation relative 0,43 à 1,94), Qabs passe sous 0,5 sans approche (distance 18 à 22). Emballement numérique, pas d'annihilation. Plage de validité : g²≤0,04 à ω=0, g²≤0,02 à ω=0,25. Les 11 autres runs conservent Econs à 5e-6 à 2e-5.

## 2. Compatible, non établi

- **Spin total.** Deux nœuds ω=0,125 parallèles (|L|=428) : t_ann 24,01 ; un seul nœud ω=0,25 (|L|=433) : 23,95 ; retards +1,13 et +1,07 sur la référence 22,88 (écart 0,06 < incertitude). Compatible avec une dépendance au spin total (ou à ∑q, ou ∑ω), mais un seul g², deux runs, et aucun contre-test antiparallèle à g²>0 valide. Je retire « ne dépend pas de la structure de la paire ». Le retard est très non linéaire en L (1,13 à 428 ; 9,37 à 856 : exposant ~3), ce qui ne prouve pas que L soit la bonne variable.
- **Loi en g².** Retard/g² : 376 (0,01) et 468 (0,02), exposant local ~1,3 (1,6 après soustraction du décalage à g²=0) : seulement 3 points, pas de loi.

## 3. Limites

- g²=0 désactive exactement le champ de jauge (θ≡0, Lg=Pg=0). Le « +1,05 » à g²=0 est un effet sans jauge ; g²=0 n'est pas la limite continue de g²→0 (l'énergie initiale de Gauss croît linéairement avec g²).
- Chaque nœud remplit la boîte périodique de 48 unités (NB=96) : l'image périodique de l'autre nœud est à 28 contre 20 pour la paire. Les t_ann absolus dépendent de la boîte ; aucun test de taille de boîte.
- Une seule réalisation, pas de test de h, dt, NB. Qabs(0)=1,951. Ekin(0) diffère d'un run à l'autre (rotation imposée en vitesse).
- Lg final grand (−96 à g²=0,02 sans spin ; −379 à g²=0,04) : le champ de jauge emporte du moment cinétique ; « L total » = L matière à t=0 seulement. Mon calcul de L reste non fiable avec g²≠0.
- Aucun mécanisme (Coulomb, dipôle magnétique, moment cinétique, entraînement) n'est isolé.

## 4. Suite

Run antiparallèle à g²=0,02 avec L≈0 mais J total ≠ 0 pour discriminer L total et structure ; g²=0,015 et 0,03 pour la loi ; orientation de la paire le long de l'axe du spin (nécessite ORIENT=z) pour tester dipôle contre moment cinétique total ; boîte plus grande ou nœuds tronqués contre les images périodiques ; réalisations multiples.
