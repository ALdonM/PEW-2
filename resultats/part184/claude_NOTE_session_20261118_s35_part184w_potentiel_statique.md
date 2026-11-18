# Note part184w : énergie d'interaction statique en fonction de la distance

Session s35, 2026-11-18. Audit isolé (agent séparé, recalcul depuis les journaux) avant écriture ; il a renversé mes deux conclusions principales, ses corrections sont intégrées. Modèle Z classique, aucune identification à une particule (discipline D). Données : `resultats/part184/gh_part184w/` (P1-P8 et relax_N96.log). Nœud relaxé N=96, 120 unités (rms_force final 2,2e-5, énergie stable à 0,01 près, mais |Z2|² dérive encore d'environ +2 par 10 unités). Méthode : énergie conservée Econs à t=0 de la paire (produit de quaternions figé, sans réponse des champs) moins deux fois celle du nœud seul, pour des séparations d=4 à 24 (boîte de 48). Pz annulé pour les cas en rotation, ω=0,25.

## 1. Chiffres (tous confirmés à ±0,1)

E_int(d) à ω fixé :

| d | z ω=0 | z par. | z anti. | x ω=0 | x par. | x anti. |
|---|---|---|---|---|---|---|
| 4 | +182 | +228 | +132 | +880 | +825 | +890 |
| 8 | −318 | −282 | −342 | +129 | +99 | +143 |
| 10 | −218 | −189 | −232 | −4,3 | −25 | +12,5 |
| 12 | −115 | −97 | −121 | −56 | −68 | −37 |
| 14 | −53 | −45 | −55 | −57 | −61 | −41 |
| 18 | −6,3 | −4,9 | −6,0 | −23,5 | −21,4 | −17,6 |
| 20 | +1,4 | +1,8 | +1,8 | −11,4 | −9,3 | −7,9 |
| 24 | +6,1 | +5,8 | +6,5 | −2,6 | −0,8 | −0,6 |

g²=0 sur l'axe (P7, P8) : proches de P2, P3. Sur l'axe, puits profond vers d≈8 ; côte à côte, cœur répulsif pour d<10 et puits vers d≈13. Le champ p est identique dans toutes les configurations d'une même orientation : toute la dépendance en spin vient du terme croisé d'énergie cinétique (rotations imposées) et de l'énergie de Gauss à t=0. Le spin ne déforme pas le nœud dans cette construction.

## 2. Ce que l'audit corrige

- **Ensemble de comparaison.** Comparer à ω fixé est trompeur : dans la dynamique, c'est la charge de rotation de chaque nœud qui est conservée, pas ω. À charge fixée, le terme cinétique entre dans le potentiel efficace avec le signe opposé (au premier ordre V − K au lieu de V + K). L'« inversion d'ordre » que j'avais annoncée entre statique et dynamique est un artefact de ce choix : **je la retire**.
- **Estimation à charge fixée** (modèle de rotor à deux degrés de liberté de l'audit, bon pour le signe, approximatif pour l'amplitude), écart à ω=0 : z antiparallèles +21,6 (d=8), +6,7 (d=10) ; x parallèles +33,5 (d=8), +13,9 (d=10) ; z parallèles −2,7 ; x antiparallèles −3,8. À d=4 la paroi des antiparallèles sur l'axe est beaucoup plus raide (+365 contre +182). Le signe redevient conforme à la dynamique : les configurations les plus retardées (z antiparallèles, x parallèles) sont les moins liées à courte distance. Seul le retard des parallèles sur l'axe (+6) reste inexpliqué statiquement. Pour d≥14, toutes les configurations sont à ≤1,6 de ω=0 en énergie, et à ≤0,35 en force pour d≥18.
- **Champ de jauge.** Son énergie de Gauss est petite en absolu (≤13), mais du même ordre que la partie dépendante du spin. Surtout, à t=0 le champ de jauge spatial est nul : le calcul statique ne contient que la partie coulombienne, aucune interaction magnétique entre courants. Je retire « la jauge change peu, son rôle est dynamique » : le statique ne teste pas son rôle magnétique.
- **Grande distance.** E_int ne tend pas vers 0 à d=24 (+6 en z, −2,6 en x), principalement par le terme de masse ν²|Z2|² (non additif dans le produit). Ce n'est pas une bosse : E croît de façon monotone de d=8 à 24, avec un extremum imposé par la symétrie de la boîte (F(24)=0). L'attribution aux images n'est pas démontrée (la queue anisotrope, de signes opposés en z et x, peut aussi venir du mode sans masse de la phase de Z1). « Pas de barrière » est restreint à d≤24 dans cette boîte.

## 3. Ce qui reste

- Les écarts liés au spin sont concentrés à courte distance (d≤12) ; au départ des paires (d=20) les forces statiques liées au spin sont faibles. Le retard se construit donc dans la phase rapprochée ou par un effet magnétique non contenu dans ce calcul.
- Le lien quantitatif entre ce potentiel à charge fixée et les retards mesurés reste à faire.

## 4. Limites

Produit figé ; vitesse imposée sans relaxation du nœud en rotation ; une seule relaxation (|Z2|² en dérive) ; boîte de 48, d ≤ 24 ; Pz annulé moitié par moitié (écart ≤1,1 à d=4) ; pas de réponse des champs ni de champ magnétique initial ; modèle rotor approximatif.

## 5. Suite

Calcul direct à charge fixée (relaxer un nœud en rotation à J fixé, puis la paire à J fixés) ; inclure le champ magnétique initial cohérent avec les courants ; boîte plus grande pour la queue ; extraire un potentiel efficace dynamique de xR−xL(t) pour comparaison.
