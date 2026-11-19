# part184zb : charge égale avec et sans jauge, et où va la charge (QPROF)

Run 44572873080 (16 paires, succès). Journaux : `gh_part184zb/`. Recalcul : `analyse_zb.py`. Audit isolé par un agent séparé (recalcul depuis les journaux) : 6 points vérifiés, corrections intégrées ci-dessous.

## Lecture des noms (piège)
Je classe par signe mesuré de J (JA·JB à t=0). En z : `par` = J de même signe (J//), `anti` = J opposé (Janti). **En x c'est inversé** : `x_par` = Janti, `x_anti` = J//. De plus en x, J et L ne vont pas ensemble : J opposé ↔ |L| grand (429 à 1367 selon ω), J de même signe ↔ |L| petit (7 à 25). En z les deux critères s'accordent (J// : Lz_in −136 à −546 ; Janti : L=0). « Parallèle » est donc ambigu en x : à préciser avant toute conclusion sur le spin.

## Résultats (t_ann = Qabs<0,5 interpolé ; charge = (|qL|+|qR|)/2 à t=0)
Sans jauge (ZB0, Y0) :

| config | ω0,077 | ω0,10 | ω0,14 |
|---|---|---|---|
| x J// | 14,93 (q153) | 14,88 (q219) | 14,95 (q407) |
| x Janti | 15,01 (q121) | 14,95 (q168) | 14,69 (q285) |
| z J// | 19,93 (q144) | 19,38 (q207) | 17,94 (q387) |
| z Janti | 22,70 (q129) | 26,31 (q181) | 30,96 (q316) |

Avec jauge g²=0,02 :

| config | ω0,10 (ZA) | ω0,14 (ZBQ=ZA) | ω0,19 (ZBG) |
|---|---|---|---|
| x J// | 22,65 (q128) | 22,37 (q223) | 22,63 (q498) |
| x Janti | 23,74 (q108) | 24,32 (q172) | 25,27 (q292) |
| z J// | 19,15 (q126) | 18,10 (q223) | 16,57 (q518) |
| z Janti | 21,78 (q114) | 25,76 (q185) | 18,86 (q326) |

Appariement par charge mesurée : jauge ω0,14 ↔ sans jauge ω0,10 : rapports 1,018 à 1,023 (x, z Janti), **1,078 pour z J//** (l'affirmation « ≤7 % » était fausse). Jauge ω0,19 ↔ sans jauge ω0,14 : 1,024 et 1,031 pour les Janti, mais **1,34 (z J//) et 1,22 (x J//)** : pas à charge égale. Jauge ω0,10 ↔ sans jauge ω0,077 : sans jauge plus chargé de 12 à 20 %. L'appariement fait sur les charges du nœud seul (117/198/184/326) était donc trop grossier, comme prévu.

Contrôle : ZBQ_om14 redonne ZA_om14 à l'identique (écart 0,00 sur t_ann, q0, Qabs, Econs : calcul déterministe, donc ce n'est pas un test indépendant).

## Conclusions (à charge à peu près égale)
1. **Côte à côte (x) : le retard est celui de la jauge, pas du spin.** Sans jauge t_ann ≈ 14,7–15,0 pour les 6 couples (ω 0,077 à 0,14, les deux signes de J), aucune dépendance en ω ni en J. Avec jauge 22,4 à 25,3, ce qui est proche de la valeur à ω=0 (23,01). L'ancien « +0,7/+1,3 parallèles, −0,4/−0,6 opposés » mesurait donc des écarts de 1 à 2 unités autour d'un retard commun de ~8 dû à la jauge. Reste un petit effet avec jauge : x Janti plus lent que x J// de +1,1 à +2,6 aux trois ω (une réalisation, seuil 0,1–0,3, charges non identiques à ω0,19).
2. **Sur l'axe (z) : J opposé est plus lent que J de même signe**, dans les 7 couples, avec et sans jauge (écart +2,3 à +13). L'annihilation n'exige donc pas une rotation interne opposée : les paires J// s'annihilent partout (16,6 à 22,6). À charge égale, la jauge change peu en z J// (−0,8 à −1,2).
3. **Non expliqué** : z Janti avec jauge n'est pas monotone en ω (21,8 → 25,8 → 18,9), alors que sans jauge il croît (22,7 → 26,3 → 31,0). À ω0,19 il est 12 unités plus rapide que sans jauge à charge égale (3 %). Suspects : relaxation de ω0,19 sur 100 unités seulement, dépendance de boîte, t_ann mal conditionné sur cet axe (±50 %). Ne pas conclure.

## Où va la charge (QPROF)
- Paires J anti : qL/q0 décroît vers 0 **puis change de signe** (sans jauge en x : 0,74 à t=5, 0,10 à 10, −0,6 à 15), avec qL=−qR et qtot constant. Comme dqL/dt = −flux au plan médian, **la charge traverse le plan médian**. Le passage par 0 précède t_ann de 2,1 à 6,9 unités (ZBG_z_anti : 0,4, le changement de signe a lieu entre t=18 et 19).
- Pas de départ en rayonnement : max|qfar| avant t_ann ≤ 2,6 pour tous les Janti.
- Paires J// : qL/q0 reste entre 0,996 et 1,036 jusqu'à t_ann, aucun flux au plan médian.
- Avec jauge, Eel est multipliée par ~10 à 25 à t_ann (x Janti ω0,14 : 9,4 → 249 ; z Janti : 10,6 → 128) : l'énergie passe dans le champ électrique. Interprétation non testée.
- Réserve : qfar vaut 17 % (ZBG_z_par) et 20 % (ZBG_x_anti) de la charge totale à ω0,19 avec jauge, contre 7 à 9 % à ω0,14 : la charge touche le bord à R>22, la boîte compte à ω0,19.

## Limites
Une réalisation par point ; relaxation de ω0,19 sur 100 unités ; zéro de ψ dépendant de la boîte ; t_ann à seuil 0,5 (±0,1 à 0,3) ; un Qabs résiduel de 0,05 à 0,12 en fin de run est le plancher du comptage. Rien dans les journaux n'est un NaN ni une dérive d'Econs (≤0,03). Rien n'identifie ces objets à un photon ou un électron (option D).

## Suite possible
Relaxation de ω0,19 plus longue et boîte plus grande ; plusieurs réalisations ; paires avec charge ajustée par ω pour égaler exactement (z J// et x J// à ω0,19) ; clarifier « parallèle » en x (J ou L) ; lire le courant au plan médian directement pour confirmer le flux.
