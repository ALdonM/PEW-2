# NOTE — session 35 (14/11/2026) — part184h : impulsion axiale annulée, séparation spin / orbital

**Statut : AUDITÉ le 14/11 par un agent isolé** (recalcul depuis les journaux). Rien n'est publié ; rien n'est identifié à une particule (option (D)). Complète la note part184g.
Run GitHub 43599197170 ; données `resultats/part184/gh_part184h/` ; workflow `pew-part184h-cancelpz.yml` ; `jauge.py` option `CANCEL_PZ=1` (boost le long de z annulant Pz par nœud), champs `Pg`, `PzLR`. Paire sur x, sép. 20, N=96, T=38, g²=0,02, ν=0,12.
Cas : Q1 ω=0,2 « Jopp » (spins parallèles) + annulation ; Q2 ω=0,3 « Jopp » + annulation ; Q3 ω=0,3 « Jsame » (spins opposés) + annulation ; Q4 ω=0,3 « Jsame » sans annulation (réplique de P5, dont le job avait échoué sans journal).

## Résultats (audités)
- **Annulation de Pz :** PzLR = [0 ; 0] à t=0 pour Q1/Q2/Q3 (Q4 : [28,7 ; 28,7]). L_y orbital tombe de 382 (P1) à 5,85 à t=0 (il repart vers ≈90 vers t=5, compensé par Lg_y). Qabs initial inchangé (1,955). **Réserve :** J n'est pas strictement préservé : qL passe de 117,9 (P1) à 111,4 (Q1), −5,5 % (L_z et Ekin −5 à −6 %, Etot −0,1 %).
- **Annihilation (Qabs<0,5, interpolé) :** P3 24,71 ; P2 24,64 ; Q3 24,62 ; Q4 24,56 ; **P1 31,02 ; Q1 30,06** ; Q2 et P4 jamais (Qabs ≈ 1,957–1,961 à t=38). Q3 et Q4 : écart 0,06 (l'annulation de Pz ne change rien) ; Q1 et P1 : écart 0,96 pour un L_y de 6 contre 382.
- **Séparations :** Q2 : demi-séparation 9,54 → 9,82 → 11,13 → 12,69 à t=10, 20, 30, 38 : **repoussement permanent**. Cas spins opposés : 1,3–1,4 à t=30.
- **P5 :** Q4 (mêmes paramètres) a réussi : l'échec de P5 est compatible avec un incident de runner (non démontré faute de journal).
- **Bilan d'impulsion (Pm+Pg) :** Q1, Q2 : conservé (≤0,9 à t=38). **Q3 et Q4 : non conservé** (Q3 : (0,47 ; −21,3 ; −6,4) à t=38, soit 27 % de Pm_z ; Pm_y ≈ −18 identique dans Q3 et Q4). Cause probable numérique (définition du Poynting sur le réseau) ; non établie. L_total n'est pas conservé non plus (Q1 : L+Lg en z = −369 à t=38 contre −443 à t=0).

## Conclusions
- Le retard (ω=0,2) et le blocage (ω=0,3) de l'annihilation **ne viennent ni de l'impulsion axiale ni du moment orbital L_y** : ils persistent quand ceux-ci sont retirés. Ils sont liés à des **spins spatiaux parallèles** (L_z total ≠ 0). Des **spins opposés annihilent normalement, même à ω=0,3** (Q3, Q4 : 24,6).
- Ce qui n'est pas isolé : (i) **interaction spin–spin vs conservation du moment cinétique** (les deux prédisent le même signe) ; (ii) **J_net** : « Jsame » a J_net ≠ 0, « Jopp » est neutre en J, la différence de spin est mêlée à une différence de charge J ; (iii) le seuil entre ω=0,2 (retard de ~5–6) et ω=0,3 (blocage) n'est pas testé (ω=0,25 manquant).
- Réserve générale : une seule réalisation par cas ; L et P mal conservés au-delà de t≈10.

## À faire
1. ω=0,25 « Jopp » (seuil) et ω=0,1/0,15 (loi de retard) ; 2. séparer spin–spin et conservation de L (par ex. L_total compensé par un moment orbital opposé, ou nœuds de spins parallèles à plus grande séparation) ; 3. séparer J_net et spin ; 4. corriger/valider le Poynting et la conservation de L, P ; 5. test de résolution h et réplique de P5 en conditions identiques.
