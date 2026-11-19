# Note de reprise : part184, état au 2026-11-19 (session s35 → nouvelle session)

Document d'état, sans résultat nouveau : tous les chiffres viennent des notes auditées citées. Les résultats de part184zb ne sont pas encore là.

## 1. Question et règles

Question : l'annihilation d'une paire de nœuds de Hopf (modèle Z classique, `numerical_minimization/protect3d_Z/jauge.py`) exige-t-elle une rotation interne opposée ? Constat de départ : des spins parallèles retardent l'annihilation. Objectif : le mécanisme exact.

Règles permanentes : pousser uniquement sur ALdonM/PEW-2 ; audit isolé (agent séparé, recalcul depuis les journaux) avant chaque note de résultats ; consulter le dictionnaire avant de créer une variable ; ne jamais appeler « photon » ou « électron » un objet du modèle sans preuve (option D) ; rien d'autre sur GitHub/Zenodo sans accord de Michel ; lire parallèle/antiparallèle sur les signes mesurés de Lz et J, pas sur les noms sj ; Econs seule fait foi (Etot, EA, Eext, xL, centroïde sont faux en jauge temporelle). Michel : tutoiement, challenger sans flatter, réponses courtes (téléphone). Trailers de commit : `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` et `Claude-Session: https://claude.ai/code/session_012qTVdBJ7QYur7BNLAidNKZ`.

## 2. Acquis (notes auditées dans `resultats/part184/`)

- 043f548 jo, 44812f6 pq, 971b80b r, 3debfa0 s, b332b7c t, 34abcd7 uv, fb6f4f4 w, a2f34b7 xy, 4aa4e97 zza (dernière).
- Nœud chargé en rotation à l'équilibre avec son champ de jauge (état stationnaire : Z2 tourne à Ω(x)=ω+ψ, loi de Gauss, équilibre magnétostatique) : disponible, stable sur 60 unités.
- Motif qualitatif des paires (g²=0,02, ω=0,10 et 0,14) : côte à côte parallèles +0,73/+1,32 ; côte à côte opposés −0,36/−0,64 ; axe parallèles −1,11/−2,16 ; axe opposés +1,52/+5,50. Références ω=0 : axe 20,26 ; côte à côte 23,01 (15,1 sans jauge).
- Retrait : « une grande partie des retards vient du départ hors équilibre ». À charge égale, le retard côte à côte parallèle passe seulement de +2,2 à +1,3.
- Le retard commun côte à côte à ω=0 (23,0 contre 15,1) survit à l'équilibre magnétostatique.
- Dans les paires de charges opposées, la charge de chaque moitié (qL, qR) tombe vers 0 avant le contact (172 → 54 à t=10 → 9 à t=15, ω=0,14).
- Exclus : dipôle pur (1−3cos²θ), box N=128 comme remède, comparaison à ω fixé (artefact : comparer à charge fixée).

## 3. En cours : workflow part184zb (lancé 2026-11-19 ~18h, run 44572873080)

Fichier `.github/workflows/pew-part184zb-charge-egale-flux.yml`, commit f6a454c. Résultats attendus dans `resultats/part184/gh_part184zb/` (journaux poussés par le workflow ; durée plausible plusieurs heures, 2 relaxations puis 16 paires).

Deux volets :
1. **Charge égale avec et sans jauge.** Charges totales mesurées : jauge ω=0,10 → 117, ω=0,14 → 198 ; sans jauge ω=0,10 → 184, ω=0,14 → 326. Appariements visés : jauge 0,10 ↔ sans jauge 0,077 (nouveau, `ZB0_*_om077`) ; jauge 0,14 ↔ sans jauge 0,10 (`ZB0_*_om10`) ; jauge 0,19 (nouveau, `ZBG_*_om19`) ↔ sans jauge 0,14 (existant : `gh_part184y/Y0_*_om014`). ω=0,077 et 0,19 sont des extrapolations : mesurer les charges réelles dans les journaux avant d'apparier. Quatre configurations par ω : z_par, z_anti, x_par, x_anti.
2. **Où va la charge.** Nouveau diagnostic QPROF (env `QPROF=1`) : champs `qx` (charge par tranche de 4 le long de l'axe, 12 tranches de −24 à 24), `qfar` (charge à R>22), `qtot`, `Eel` (énergie électrique). Rejeu avec QPROF des 4 paires de ZA à ω=0,14 (`ZBQ_*_om14`, doivent redonner les t_ann de part184za : contrôle). Questions : la charge traverse-t-elle le plan médian (courant), reste-t-elle dans le champ électrique, part-elle en rayonnement (qfar) ? Noter que dqL/dt = −flux à travers le plan médian si qL+qR est conservé.

Avertissement de lecture : t_ann = Qabs<0,5 interpolé (incertitude de seuil 0,1 à 0,3) ; sur l'axe antiparallèle, t_ann mal conditionné (±50 %).

## 4. Limites connues

Une réalisation par point ; relaxation non convergée à ω=0,14 (|Z2|² −2,5 % sur 20 unités) ; zéro de ψ dépendant de la boîte (±50 % sur l'amplitude) ; profils différents entre ω=0 et ω≠0 ; relaxation avec jauge à ω=0,19 sur 100 unités seulement.

## 5. Suite après part184zb

Lire les journaux (recalcul indépendant), audit isolé, puis note. Ensuite selon résultats : relaxations plus longues, quelques réalisations pour barres d'erreur, potentiel statique à charge fixée, boîte plus grande, gravité (champ de Nordström) toujours reportée par Michel, idée du mode sans masse de Z1 comme porteur (basse priorité), synthèse de la série part184.
