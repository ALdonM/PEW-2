# Note part184r : orientation de la paire par rapport à l'axe du spin, loi en g² fine

Session s35, 2026-11-17. Audit isolé (agent séparé, recalcul depuis les journaux) avant écriture ; ses corrections sont intégrées. Modèle Z classique, aucune identification à une particule (discipline D). Données : `resultats/part184/gh_part184r/` (R_*). Nœud relaxé N=96 (RELT=120), NB=96, SEPC=20, ν=0,12, T=45, une réalisation par point. `ORIENT=z` ajouté à jauge.py (paire le long de z, miroir z→−z). Suite de la note part184pq.

## 1. Établi

**t_ann (Qabs<0,5 interpolé ; incertitude ~0,1 à 0,3, stable pour les seuils 1,5 à 0,2) à g²=0,02, ω=0,25, sauf mention :**

| Orientation | spins (Lz à t=0) | t_ann | référence ω=0 | écart |
|---|---|---|---|---|
| x (côte à côte) | parallèles (−656) | 32,25 | 22,88 | +9,4 |
| x | opposés (−4) | 23,15 | 22,88 | +0,3 |
| z (sur l'axe du spin) | parallèles (−647) | 17,58 | 19,90 | −2,3 |
| z | opposés (+1) | 40,36 | 19,90 | +20,5 |

Chaque écart n'a de sens qu'à orientation fixe : à ω=0, z donne 19,90 et x 22,88 (le nœud est anisotrope, le produit A·B dépend de l'orientation). La correspondance sj/spin est inversée entre x et z (le miroir x inverse Lz, le miroir z le conserve) : « parallèle » se lit sur Lz mesuré, jamais sur le nom du paramètre.

**Loi en g² (x, ω=0,25 parallèle) :** retard +1,05 (g²=0), +3,76 (0,01), +5,82 (0,015), +9,37 (0,02), puis **pas d'annihilation à g²=0,03 dans T=45** (retard ≥19,6 ; ω=0 : 25,41). Croissance superlinéaire ; Econs conservé (1e-5), ce n'est pas un emballement. Run de référence invalide à ne pas citer : Z_g004_om025par (Econs saute).

Nœud relaxé identique entre R_x et Y/Z (Z2 et Econs à t=0 égaux) ; Econs conservé à ≤1,7e-5 sur les 8 runs.

## 2. Hypothèse « dipôle magnétique via la jauge » : seulement compatible, plutôt fragilisée

Le motif (parallèles côte à côte : retard ; parallèles sur l'axe : accélération ; antiparallèles sur l'axe : retard fort, ratio ~2,2 avec le côte à côte ✓) ressemble à une interaction dipôle–dipôle. Mais l'audit montre :
- les antiparallèles côte à côte (+0,3) devraient être attractifs (≈ −9) : ✗ ; seules 2 configurations sur 4 tiennent en signe et amplitude ;
- chaque nœud tournant reçoit au départ une impulsion axiale Pz ≈ +27 (rotation imposée en vitesse). Sur l'axe z (z_same), les deux Pz sont opposés : la paire se rapproche plus vite dès t=2 (séparation 19,79 contre 19,95 ; 17,52 contre 18,51 à t=10) : l'accélération −2,3 est en partie cinématique ;
- un seul nœud tournant (μB=0) donne le même retard (+1,07) que deux nœuds ω=0,125 (+1,12), proches du +1,05 sans jauge : compatible avec un terme d'environ +1 indépendant de la jauge ; le test « un seul nœud » n'a pas été fait à g²=0 ;
- le retard X (ω=0,10 / 0,20 / 0,25 : +0,74 / +4,21 / +9,78) est plus raide que ω² ;
- alternatives non exclues : interaction coulombienne des charges J (J et L corrélés, aucun test ne les sépare), anisotropie du nœud, images périodiques (distance à l'image 28 contre 20 pour la paire).

À retirer : « divergence entre g²=0,02 et 0,03 » (non établi : retard ≥19,6, seuil possible, non démontré ; à vérifier avec g²=0,025 et T plus long) ; « le retard dépend du moment cinétique total seulement » (déjà retiré en part184pq). Dans le |L|=856 des paires parallèles en x, Ly=550 est orbital (bras de levier × Pz), le spin est Lz≈−656.

## 3. Limites

Une réalisation par point, un seul N, une seule boîte, pas de convergence en h, dt, NB ; les queues du nœud (NB=N) recouvrent celles des images périodiques ; t_ann défini par un seuil sur Qabs ; Ekin(0) diffère d'un run à l'autre (rotation imposée).

## 4. Suite (part184s lancé : run 43999150592)

Impulsion axiale annulée (CANCEL_PZ) en orientation z et x ; z avec g²=0 (le signe d'orientation disparaît-il sans jauge ?) ; un seul nœud tournant à g²=0 ; antiparallèle à g²=0,03 (attraction ?). Ensuite : test à 45° de l'axe, g²=0,025, T plus long, nœuds tronqués contre les images.
