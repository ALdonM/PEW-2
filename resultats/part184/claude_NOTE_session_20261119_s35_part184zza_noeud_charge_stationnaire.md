# Note part184z-za : nœud chargé en rotation à l'équilibre avec le champ de jauge, puis paires

Session s35, 2026-11-19. Deux audits isolés (agents séparés, recalcul depuis les journaux, tests synthétiques du code) avant écriture ; leurs corrections sont intégrées. Modèle Z classique, aucune identification à une particule (discipline D). Données : `resultats/part184/gh_part184z/` (relaxations avec jauge relaxg_R*.log, nœuds seuls Z_single_*) et `gh_part184za/` (10 paires ZA_*). N=96, g²=0,02, séparation 20, T=60.

## 1. Méthode

État stationnaire exact en jauge temporelle d'un nœud chargé en rotation : Z2 tourne à la fréquence locale Ω(x) = ω + ψ(x), θ croît de t·∇ψ, loi de Gauss Δψ = g²·CS·Ω|Z2|² (CS=2, fond neutralisant périodique, ψ=0 loin du nœud), avec a_static + Ω²Z2 = 0 et dE/dθ = 0 (équilibre magnétostatique avec les courants propres du nœud). Relaxation par dynamique amortie (option RG2 de jauge.py), 60 unités à partir des nœuds de part184x. Pour les paires (option STATE) : θ = θ_A + miroir de θ_A pour B (lien réfléchi : θ'(i) = −θ(N−2−i), vérifié : énergie identique à 1e-7), ψ résolu de façon auto-cohérente pour toute la paire, vitesse = dérivée du produit + ψ·rotation de Z2. Le signe de la loi de Gauss du code est confirmé (mode gauss, dérivation par invariance de jauge) : ψ < 0 au cœur, le cœur tourne plus lentement que ω, comme pour une Q-ball jaugée.

## 2. Établi

- **L'équilibre magnétostatique change fortement le nœud.** Relaxer θ fait passer l'énergie de 916 à 767 à ω=0 (−16 % ; reproduit indépendamment à N=40 : terme de Faddeev −213 par écrantage du terme (n3−1)F, gradient covariant −19, énergie magnétique +83). |Z2|² 1538 → 1277. Ce n'est pas un bug : le même nœud évalué à θ=0 retrouve l'écart (926,6 contre 767,1).
- **Les nœuds seuls ainsi préparés sont stationnaires** sur 60 unités : Econs à 0,01 près ; ω=0 : Ekin ≈ 0, |Z2|² ±0,8 % ; ω=0,10 : Ekin 9,22 → 9,27 ; ω=0,14 : Ekin 21,7 → 22,1, Lz+Lg −369 ± 4 % (oscillation). Ancienne méthode (part184x) : Ekin 18,7 → 45, Lz −664 → −416.
- **Contrôle** : le même nœud avec θ=0 au départ s'agite (Ekin 0 → 25-29, énergie extérieure −7 → +34). L'équilibre magnétostatique est indispensable. Toutes les paires à g²>0 antérieures partaient de θ=0, donc hors équilibre (excédent mesuré 149 à 153 par nœud pour les nœuds de part184x ; non mesuré pour les séries o à w).
- Charges à ω égal plus faibles qu'à g²=0 (charge totale 117 / 198 contre 184 / 326) : contraction de |Z2|² (×0,80 / ×0,76) et fréquence moyenne abaissée par ψ (×0,78). J total ≈ −1,9 × charge.

## 3. Paires (t_ann = Qabs<0,5 interpolé, confirmés, stables à <0,05 selon le seuil ; Econs conservé ; résidu de Gauss initial ≤3e-5 relatif)

Références ω=0 : sur l'axe 20,26 ; côte à côte 23,01. Retards :

| Configuration (charges J) | ω=0,10 | ω=0,14 |
|---|---|---|
| côte à côte, spins parallèles (opposées) | +0,73 | +1,32 |
| côte à côte, spins opposés (égales) | −0,36 | −0,64 |
| sur l'axe, spins parallèles (égales) | −1,11 | −2,16 |
| sur l'axe, spins opposés (opposées) | +1,52 | +5,50 |

Le motif qualitatif des séries antérieures survit en signe aux deux ω.

## 4. Ce que l'audit corrige

- **Je retire** « une grande partie des retards à g²>0 venait du départ hors équilibre ». À **charge égale** (les nœuds à ω=0,14 ici ont presque les charges des nœuds de part184x à ω=0,10 : charge dans la boule 82 contre 85, J −369 contre −373), le retard côte à côte des spins parallèles passe de +2,2 à +1,3 seulement. La forte baisse apparente (+6,9 → +1,3 à ω égal) vient surtout de la charge plus faible. Le retard côte à côte croît fortement avec la charge (J≈58 : +0,7 ; J≈84 : +1,3 à +2,2 ; J≈123 : +6,9).
- **Je retire** « le retard sur l'axe des spins opposés chute de +7,8 à +5,5 » (dans le bruit de ±50 % vu en part184u) et « les retards se rapprochent des valeurs sans jauge » (vrai pour le côte à côte parallèle seulement).
- Ces comparaisons concernent part184x (ω≤0,14), pas les séries à ω=0,25 (part184o à w), non retestées.
- **Le retard commun côte à côte à ω=0** (23,0 avec jauge contre 15,1 sans) survit à l'équilibre magnétostatique : il ne vient pas du départ hors équilibre (réserve : profils différents).
- **Constat nouveau** : dans les paires de charges opposées (côte à côte parallèles, axe opposés), la charge de chaque moitié s'écoule vers 0 avant le contact, dans les trois séries (ici 172 → 54 à t=10 → 9 à t=15). Les nœuds arrivent presque neutres : on ne mesure pas une force entre nœuds chargés en rotation jusqu'au contact.

## 5. Limites

Une réalisation par point ; relaxation non convergée à ω=0,14 (|Z2|² −2,5 % sur les 20 dernières unités) ; zéro de ψ dépendant de la boîte (amplitude ±50 %, Ω décalé de ±6 % dans les paires) ; profils différents entre ω=0 et ω≠0 ; superposition θ_A + θ_B testée sur un nœud synthétique seulement (force résiduelle 1,7 % de celle à θ=0) ; impulsion axiale nette non nulle (~43 pour les configurations « même sens » de Pz) ; diagnostics Etot, EA, centroïde faussés en présence de jauge (non utilisés).

## 6. Suite

Comparer à charge égale de façon systématique (choisir ω pour fixer la charge) ; paires sans jauge (g²=0) aux mêmes charges ; suivre l'écoulement de charge des paires de charges opposées (où va-t-elle : rayonnement, champ de jauge ?) ; relaxations plus longues ; quelques réalisations.
