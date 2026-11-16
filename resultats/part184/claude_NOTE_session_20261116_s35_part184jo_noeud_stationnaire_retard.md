# Note part184j à o : état initial non stationnaire, puis retard d'annihilation avec un nœud mieux relaxé

Session s35, 2026-11-15/16. Audit isolé (agent séparé, recalcul depuis les journaux) effectué avant écriture ; ses corrections sont intégrées. Modèle Z, classique ; aucune identification à une particule (discipline D). Données : `resultats/part184/gh_part184{j,k,l,m,n,o}/`.

## 1. Ce qui est établi

**(a) Le nœud « relaxé » des runs j, k, l était hors équilibre dans la grande boîte.** Relaxé à N=48 (côté 24) puis enchâssé dans N=96, il reçoit un choc : Ekin(t=1) = 12,49 / 12,70 / 12,69 pour des relaxations de 20 / 100 / 300 unités (V1-V3), et |Z2|² passe de ~1157 à ~1990 (t=12). Les relaxations de 100 et 300 unités sont convergées (rms_force 2,7e-6 et 1,2e-6) et donnent le même résultat : ce n'est pas la durée de relaxation. Le |Z2|² relaxé dépend de la boîte : 1157 (N=48), 1433 (N=64), 1522 (N=96, 80 unités). Avec N=64 (V4) : Ekin(t=1) = 4,03 et |Z2|² de 1433 à un pic de 1713 (t=16).

**(b) Relaxé directement en N=96 (part184n), le nœud seul respire beaucoup moins.** W2 (80 u) : Ekin(t=1) = 0,00 (max 0,05 sur tout le run), |Z2|² 1522 → pic 1565 (+2,8 %). W1 (40 u) : +12,7 %. Réserve de l'audit : la relaxation de W2 n'est pas convergée (rms_force 4,9e-5, |Z2|² encore croissant : 1512 à 70 u, 1522 à 80 u). On ne peut donc pas dire « stationnaire », seulement « beaucoup plus proche de l'équilibre ». Le nœud occupe toute la boîte (NB=96) : effets de bord périodiques non exclus.

**(c) Un nœud en rotation (ω=0,25, W3) n'est pas non plus un état propre.** Lz −316,6 → −291,0 à t=24 (−8,1 %), JA 137,6 → 64,0, |Z2|² 1522 → pic 2177 (t=17). La perte de Lz de 16 % observée en part184l est donc divisée par ~2 avec un meilleur nœud, mais pas annulée.

**(d) Retard d'annihilation avec le nœud W2 (part184o, N=96, NB=96, SEPC=20, g²=0,02, ν=0,12, T=45, une réalisation par point).**

| Run | ω | spins | t_ann (Qabs<0,5, interpolé) | retard vs X1 | retard/ω² |
|---|---|---|---|---|---|
| X1 | 0 | | 22,96 | | |
| X2 | 0,10 | parallèles | 23,70 | 0,74 | 74 |
| X3 | 0,20 | parallèles | 27,17 | 4,21 | 105 |
| X4 | 0,25 | parallèles | 32,74 | 9,78 | 157 |
| X5 | 0,25 | opposés | 23,23 | 0,28 | |

Les spins parallèles retardent l'annihilation ; les spins opposés non (0,28 est du niveau de l'incertitude d'interpolation, ~0,1 à 0,3 : « pas de retard significatif »). Le retard croît plus vite que ω² (exposant local ≈2,5 entre 0,10 et 0,20, ≈3,8 entre 0,20 et 0,25) ; ce n'est pas une loi de puissance simple et je n'en déduis ni loi ni seuil. Avec l'ancien nœud (non stationnaire) le retard à ω=0,25 était de +11,8 : même signe, même ordre. Aucun blocage à ω=0,25 dans T=45 (l'ancienne série bloquait à 0,3, non testé ici).

**(e) Convention des noms.** Dans le code, sj=same / opp désigne la rotation *interne* de phase. Parce que B est la réflexion de A (sq=opp), sj=opp donne des spins spatiaux parallèles et sj=same des spins opposés. Mesuré à t=0, ω=0,25 : X4 |L|=855 (Ly=551,5 ; Lz=−653), X5 |L|=8,5. L'audit rappelle que L a une composante Ly du même ordre que Lz (X4) : parler de « Lz seul » sous-estime |L|.

## 2. Ce qui n'est pas établi / à nuancer

- Une seule réalisation par point, pas de barres d'erreur ; pas de test de h, dt, N sur la série X.
- Dans X, chaque nœud remplit toute la boîte (NB=96) et les deux sont décalés de ±20 mailles par `np.roll` : ils se recouvrent et ont des images périodiques. Ce biais est commun à X1-X5 ; l'effet de la géométrie de boîte sur le retard n'est pas testé.
- Qabs(t=0) = 1,951 et non 2 pour tous les X (discrétisation et recouvrement).
- Lz n'est pas monotone dans X2-X4 (X4 : −653, −449 à t=10, −665 à t=35, min −695) : mon calcul de L reste non fiable avec g²≠0 ; je ne l'utilise pas comme variable explicative.
- Etot (potentielle seule, l.148) chute de ~40 % dans X : ce n'est pas une fuite d'énergie. La grandeur conservée est Econs (conservée à 1e-5 relatif sur X1-X5).
- Les runs j, k, l (nœud enchâssé non stationnaire) ne sont pas comparables quantitativement à o ; seules leurs conclusions qualitatives sont reprises. Le rejet de l'hypothèse d'un seuil de masse à ω≈0,17 (part184l : J retenu dans la boule ≈0,33 / 0,32 / 0,30 / 0,28 à t=15, indépendant de ω) a été obtenu avec ce nœud ; il reste à refaire avec W2.
- Le mécanisme du retard (spin–spin, dispersion de J, respiration du nœud) n'est pas isolé.

## 3. Affirmations retirées ou corrigées par rapport à mes messages de session

- « Relaxer plus longtemps ne change rien » → « la durée à N donné est sans effet ; la taille de boîte de relaxation compte ».
- « Nœud stationnaire » (W2) → « beaucoup plus proche de l'équilibre, relaxation non convergée ».
- Pic |Z2|² de W3 : 2177 (t=17), pas 2164 ; pic de V4 : 1713 (t=16).
- Pas de loi en ω² ni en ω^n.

## 4. Suite possible

Relaxer W2 jusqu'à rms_force ~1e-6 ; refaire la série avec séparation plus grande ou nœud tronqué pour limiter le recouvrement ; ω=0,27 / 0,28 / 0,30 ; deux ou trois réalisations (perturbations) pour estimer l'incertitude ; refaire le test de rétention de J (part184l) avec W2.
