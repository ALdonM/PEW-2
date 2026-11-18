# part187 — halo d'un nœud du modèle Z (queue, énergie hors du cœur, multipôles)

Question (13–18/11/2026) : si l'on veut une analogie « particule avec halo », que dit le modèle Z de la queue d'un nœud isolé ?
Ce dossier ne contient que de la **mesure** (aucune hypothèse de physique ajoutée) : `halo_mesure.py` (lecture d'un champ),
`halo_relax.py` (relaxation dans une boîte plus grande), `test_halo.py` (tests sur champs analytiques).

## Fichiers
- `halo_mesure.py` : lit un champ `phi (4,N,N,N)` (float16 accepté) et écrit un JSON + un résumé imprimé.
  Même fonctionnelle que `zlat.py` (réécrite en numpy, sans JAX). Mesures :
  1. énergie cumulée E(<r)/Etot (total, E2, E4, EV), **au-dessus du vide** : on retranche la densité du vide du régulateur d2
     (−μ² d2/(1+d2) par unité de volume, soit −4,6 sur 876 pour L=32) ; les totaux bruts du journal sont aussi donnés ;
  2. profils sphériques de 1−ρ, Θ = |∇ζ|², densité d'énergie, courant de phase |j| (avec et sans courant moyen) ;
  3. pentes log-log locales par fenêtres et **pente/r** (constante pour une exponentielle, décroissante pour une loi de puissance) ;
  4. multipôles l = 1..4 de 1−ρ et de la densité d'énergie par coquille (amplitude rms de la partie l / moyenne) ;
  5. second moment de l'énergie (r < 8) : axe unique, oblate/prolate ; dipôle de courant D(<R) sans courant moyen ; flux par plan.
  Contrôle d'images périodiques : rapport axes/diagonales de 1−ρ à même rayon (cubique : 1 sans images).
- `halo_relax.py` : place le nœud au centre d'une boîte N³ (vide (1,0,0,0) autour) et minimise l'énergie (L-BFGS-B, **float64** : en
  float32 L-BFGS-B s'arrête sur le bruit d'arrondi, ~1e-7·E). Points de reprise tous les `--save` itérations.
- `test_halo.py` : pentes r⁻², r⁻⁴, r⁻⁶, exp(−kr) ; multipôles l=1,2 d'amplitude connue ; énergie cumulée d'une gaussienne ;
  centre périodique ; énergie nulle du vide. À lancer avant toute mesure (le workflow le fait).
- `../../.github/workflows/pew-part187-halo.yml` : L=32 (N=64), L=48 (N=96), L=64 (N=128), h=0,5, mêmes paramètres que wk_T0
  (μ=0,375, e0=0,25, d2=1e-3). Les champs relaxés sont des **artefacts** (pas commités) ; journaux et JSON dans `resultats/part187/`.

## Lecture des résultats
- Comparer d'une boîte à l'autre : le rayon médian de l'énergie, la fraction d'énergie hors de r, les pentes locales et pente/r.
- Zone fiable : r < ~L/4 (rapport axes/diagonales de 1−ρ à moins de 2 %) ; au-delà, les images de la boîte dominent.
- l=2 : en boîte cubique, des images identiques n'ajoutent à un nœud sphérique que l=0, 4, 6 ; un l=2 est donc d'abord celui du nœud,
  **sauf** si un courant de phase moyen traverse la boîte (flux Φ ≠ 0 : brise la symétrie cubique) — vérifier `courant_moyen`.
- Le nœud de départ n'est pas au viriel de Derrick (E4/(E2+3EV) = 1,30) : dans une boîte plus grande il peut grossir ; sa taille
  fait partie du résultat.

## Résultat préliminaire (test local N=96, 100 itérations)
E : 912,9 → 858,1 ; E2 : 367 → 313 ; E4 ≈ constant ; E4/(E2+3EV) : 1,26 → 1,44 ; Q = −0,983 inchangé. Le nœud **bouge** en boîte L=48 :
sa « stabilité » à L=32 était en partie celle de la boîte. D'où l'intérêt de L=32/48/64.

## Limites
Un seul nœud (Q=−1), un seul h, quatre paramètres du modèle fixés ; aucune dynamique ; rien n'est identifié à une particule (option (D)).
