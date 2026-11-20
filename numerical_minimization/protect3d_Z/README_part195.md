# part195 — cinétique rescalé : vitesse des ondes ∝ ρ_vac
Contexte : part194 (milieu fixe) laisse c₀ ∝ 1/ρ_vac (KT=1), inverse du glossaire (C̃₀ ∝ Ã_vac). On ajoute KT : L = ½·KT·|∂ₜφ|² − ½E, avec KT = ρ_vac⁻⁴ ⇒ c₀ ∝ ρ_vac. Vérifié hors simulation (mode plan de phase, N=32, h=0,5) : c = 0,994 / 0,696 / 0,497 pour ρ_vac = 1 / 0,7 / 0,5 (KT=ρ_vac⁻⁴) contre 0,994 / 1,42 / 1,99 (KT=1) ; 0,994 = dispersion du réseau.
Runs : mêmes réglages que part194 (96³, éponge 16→24, MEANC=0, t=40, WA=WR=WV=ρ_vac⁻²), ρ_vac=0,5 pour μ_a=0,375 et 0,05, ρ_vac=0,7 pour μ_a=0,05.
## Prédictions [P]
- Statique inchangée : M, R_rms, G_eff, pente à ±5 % de part194 au même (μ_a, ρ_vac) (l'équilibre ne dépend pas de KT ; seul le chemin de relaxation, plus lent si c plus petit, change). Réfutation : écart > 10 % ⇒ relaxation incomplète à t=40 (non stationnaire) à examiner.
- Charge −1 conservée.
Suite : retard de phase de la houle (démodulation k=3) vs ρ_vac ; puis paires.
