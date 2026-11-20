# part194 — milieu entièrement fixe (potentiel inclus), éponge sphérique
Contexte : part193 (éponge) : à ρ_vac=0,5 avec s=2 (raideurs gradient fixées) G_eff est encore ×2,8 car le potentiel V = μ_a²ρ_vac²(...) n'est pas rescalé (masse effective de la porteuse μ_a·ρ_vac).
Modèle : E = ρ_vac²ρ′²[WR(∇ρ′)² + WA ρ′²|∇n|²] + E4 + WV·μ_a²ρ_vac²(...) avec WR=WA=WV=ρ_vac⁻² : toutes les énergies statiques, exprimées en ρ′, sont identiques à celles de ρ_vac=1 ; ρ_vac n'entre plus que par le terme cinétique ½|∂ₜφ|² = ½ρ_vac²|∂ₜZ′|² (donc vitesse des ondes).
Runs : μ_a=0,375 ρ_vac ∈ {0,7 ; 0,5} ; μ_a=0,05 ρ_vac ∈ {1 ; 0,5} ; éponge 16→24, MEANC=0, 96³, t=40.
## Prédictions [P]
- Statique : M, R_rms, ρ_min/ρ_vac, profil δρ/ρ_vac et G_eff identiques à ρ_vac=1 à ±5 % (si le relaxé a atteint l'équilibre à t=40). Référence μ_a=0,375 : part193 s*_r100 (M 788,6 ; R 6,27 ; G_eff(9) −0,00085) ; μ_a=0,05 : le run ρ_vac=1 de part194.
- Réfutation : écart > 15 % ⇒ le modèle dépend de ρ_vac par un autre canal (cinétique/friction, relaxation incomplète) : à investiguer avant d'interpréter.
- Si confirmé : G indépendant de la porteuse en statique par construction ; reste à voir la dynamique (vitesse de la houle ∝ ρ_vac⁻¹ dans ce Lagrangien, signe opposé à C₀ ∝ Ã : à corriger en rescalant le cinétique).
