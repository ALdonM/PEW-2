# part198 — partie « espace » : vitesse des ondes libres ralentie davantage dans le nœud (KT(x) tabulé)
Idée : statique inchangée (même nœud, même G_eff, K0 de part196b réutilisé), seul le terme cinétique change : KT(x)=ρ_vac⁻⁴·ρ′(x)^(−2p) (carte fixe issue du nœud). Vitesse des ondes libres c=ρ′^(1+p) : p=0 → part196b (partie temps seule) ; p=1 → c=ρ′², ce qu'il faudrait pour un retard double (γ=1) si l'horloge locale suit ρ′.
Prédictions pré-enregistrées (k=3, nœud μ_a=0,375, ρ_vac=1) : retard de groupe ≈ ∫(ρ′^−(1+p) −1)dx : au moins ~1,5× (p=0,5) et ~2× (p=1) le retard de part196b pour b≥5, à ±20 %. Référence porteuse seule : p196b_r100_Vw (KT uniforme, inchangé).
ATTENTION : ce test ne fixe que la cinématique des ondes ; γ n'est défini qu'en comparant à l'horloge/chute libre d'une particule test (nœud massif) : étape suivante.
Analyse : python analyse_part198.py ../../resultats/part198
