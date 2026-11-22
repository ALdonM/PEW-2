# part199 — chute libre d'un nœud test dans une pente de porteuse imposée (substitut dynamique de la particule test)
Modèle part195 (rigidités fixées, ρ_vac=1, μ_a=0,375). Nœud relaxé (p193_s2_r100) × pente ρ_t(x)=1+g·x ; potentiel (minimum) et éponge rappellent vers ρ_t(x) (env GRAD). T=32, 96³, sans friction. Jobs : g=0 (témoin), +0,01, +0,005, −0,01.
Mesure : x4(t) = centre de la densité de Faddeev (nulle dans la porteuse), ajustement x=x0+½ a t² (après t≈6, transitoire de relaxation).
Lecture de γ : ondes libres c=ρ′=1+g x. Si a_chute = −∇Φ_N et c = 1+(1+γ)Φ_N, alors γ = g/|a| − 1 (γ=0 ⇔ |a|=g ; γ=1 ⇔ |a|=g/2).
Prédiction pré-enregistrée (énergétique) : a ≈ 2·E2/E_tot·g ≈ 0,57 g (E2=267 sur 788 → 2·0,34 = 0,68 g si E_tot=788) → |a| compris entre 0,5 g et 0,8 g ; a ∝ g (g=0,005 : a divisé par 2, ±20 %) ; sens vers ρ′ décroissant (g=+0,01 : x décroît) ; témoin g=0 : dérive < 0,3 sur T=32.
Fausse si : a ∝ g rejeté, ou sens opposé, ou |a| hors [0,35 g ; 1,0 g].
Limites : seule la cinématique d'un nœud ; ρ_t imposée (pas le champ d'un vrai gros nœud) ; potentiel V recentré sur ρ_t (hypothèse du substitut) ; énergies diagnostiques polluées par la discontinuité périodique (zone absorbée).
Analyse : python analyse_part199.py ../../resultats/part199
