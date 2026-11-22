# part200 — terme de Faddeev temporel (inertie quartique) : refaire part199 et vérifier part196b (ρ_vac=1)
Lagrangien L★★ (ν=0, g=0, CP=1, ρ_vac=1) : L = ½KT|∂ₜφ|² + ½KF·cF·Σᵢ(n·∂ₜn×∂ᵢn)² − ½E ; KT=KF=1. Intégration : saut-de-mouton, force dépendant de v évaluée avec la demi-vitesse (erreur ∝ dt ; test 48³ : dérive d'énergie +0,3 % sur T=3 à dt=0,005, nulle à KF=0).
Jobs : part199 refait (f_g0, f_gp10, f_gp05 : g=0, 0,01, 0,005 ; T=18) ; part196b refait (w_Kw ε=0,05, w_Kb ε=0,025, k=3, T=20, départ = nœud seul de part196b ; la porteuse seule Vw de part196b reste valable : le terme est nul dans la porteuse).
Prédictions pré-enregistrées :
 1. Chute : inertie ≈ E_tot (≈ 788) au lieu de 178 → a/g ≈ 4·E2/E_tot ≈ 1,2 (E2≈267) ; plage [0,7 ; 1,8] ; a ∝ g ; sens vers ρ′ décroissant ; témoin g=0 immobile.
 2. Équivalence : a/g proche de 1 (temps seul, γ=0) ou 0,5 (γ=1) → on lit alors γ = g/|a| − 1 ; une valeur entre les deux ne conclut pas.
 3. Retard d'onde (k=3) : plus grand qu'en part196b (inertie du nœud plus grande) : au moins +10 % pour b<5 ; hors nœud (b≥9) inchangé à ±15 %. ε=0,05 vs 0,025 : linéaire à 10 %.
Fausse si : a/g hors [0,7 ; 1,8] ; ou retard de b<5 inchangé (±5 %).
Limites : pente imposée (pas un gros nœud) ; une réalisation ; schéma à erreur ∝ dt ; énergie = Ekm + E2+E4+EV (Ek non corrigé dans Etot).
Analyse : python analyse_part200.py ../../resultats/part200
