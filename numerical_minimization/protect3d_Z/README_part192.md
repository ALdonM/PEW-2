# part192 — rigidité du milieu fixe (étape 2 de part191)
Contexte (M. Aldon, 20/11) : G appartient au milieu/écume, pas à l'amplitude de la porteuse. part191 : G_eff dérive de ×1,45–2,4 quand ρ_vac passe de 1 à 0,5 (M ∝ ρ_vac^0,6–0,75).
Modèle : E2 = ρ'^2·[WR·(∇ρ')² + WA·(partie angulaire)]·ρ_vac², le reste comme part191 (CP=1, MEANC=1, μ_a=0,05, boîte 96³ L=48, friction 0,5, t=40). WA = ρ_vac⁻² : la raideur du nœud est fixée en absolu (le nœud ne voit plus ρ_vac). WR = ρ_vac⁻ˢ : raideur radiale de la porteuse (s=0 : naturelle, plus ρ_vac est faible plus la porteuse est molle ; s=2 : fixe aussi). ρ_vac ∈ {0,7 ; 0,5}. Référence ρ_vac=1 : part191 m005_r100.
## Prédictions écrites avant les résultats [P]
- (K) masse et taille du nœud indépendantes de ρ_vac à ±15 % de la référence (M≈571, R_rms≈9,9) ; sinon le terme de potentiel (non rescalé, μ_a²ρ_vac²) ou la mesure du nœud dans la boîte est en cause.
- (C, s=0) δρ/ρ_vac ∝ ρ_vac⁻² au même nœud : G_eff ×~2 à 0,7, ×~4 à 0,5 (écrêté par le mur en 1/ρ² : ρ_min/ρ_vac).
- (C, s=1) G_eff ∝ ρ_vac⁻¹ ; (C, s=2) G_eff constant à ±25 % (reste : masse effective μ_a ρ_vac du carrier plus petite).
- Lecture : une porteuse plus faible et plus molle se déforme plus pour la même masse ⇒ G plus grand, sens « G = fluidité du milieu » ; opposé à G ∝ C̃₀ du glossaire.
Réfutation : s=0 avec G_eff constant à ±15 % ; ou nœud instable / charge non conservée.
Analyse : analyse_part192.py (resultats/part192).
