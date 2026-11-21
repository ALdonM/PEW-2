# part196 — retard de phase d'une onde libre de la porteuse, milieu fixe
(Vocabulaire : pas de vide dans PEW ; « V » = porteuse seule à l'amplitude de fond ρ_vac, sans onde bouclée ; « K » = avec un nœud.)
Question : une onde libre de la porteuse (paquet de phase k=3, sans masse) est-elle retardée en traversant le creux d'un nœud, dans le modèle à milieu fixe (WA=WR=WV=ρ_vac⁻², KT=ρ_vac⁻⁴, μ_a=0,05, éponge sphérique 16→24) ? Comment le retard dépend-il de ρ_vac ?
Runs : ρ_vac=1 (t=20) et 0,5 (t=36) : K0 (nœud seul), Kw (nœud + onde, ε=0,05), Vw (porteuse + onde, sans nœud) ; ρ_vac=0,5 : Kb, Vb (ε=0,025, linéarité). Nœud de départ = champ final part194 / ρ_vac. Paquet : x₀=−8, σ=3, vitesse de groupe réseau ≈ 0,73·ρ_vac ; centre vers x≈+6 en fin de calcul. Vérifié sur petite boîte (N=48, ρ_vac=0,5) : le paquet avance à ≈0,345 (attendu ≈0,365), avec une composante résiduelle en arrière (≈ 35 % en intensité de x₀) hors de la fenêtre de mesure.
Mesure : démodulation complexe à k=3 sur x ∈ [2,5 ; 14] : retard(b) = arg(z_K/z_V)/k, z = Σ χ·e^{−ikx}, χ = angle de la rotation de phase (K : par rapport au nœud seul K0 ; V : par rapport à la porteuse au repos), par bins de distance transverse b = 3–5, 5–7, 7–9, 9–11 (positif = retard). Comparaison avec ∫(1−ρ/ρ_vac)dx le long de la ligne de visée, calculée sur le nœud seul K0 (vitesse locale c ∝ ρ).
## Prédictions [P]
- (R1) retard > 0 pour tous les b, décroissant avec b.
- (R2) retard ≈ ∫(1−ρ′)dx (optique géométrique, c_local = c₀ρ′) à ±30 % ; pour k=3 (λ≈2), régime géométrique, mais le noeud μ_a=0,05 est gros (R≈10) : correction attendue.
- (R3) indépendance de ρ_vac : le retard (en unité de longueur, retard/λ) est le même à ρ_vac=1 et 0,5 à ±15 % (la forme statique est identique, et tout ralentit en bloc).
- (R4) linéarité ε=0,05/0,025 : retards compatibles à ≲20 %.
Réfutation : retard nul dans le bruit (houle insensible au nœud) ; ou retard dépendant de ρ_vac à > 30 % (le milieu fixé ne découple pas la houle).
Limites : une réalisation, un nœud, périodique non (éponge sphérique), partie temps de la métrique seulement (pas de partie espace), spin 0 (pas un photon).
