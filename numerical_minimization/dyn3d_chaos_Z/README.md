# part166 — dynamique 3D d'un chaos d'ondes du modèle Z (PEW)

Question (M. Aldon, 03/11/2026) : un chaos d'ondes de l'Onde Mère, avec expansion, laisse-t-il des ondes bouclées (nœuds de Hopf) ?
Suite de part165 (relaxation L-BFGS, non concluante). Ici : vraie dynamique (équation d'onde), réseau périodique 64³, h = 0,5.

- `zlat.py` : énergie du modèle Z sur réseau (|∇Z|² + Faddeev(n) anti-damier + μ²(ϱ² + 1/ϱ² − 2)), charge de Hopf = degré de ζ.
- `dyn.py` : saut-de-mouton avec friction de Hubble (ère de rayonnement, a = √(t/t0)) ou constante ; reprise sur checkpoint.
- Workflow : `.github/workflows/pew-part166-dyn3d-chaos-Z.yml` (6 runs × 2 tranches, lancement manuel).

Paramètres : e₀ = 0,125, μ = 0,125, ξ₀ = 6, dt = 0,0025 (dt/5 pendant 0,2 au départ). Terme de Faddeev temporel omis [D].
Runs : rad10_s1, rad10_s2 (a : 1 → 2, H₀ = 0,05), rad40_s1 (a : 1 → 2, H₀ = 0,0125), fric02_s1 (γ = 0,02), cons_s1 (sans friction),
rad10_knot_s1 (un nœud |Q| = 1 inséré dans le chaos).
