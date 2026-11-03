# part168 — dipôle de courant de phase D du nœud de Hopf du modèle Z (PEW)

Suite de part167 (référentiel privilégié). Un soliton en mouvement v dans le vide qui coule voit son énergie décalée de
δE = 2 γω v·D (+ O(ω²)), avec D = ∫ Im(Ψ†∇Ψ) d³x. Sur réseau périodique, avec Z = e^{ik·x}Ψ, l'identité exacte dE/dk = 2D
donne D au minimum. La valeur trouvée en part167 (−549) était un courant persistant non relaxé (D = L·Φ).

Runs (64³ et 96³, h = 0,5, e₀ = 0,125, μ = 0,125, L-BFGS float64) : k = 0 depuis un hérisson neuf et depuis l'état relaxé de part165,
k_z = ±0,02 (contrôle : E(+k) − E(−k) = 4kD), et 96³ (effet de taille de boîte). Convergence à surveiller : Phi_std (le flux doit
devenir uniforme), gnorm, et D selon la taille de boîte. Résultats déposés automatiquement dans `resultats/part168/`.
