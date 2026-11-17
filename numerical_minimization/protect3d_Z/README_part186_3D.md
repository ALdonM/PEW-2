# README part186 (3D) — houle / écume après contraction, modèle Z — STATUT : préparé, NON lancé

Question (M. Aldon, 17/11) : quand l'espace se contracte jusqu'à ce que les nœuds remplissent la boîte, une houle (oscillation cohérente de ρ à k=0) puis une écume (modes k≠0) apparaissent-elles au relâchement ? Le test 2D (part186, jouet) n'a pas tenu la charge ; on utilise donc le modèle Z 3D de protect3d_Z.

Fichiers : dyn186.py (= dyn.py + MODE=cycle, INIT=knots, spectre de δρ), gen_workflow_part186.py → .github/workflows/pew-part186-cycle-houle-ecume.yml.
Runs (64³, h=0,5, μ=0,375, e0=0,25, dt=0,0025, a(t) : 1 jusqu'à t=30 avec friction 0,5, contraction a:1→AMIN sur 40 avec friction 0,02, relâchement sur 3 sans friction, évolution libre jusqu'à t=130) :
- k2_a05 : 2 nœuds de même signe, AMIN=0,5 (test)
- k1_a05 : 1 nœud, AMIN=0,5 (témoin : un seul nœud)
- k2_a10 : 2 nœuds, AMIN=1 (témoin sans contraction : bruit numérique seul)

Critères écrits avant le run :
1. Prérequis : Q=±2 (±1 pour k1) et deg inchangé en fin de relaxation (t=30) et à t=70 (a minimal). Sinon, résultat négatif rapporté tel quel.
2. Houle : après t=73, rho_mean oscille de façon cohérente ; P0 (k=0) > Pk au début du relâchement ; fréquence mesurée, comparée à 2μ=0,75.
3. Écume : croissance de spec[i] dans des bandes précises, absente du témoin k2_a10 ; sinon c'est du bruit numérique.
4. Charge : Q(t), deg(t), deg_bloc(t) pendant la contraction et le relâchement : conservée (cas A) ou perdue (cas B), avec cos_min pour séparer saut de réseau et vraie perte.
5. Fusion : clusters (énergie, charge) à a minimal : 2 amas = empilement ; 1 amas = vortex unique.
Limites connues : une graine, 64³ (la boîte ne contient que ~3,7 diamètres de nœud), pas de friction de Hubble, terme de Faddeev temporel omis [D], a(t) imposé (pas de gravité dynamique).
Mise en place : copier dyn186.py et .github/workflows/... dans le dépôt, puis lancer le workflow à la main (Actions → PEW part186 → Run).
Vitesse locale mesurée : ~0,5 s/pas sur 2 cœurs, soit plusieurs heures par run : à lancer sur GitHub Actions.
