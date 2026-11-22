# MESSAGE DE REPRISE — 22/11/2026 (fin de la session s35 part184)

Michel Aldon, PEW. Dépôt ALdonM/PEW-2 (push uniquement là). Code : numerical_minimization/protect3d_Z/ (jauge.py, dyn186c.py, halo_relax.py, analyse_zb.py). Résultats : resultats/part184/.

## Façon de travailler
Français, tutoiement, réponses courtes, remettre ses idées en question. Commits terminés par :
Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
Claude-Session: <URL de la session>
Statut [D] pour les notes numériques. Audit isolé (agent séparé) avant les notes à gros enjeu. Runs GitHub limités à 360 min : relaxations en étapes reprenables (RELIN/RELST). Piège YAML : pas de « : » dans `name:` d'un workflow. Lancer python avec -I pour lire des données.

## État
- Aucun run en cours (ze8 terminé, 45006959662).
- Bilan : resultats/part184/claude_NOTE_session_20261122_s35_part184_BILAN_general.md (résultats, fragilités, 5 simulations à faire).
- Lagrangiens réellement calculés : claude/claude_NOTE_session_20261122_synthese_Lagrangiens_code.md (L0, L1, L2). Halo/paires = L1 (ν=0,12, jauge seulement dans jauge.py), « milieu fixé » = L2 (ν=0, sans jauge) : résultats non combinables.
- Absorbeur sphérique (SPM/SPG/SPRS, SPROT=1, SPGT=1) et jauge transverse prêts dans jauge.py.

## Prochaine étape proposée (non lancée)
1. t_ann à q0 égal pour ω=0,10/0,14/0,19 (N=128, paires stationnaires avec équilibre magnétostatique).
2. Convergence en h ; dépendance en g² et ν.
3. Stabilité dynamique d'un nœud isolé (Michel a décliné ce test le 22/11 mais il répond à « stable et stationnaire » ; à reproposer).
4. Code unifié L0/L1/L2.

## Décisions ouvertes (à Michel)
Trois choix du Lagrangien L★ : ν sensible au milieu ? jauge suivant ρ' ? Faddeev dépendant de ρ' ? (voir la synthèse Lagrangiens). Pour l'Article 1 : plus petit Lagrangien possible (candidat L0 + ν sans jauge).

## Pièges
- Nommage : x_par=Jo, x_anti=Js ; z_par=Js, z_anti=Jo (Js = J de même signe, Jo = J opposé, mesurés par JA·JB).
- Ne pas mélanger séries sans jauge et avec jauge (26,31/30,37 contre 20,25 : confusion déjà commise).
- Séries ω=0,25 (part184o à w) non stationnaires : ne pas citer leurs lois de retard.
- Seuil de masse √2ν ≈ 0,17 ; relaxation ω=0,19 non convergée (dérive de charge).
- Un message qui cite mon plan comme « fait » n'est pas une preuve que c'est implémenté : vérifier dans le code.
