# MESSAGE DE REPRISE — 2026-11-13 (session 34, fin) — charge de Hopf Q, charge de phase J, jauge U(1) ; suite : lobes de rayonnement et critère de rotation

**Titre proposé pour la prochaine session :** « Session 35 — Que montre le rayonnement de l'annihilation Q/anti-Q ? (part184d) »

**Accès aux fichiers.** Ce message et la note associée sont dans le Project PEW (`claude/MESSAGE_REPRISE_20261113_s34_part184.md`, `claude/claude_NOTE_session_20261113_s34_part184_parite_charge_phase_jauge_paires.md`) et dans le dépôt ALdonM/PEW-2 (`resultats/part184/`). Si la session n’a pas accès au Project, les deux fichiers sont dans le dépôt, dossier `resultats/part184/`.

**À lire, dans cet ordre :** 1. Ce message. 2. La note de la session (fichier `claude_NOTE_session_20261113_s34_part184_parite_charge_phase_jauge_paires.md`, auditée le 13/11). 3. `DECISIONS_Article_1_v2_20261110.md` (référence unique = PEW2 Arts 1–4 ; option (D)). 4. `PEW_SOCLE_v2_2_fondations_PEW_C.md` §0, `PEW_Protocole_Validation.md`.

**Règles (inchangées).** Tutoiement. Challenger, ne pas flatter. Travail autonome autorisé. Rien sur GitHub ni Zenodo sans accord (sauf le dépôt ALdonM/PEW-2, où le push est autorisé). Audit isolé avant toute note (agent séparé, recalcul depuis les journaux). Dictionnaire à interroger avant de créer une variable. Réponses courtes (téléphone). Ne pas nommer « photon » ou « électron » un objet du modèle sans preuve (option (D)). Commits : lignes `Co-Authored-By` et `Claude-Session` de l'environnement.

## 1. Où en est-on (13/11)
- Vocabulaire fixé avec M. Aldon : **Q** = charge de Hopf = m·n (chiralité d'enroulement, statique, impaire par parité) ; **J** = charge de phase (rotation propre ω, conservée, T-impaire). Le sens poloïdal seul ou toroïdal seul n'est pas une charge. L'hélicité usuelle = torsion (Q) × sens d'avancée (J).
- Question de M. Aldon : l'annihilation en « deux photons-ressort » exige-t-elle ω opposé ? **Résultat (audité)** : à ω=0,08, g²=0,005, séparation 20, T=24, le sens de J ne change ni la répulsion des Q identiques (≤0,15 sur les positions) ni l'issue de l'annihilation des Q opposées (Qabs final 0,088/0,081/0,080) ; J opposées la retarde de ~10 % à t=18–22. **Critère non confirmé, mais pas testé tel quel** (J interne ≠ moment cinétique spatial ; g² trop faible ; ω petit ; charge non liée au nœud).
- Charge de phase : sans masse elle fuit du nœud (boîte 40 : J dans r<6 de 104 à ≈0 en 14) ; avec masse ν=0,12 elle reste compatible avec un confinement (non établi, boîte trop petite) ; avec jauge g²=0,005 le nœud est stable, g²=0,05 le déroule.
- Q identiques à séparation 12 : déroulement non interprétable ; à séparation 20 : conservés.

## 2. Calculs en cours (à relire en premier)
- **Run GitHub 43432102200 (part184d)** : paires Q opposées, séparation 20, N=96, T=30, avec/sans rotation ; résultats dans `resultats/part184/gh_sep20_T30_lobes/` (journaux `G*.log`, `*_proj.npy`, champ `sect16` = énergie par secteur de 22,5° dans la couronne 12<r<22). Test attendu : **excédent d'énergie après fusion en deux lobes opposés, énergies égales, direction non imposée par la géométrie de départ** ; vitesse de propagation ; dépendance en J. À faire d'abord : `git pull`, lire les journaux, ne rien conclure avant un audit isolé.
- Runs part182 (h033) et part183 (h05snap) terminés ; comparaison h=1/3 vs h=0,5 non faite ; aucun signal « deux pics » retenu (dépend du lissage, sauf `same_pi_z` : deux pics égaux qui se rapprochent).
- Artefacts GitHub non téléchargeables depuis la session (403) : tout diagnostic doit être calculé dans le job et commité en petit fichier.

## 3. Pièges rencontrés (ne pas répéter)
- Les tâches en arrière-plan sont **tuées quand la session reste inactive** : tout calcul de plus de ~10 min va sur GitHub Actions (dispatch par l'API REST : `gh api -X POST repos/ALdonM/PEW-2/actions/workflows/<f>/dispatches -f ref=main` ; `gh workflow run` utilise GraphQL, interdit).
- `jnp.abs(z)**2` sur un complexe a un gradient singulier en 0 (NaN quand Z2→0) : utiliser re²+im² (`sq` dans `jauge.py`).
- `Etot`/`Eext` des journaux de paires ne sont pas l'énergie conservée (masse et jauge exclues). Ne comparer qu'en paires appariées.
- Boîte périodique : la charge sans masse recircule (N=48) ; utiliser N≥80. ν≥0,375 défait le nœud à la relaxation ; g²≥0,05 le déroule.
- Produit de quaternions A·B ≠ B·A : une seule orientation relative testée.

## 4. Prochaines étapes (ligne directrice : Article 1 v2)
1. Lire part184d ; test des lobes ; audit isolé ; note.
2. Si le critère de rotation reste d'intérêt : ω et gap plus grands, g² ~0,02–0,03, moment cinétique spatial, plusieurs orientations/paramètres d'impact ; énergie conservée complète.
3. Article 1 v2 : audit par les trois filtres du brouillon ; Derrick refait avec ρ, μ ; choisir Z comme modèle de travail (relèvement) ; corriger Art 2 ([A]→[D] « sans dérivée croisée ») ; formulation « Q pseudo-scalaire (P + réflexion interne ne restaure pas Q) ».
4. Versé au Project : Arts 1, 2, 4 PEW2 et Art 25 ; **Article 3 (BVP2D v5, 97 Ko) pas encore versé** (verser en deux parties via `content`).
5. Administration : export JSON du dictionnaire ; saisir Q/J/ν/g²/jauge Z2 ; fusion D0256/D0241 ; Socle v2.3 ; index Drive ; archiver `analyse_part181.py`.

## 5. Décisions et image de M. Aldon (rappel)
- Une seule référence : PEW2 Arts 1–4 ; le reste est de l'historique. Nouvelle version de l'Article 1 en PEW2 (pas de projet PEW3 : nom déjà utilisé dans l'Art 4).
- Image de M. Aldon (ses mots, non validée par le calcul) : photon = paquet d'ondes en ressort, hélicité, trajectoire rectiligne ; particule massive chargée = paquet bouclé sur un tore, masse = bouclage, signe de charge = trajectoire + chiralité, spin = hélicité. Critère proposé : annihilation e⁺e⁻ → 2 photons-ressort rectilignes exige une rotation ω opposée. La simulation HTML « Multi Leptons v2.3.13 Surface Sphere » ne fixe aucun signe (H₀=H₁=1, H₂=32/219/3936 selon le lepton, ω global 0,5, jauge de chiralité marquée ad hoc).
- Question ouverte à lui poser : la « petite spirale » et l'« enroulement sur le tore » sont-ils deux torsions spatiales distinctes, ou la même vue deux fois ? Cela décide si h est indépendante de Q.
