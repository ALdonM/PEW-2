# NOTE — session 34 (11–13/11/2026) — part184 : parité de Q et de J, charge de phase localisée, jauge U(1) sur Z2, paires (Q, J)

**Statut global : AUDITÉ le 13/11 par un agent isolé** (recalcul depuis les journaux, relance de `parite_phase.py A`, vérifications de `jauge.py`). Formulations **corrigées par l'audit**. Rien n'est publié ; rien n'est identifié à une particule (option (D)).
Code et journaux : dépôt ALdonM/PEW-2, `numerical_minimization/protect3d_Z/{parite_phase.py, jauge.py, angflux.py, snapdiag.py}` ; `resultats/part184/**`.

## 0. Pourquoi (questions de M. Aldon)
Image de M. Aldon : photon = paquet d'ondes en ressort (hélicité, trajectoire rectiligne) ; particule massive chargée = paquet bouclé sur un tore ; masse = bouclage ; signe de charge = trajectoire + chiralité ; spin = hélicité. Questions : (i) signe de la charge = sens poloïdal ou toroïdal ? (ii) l'annihilation e⁺e⁻ → 2 photons-ressort exige-t-elle une rotation propre ω opposée ? Tests dans le modèle Z, sans identifier le nœud à l'électron.

## 1. Définitions
- **Q** = charge de Hopf = degré de ζ = Z/|Z| ; pour un nœud axisymétrique Q = m·n. Statique, topologique, impaire par parité.
- **J** = charge de phase ∫Im(Z̄ ∂ₜZ) (en jauge : Z2 seul). Noether, conservée ; nulle pour une configuration figée. **Pas l'hélicité spatiale** (celle-ci est dans Q) ; hélicité usuelle = torsion (Q) × sens d'avancée (J).
- Le sens poloïdal seul ou toroïdal seul ne sont pas des charges (chacun s'inverse quand on retourne l'objet) ; seul leur produit est intrinsèque.

## 2. Résultats
**A1 Parité [A].** Hopfion N=64, R=6 : P (x→−x) change Q (−1→+1, degré exact) ; C (Z→Z*) non ; énergie identique (1144,689) ; J invariante sous P, change de signe sous C. Q pseudo-scalaire. Dégénérescence de (±Q,±J) vraie **pour un nœud isolé**, sans conséquence sur l'interaction d'une paire. Z1→Z1* (det −1) pas une symétrie en général : 0 % sur hopfion pur, ~0,9 % sur hopfion + bruit 5 %, 0,03–3,6 % sous SO(4) quelconque. Donc « P + réflexion interne » ne restaure pas Q ; h·χ change de signe sous P (≠ charge électrique) ; J seul est invariant sous P mais pas localisé dans Z (A2).

**A2 Charge de phase sans masse : elle s'échappe [B].** Rotation locale ω=0,2 (rayon 8), conservatif. **N=80 (boîte 40)** : J total conservé (308,296) ; J dans r<6 : 103,9 / 25,0 / 8,3 / −0,4 à t = 0 / 6 / 10 / 14 ; r<9 : 259,9 / 49 / 16 / 1,3 ; Q≈−0,98. **N=48 (boîte 24)** : la charge recircule dès t≈10 (J9=135 à t=14) : seule la boîte 40 est concluante.

**A3 Masse sur Z2, sans jauge [C].** 2ν²|Z2|² (ν=0,12), ω=0,08, N=48 : Q=−0,977 constant ; charge dans r<11 : 45,07 (t=20), 41,65 (t=30) sur 47,11. **Réserve audit** : r<11 couvre presque toute la boîte de 24, T=30 dépasse le temps de traversée ; r<6 passe de 24,7 à 13,2 en oscillant : localisation **non établie**, seulement compatible avec un gap. ν=0,375 : nœud défait à la relaxation (cause non établie).

**A4 Jauge U(1) sur Z2 seul [B construction, C dynamique].** Vide (1,0) invariant (photon sans masse). Faddeev jaugé Ω_ij = n·(D_i n × D_j n) + (n3−1)F_ij. Vérifié : invariance de jauge exacte ; fermeture de Ω en continu (signe opposé : échec) ; CS=2 (Noether) ; accélérations cohérentes ; aucune erreur de code. **Limites** : fermeture sur réseau non testée ; résidu de Gauss conservé par construction, fond uniforme −CS·q̄ non annulé. g²=0,005 : stable jusqu'à t=20, quasi découplé (Bmax≈0,04). **g²=0,05 : Q→0 entre t=4 et 10, déroulement physique** (relancé par l'audit à dt=0,005, énergie conservée) ; seul le saut d'énergie ultérieur est numérique.

**A5 Paires, séparation 12 (N=80).** Q opposées, J opposées vs identiques à t=14 : Etot 1551,8/1553,3, Eext 344,8/345,9 (0,1–0,3 %), **Qabs 0,085/0,082 (écart 0,8–3,5 % selon t, ±0,003)**. Q identiques (J opp./id., dt=0,01, ν=0,08) : Q −1,95 → 0 entre t=5 et 10, indépendant de dt, ν, J ; **cause non établie** (recouvrement initial, produit de quaternions non commutatif, réseau, ou physique) : **non interprétable**. Etot/Eext des journaux ≠ énergie conservée (masse et jauge exclues).

**A6 Paires, séparation 20 (N=96).** *Q identiques* (J opp., J id., sans J, t=14) : Q −1,955 → −1,96 conservé ; |x| finaux 10,81–10,96 (écart ≤0,15) ; Z2 à t=14 : 3534/3307/3217 (+10 %), Eext 30,6/30,4/28,8 → « J ne change rien » trop fort. *Q opposées* (t=24) : Qabs final 0,088 (J opp.) / 0,081 (J id.) / 0,080 (sans J) ; positions ±1,5–1,7 ; **J opposées retarde l'annihilation d'environ 10 %** (Qabs 1,132 vs 1,041 à t=18 ; 0,505 vs 0,446 à t=20) ; écart disparu à t=24 ; J identiques et sans J indiscernables. La charge de phase 47,25 par nœud (J id.) est conséquence de la symétrie miroir exacte ; pour J opp., qL passe de +47 à −100 : flux à travers x=0 où les nœuds fusionnent, pas un transport nœud à nœud.

## 3. Conclusion autorisée
Dans le modèle Z avec masse sur Z2 (ν=0,12), jauge faible (g²=0,005), ω=0,08 (Ekin(0) ≈ 0,25–0,28 % de Etot), orientation relative unique, séparation 20, T=24 : **le sens de J ne modifie ni la répulsion des Q identiques (positions à 0,15 près) ni l'issue de l'annihilation des Q opposées (Qabs final ±0,008) ; il la retarde d'environ 10 % à t=18–22 quand les J sont opposées**. Le critère « rotation opposée nécessaire à l'annihilation » n'est **pas confirmé**, et **pas testé tel quel** : J est une rotation interne U(1), pas un moment cinétique spatial ; la charge n'est pas liée au nœud (A2, A3) ; g² trop faible pour une force de Coulomb. Runs déterministes (pas de graine). Suggestion non prouvée : une charge de phase ne se localise que si ses excitations ont un gap.

## 4. Test « deux photons dos à dos » — EN COURS
- `angflux.py` sur h033_opp_pi_x (t=10–13,4) : deux lobes opposés le long de x (28–35 % de l'énergie de couronne vs 12,5 % isotrope), mais axe imposé par la géométrie de départ, Qabs≈1,98 (pas annihilé) → **non retenu**.
- Ondes de Z : mode d'amplitude ρ massif (m≈2μ≈0,75, approximatif) ; modes angulaires sur S³ sans masse au niveau linéaire. Une simulation classique ne peut pas montrer qu'un quantum est un photon (E=ħω, comptage) : elle teste seulement deux paquets opposés, énergies égales, vitesse c, sans dispersion.
- **Run GitHub 43432102200 (part184d)** : paires Q opposées, séparation 20, N=96, T=30, avec (G1) et sans (G3) rotation ; `sect16` (énergie par secteur, couronne 12<r<22) et `_proj.npy` dans `resultats/part184/gh_sep20_T30_lobes/`. À t=0 les lobes sont ceux des nœuds ; **critère : excédent après fusion**, lobes opposés d'énergies égales, direction non imposée.

## 5. Autres résultats (non audités ici)
part183 (h05snap, t≤40) : Q≈−1,97 ; seul `same_pi_z` montre deux pics égaux (g2/g1=1,00) qui se rapprochent (σ=2 : 7,5→5,5 ; σ=3 : 6,5→4,5), dépend du lissage ; autres : second pic instable. **Aucun signal retenu.** part182 h033 terminé, non comparé à h=0,5. Article 1 v2 : brouillon `PEW2_ARTICLE_1_v2_brouillon_20261110.tex` non audité.

## 6. À faire (issus de l'audit)
1. Journaliser l'énergie conservée complète ; relancer à dt=0,01 et 0,005.
2. ω plus grand (0,2–0,4) avec gap plus grand, g² 0,02–0,03, rotation spatiale à la place de J, plusieurs orientations relatives et paramètres d'impact.
3. Q identiques à séparation 12 : raffinement h/2 et relaxation de la paire préparée.
4. Test réseau de la fermeture du terme de Faddeev jaugé ; vrai fond neutralisant ; Gauss sur un cas non trivial.
5. Dictionnaire : Q, J, jauge U(1) sur Z2, ν, g² à saisir.
