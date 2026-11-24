# part206 — viriel exact du reseau (24/11/2026), script virial_reseau206.py
Energie L1 sans jauge avec pas anisotrope ; changer hx a champ fixe = etirement exact du champ discret (sans interpolation ni couture).
Noeud relaxe L=32, nu=0, h=0.5 (M_rel = 880.9) :
- -dE/dln hx = +54.14 (6.15 % de M) ; en retirant le vide (dE_vide/dlnhx = -4.61) : viriel du noeud S_x = +49.5 (5.6 %) = valeur de la formule par morceaux (49.5) -> coherence.
- axe z : 13.45 -> S_z = 8.8 (idem profil cumule).
=> le noeud relaxe viole le viriel du reseau (Derrick) de ~5.6 % en x ; plateau a grande boite (nu=0.12, h=0.5) : ~2.4 %.
Prediction de boost avec ce viriel : P/v = gam (M_rel + S_x) = 935.1 ; boost mesure : 949.6 (ecart 1.5 %, contre 0.16 % sur champ compact) :
les donnees boostees du noeud sont construites par interpolation cubique (non exactes), cet ecart reste a elucider.

## part207 (nu=0.12, sans jauge, relaxations convergees) : viriel du reseau S_x vs pas h
L=32 : h=0.5 -> 29.3 (3.1 %) ; h=0.4 -> 20.1 (2.13 %) ; h=0.32 -> 14.7 (1.55 %). Ajustement a+b h^2 sur h=0.5 et 0.32 : a=4.6 (0.5 %), b=99 ; predit 20.4 a h=0.4 (mesure 20.1).
L=40, h=0.4 : 13.75 (1.46 %) -> effet de boite ~ -6 a -7 par rapport a L=32.
=> le defaut de viriel decroit comme h^2 et extrapole a ~0 (0.5 % = niveau gam-1) : effet de maille, pas un defaut du modele.
Axe z : 8.8 (h=0.5, nu=0) ; 7.9 (L=32 h=0.4) ; 4.7 (L=32 h=0.32).

## part208 : covariance AVEC JAUGE (test_covariance_gauge208.py, sortie covariance_gauge208_output.txt)
Champ statique compact quelconque (phi + liens theta_y, theta_z), jauge A_x=0 -> boost en jauge temporelle (theta_dot=-v d_x theta, phi_dot=-v D_x phi). nu=0.12, g2=0.02, v=0.1.
Erreur P/P_pred-1, KF=1 : -1.24e-2 (h=0.5), -7.96e-3 (0.4), -5.11e-3 (0.32) = 4.96..4.99 h^2 -> 0 en h->0. Erreur H : -8.0e-5, -5.2e-5, -3.3e-5 (~h^2).
Temoin KF=0 : -3.5e-2, -3.1e-2, -2.8e-2 (ne tend pas vers 0 : non covariant).
Residu de Gauss des donnees boostees : densite constante ~0.13 (champ hors couche, Ampere non verifie ; sans effet sur P car A_x=0).
=> terme de Faddeev temporel jauge + champ E + couplage Z2 : covariants a O(h^2) (1.2 % a h=0.5, 0.8 % a h=0.4).
NON teste : noeud relaxe jauge (Gauss en regime relaxe), conservation de la charge de Hopf, cout dynamique (acc N=64 : 2.9 s local ; N=96 ~4 s/pas en CI).
