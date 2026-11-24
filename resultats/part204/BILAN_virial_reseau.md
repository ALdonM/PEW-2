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
