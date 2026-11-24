# part206 — viriel exact du reseau (24/11/2026), script virial_reseau206.py
Energie L1 sans jauge avec pas anisotrope ; changer hx a champ fixe = etirement exact du champ discret (sans interpolation ni couture).
Noeud relaxe L=32, nu=0, h=0.5 (M_rel = 880.9) :
- -dE/dln hx = +54.14 (6.15 % de M) ; en retirant le vide (dE_vide/dlnhx = -4.61) : viriel du noeud S_x = +49.5 (5.6 %) = valeur de la formule par morceaux (49.5) -> coherence.
- axe z : 13.45 -> S_z = 8.8 (idem profil cumule).
=> le noeud relaxe viole le viriel du reseau (Derrick) de ~5.6 % en x ; plateau a grande boite (nu=0.12, h=0.5) : ~2.4 %.
Prediction de boost avec ce viriel : P/v = gam (M_rel + S_x) = 935.1 ; boost mesure : 949.6 (ecart 1.5 %, contre 0.16 % sur champ compact) :
les donnees boostees du noeud sont construites par interpolation cubique (non exactes), cet ecart reste a elucider.
