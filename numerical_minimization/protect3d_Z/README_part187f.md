# part187f — paires de noeuds a coeurs figes, relaxees a nu=0,12, L=64 (z, x, diagonale)
But : E_int(d) = E(paire relaxee) - 2 E(seul) sans le biais de l'ansatz non relaxe de part184w. Noeud isole : relaxe L64 N128 nu=0,12 (run 44638347471, E=931,32 au-dessus du vide). Une paire libre noeud/anti-noeud s'annihile : on fige les boules r<5 autour des deux centres (valeur du noeud isole) et on relaxe tout le reste (halo_relax.py --free). Le noeud seul est deja un minimum : son energie ne change pas quand on fige son coeur.
Matrice : charges opposees (miroir z->-z pour z, x->-x pour x et diag) et egales ; directions z (axe du noeud), x (perpendiculaire), diag (1,1,0) ; d ~ 12, 18, 24 (diag : 11, 18, 24) ; 18 jobs.
Predictions pre-enregistrees (avant les resultats) :
 1. z, charges opposees : decroissance exponentielle de taux ~0,34 : E_int(18)/E_int(12) = 0,13 +/- 0,06 (part184w non relaxe : 0,11) ; E_int(24)/E_int(18) < 0,3.
 2. x, charges opposees : decroissance plus lente que sur z : E_int(24)/E_int(18) entre 0,3 et 0,6 (part184w : 0,40). Pas de loi en d^-3 franche : pente log-log entre 18 et 24 dans [-5,-2].
 3. Test discriminant dipole/pair : le terme de courant de phase (impair par miroir) change de signe entre charges egales et opposees, les termes e_n, e_rho, eV (pairs) non. Si la queue de E_int est dominee par le courant, E_int(egales) ~ -E_int(opposees) a d=24 ; si elle est dominee par les termes pairs, E_int(egales) ~ E_int(opposees). Prediction : termes pairs, soit |E_int(egales) - E_int(opposees)| < 30 % de |E_int(opposees)| a d=24 sur z et x.
 4. E_int < 0 (attractif) a d=12 sur z pour les charges opposees, a mieux de |E_int| > 30 (part184w : -130 non relaxe ; la relaxation doit le reduire).
Fausse si : 1 donne un rapport > 0,3 (queue lente sur z) ; ou 3 donne E_int(egales) de signe oppose avec la meme amplitude (queue de courant dominante).
Reserves : coeurs figes (r<5) : images periodiques a 64-d ; relaxation tronquee par --tmax (|g| a lire dans le journal).
Analyse : paire_*.json (E_int, Q) et relaxe_*.log.
