# part184ze8 — t_ann en fonction de la charge relaxée q0 (z Jo, jauge, N=128, ω=0,19) [D]

Run 45006959662, 5 paires issues des étapes de relaxation s1..s5 (relaxation ω=0,19 non convergée : la charge croît de ~4 % par étape de 25 unités). Toutes les paires ont fini (60 unités, absorbeur + jauge transverse).

| étape | q0 | t_ann |
|---|---|---|
| s1 | 341,0 | 18,54 |
| s2 | 344,5 | 17,64 |
| s3 | 346,9 | 16,99 |
| s4 | 348,5 | 16,65 |
| s5 | 349,5 | 16,34 |

Points antérieurs (N=128, ω=0,19) : q0 336 → 20,25 (100 unités de relaxation) ; q0 350 → 16,03 (250 unités).

**Résultat [D]** : t_ann décroît de façon monotone et lisse avec q0 (pente ≈ −0,3 par unité de q0 entre 336 et 350). Pas de seuil net. Le « creux » de z Jo à ω=0,19 s'explique donc par la durée de relaxation (charge plus élevée → annihilation plus rapide), pas par un effet propre à ω. Écart résiduel à q0≈350 : 16,34 (s5) vs 16,03 (250 unités), ≈0,3, soit la dispersion due à l'historique de relaxation.

**Conséquence** : comparer des ω différents exige de comparer à q0 égal, pas à durée de relaxation égale.

**Hypothèse de dérive (inférence, non testée)** : la dérive de charge à ω=0,19 serait un mode mou. L'idée initiale « ω>ν=0,12 » est réfutée par ω=0,14 > ν qui converge (relaxation N=128 : charge 244→188, force 1,5e-5). Le seuil, s'il existe, est entre 0,14 et 0,19.
