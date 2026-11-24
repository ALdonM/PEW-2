# part204b — test exact de l'outil de mesure et du Lagrangien discret (24/11/2026)

Identite (theorie covariante, champ statique quelconque a support compact, sans relaxation) : P = gam v (M_rel + Sxx), Sxx = -dE/dln(lambda_x),
M_rel = energie relative au vide uniforme (densite -1.406e-4, non nulle car d2 != 0).
Resultats (test_covariance204.py, sortie dans covariance_test_output.txt) :
- KF=1 : erreur P = -1.6e-3 (h=0.5), -1.0e-3 (0.4), -6.8e-4 (0.32) -> ~ h^2 ; H idem (a l'offset de vide pres).
- KF=0 : erreur -21 % (non covariant, temoin negatif correct).  Sans terme quartique (e0=1e3) : -1.3e-2 (h=0.5), -0.85e-2 (0.4), h^2.
=> l'outil de mesure et le Lagrangien KF=1 (sans jauge) sont covariants a 0.2 % pres a h=0.5.

CORRECTION de part202/203 : M doit etre relative au vide (offset -1.4e-4*L^3 : -4.6 a L=32, -24.7 a L=56). La "derive de E avec L" etait presque entierement cet offset.
P/(M_rel v), KF=1, h=0.5 (covariant : 1.005)
 nu=0    : L=32 1.0779 ; 40 1.0626 ; 48 1.0522 ; 56 1.0484      (M_rel 880.9, 871.0, 866.2, 863.8)
 nu=0.12 : L=32 1.0585 ; 40 1.0507 ; 48 1.0490                    (M_rel 934.0, 932.1, 931.6)
Excès résiduel ~ +4.5 a +5 % (vs 0.5 % attendu), qui diminue avec L mais ne semble pas tendre vers 0.
