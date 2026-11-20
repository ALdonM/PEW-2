# part184ze : couche absorbante sphérique, test de réflexion (20/11/2026)

Statut : [D] outil et test linéaire seulement, aucun résultat physique sur les paires.

## Option ajoutée à jauge.py (mode pair)
Variables d'environnement : SPM (amortissement max sur v), SPG (sur w, jauge), SPRS (rayon de début, défaut 0,75 x demi-boîte).
Profil quadratique en R, v et w multipliés par exp(-sigma dt) à chaque pas. Défaut 0 : comportement inchangé.

## Test (sponge_test.py, vide zeta=(1,0), nu=0,3, g2=0,02, N=48, h=0,5, demi-boîte 12, absorbeur de R=6 à 12, impulsion gaussienne s=1,5)
Ein/E0 = énergie (proxy 2 x cinétique) dans R < 5 après le passage de l'onde.
- Matière, cube périodique : 8 % revient à t=25 (aller-retour par le bord). Avec SPM=1 : 0,4 % au plus.
- Jauge (impulsion transverse de theta), cube périodique : 20 % revient à t=25. Avec SPG=1 : 0,4 % au plus.
- Énergie totale (moins le vide) avec SPM=1 / SPG=1 : tombe à moins de 0,3 % de E0 à t=35.
Limites : proxy d'énergie, un seul profil et une seule longueur d'onde, petite boîte, sans noeuds ni charge. Absorber v ou w seul n'a pas été testé avec matière chargée (loi de Gauss).
