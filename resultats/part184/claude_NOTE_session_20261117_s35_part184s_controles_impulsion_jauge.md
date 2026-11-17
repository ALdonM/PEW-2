# Note part184s : contrôles (impulsion axiale annulée, g²=0, un seul nœud) sur le retard lié au spin

Session s35, 2026-11-17. Audit isolé (agent séparé, recalcul depuis les journaux) avant écriture ; ses corrections sont intégrées. Modèle Z classique, aucune identification à une particule (discipline D). Données : `resultats/part184/gh_part184s/` (S_*), comparées à part184o/p/r. Nœud relaxé N=96 (RELT=120), NB=96, SEPC=20, ν=0,12, T=45, une réalisation par point. Corrige la note part184r. Les comparaisons n'ont de sens qu'à orientation et g² identiques (références ω=0 différentes : x 22,88 / z 19,90 à g²=0,02 ; x 15,07 / z 20,91 à g²=0).

## 1. Chiffres (t_ann = Qabs<0,5 interpolé ; incertitude ~0,1 à 0,3 ; tous confirmés par l'audit, Econs conservé à ≤0,03)

| Run | g² | spins (Lz t=0) | Pz initial | t_ann | écart à ω=0 |
|---|---|---|---|---|---|
| z ω=0 | 0,02 | | 0 | 19,90 | |
| z parallèles | 0,02 | −647 | ±27 l'un vers l'autre | 17,58 | −2,3 |
| z parallèles, Pz annulé | 0,02 | −591 | 0 | 26,22 | +6,3 |
| z antiparallèles | 0,02 | +1 | 29,3 / 26,0 (même sens) | 40,36 | +20,5 |
| z antiparallèles, Pz annulé | 0,02 | +7 | 0 | 40,32 | +20,4 |
| z ω=0 | 0 | | | 20,91 | |
| z parallèles | 0 | −647 | | 18,60 | −2,3 |
| z antiparallèles | 0 | +1 | | 28,84 | +7,9 |
| x parallèles, Pz annulé | 0,02 | −606 | 0 | 29,73 | +6,85 (sans annulation +9,37) |
| x un seul nœud tournant | 0 | −330 | | 15,42 | +0,35 (référence 15,07) |
| x antiparallèles | 0,03 | −4 | | 25,53 | +0,12 (référence 25,41) |

## 2. Établi

- **L'« accélération » des spins parallèles sur l'axe (−2,3) était cinématique** : elle apparaît à g²=0 comme à 0,02, les nœuds se rapprochent dès t=5 sous l'effet des Pz initiaux opposés, et elle disparaît quand Pz est annulé (+6,3 de retard). Réserve : CANCEL_PZ modifie aussi J (−6,5 à −8 %), Ekin (−7 à −8 %) et la source de jauge ; pas de contrôle « annulé » à g²=0 ni pour les antiparallèles en x.
- **Les spins antiparallèles sur l'axe sont fortement retardés** (+20,4 à g²=0,02, indépendant de l'annulation de Pz), et déjà à g²=0 (+7,9) : il existe donc un effet **sans champ de jauge** (secteur scalaire). Le champ de jauge ajoute ~+12,5 dans cette configuration (≈60 %). La jauge n'est pas le seul vecteur.
- **En x, le retard des spins parallèles est surtout lié à la jauge** (+1,05 à g²=0 ; +9,4 à g²=0,02), dont ≈2,5 sur 9,4 disparaissent si l'on annule Pz (mêlé à la perte de J et d'Ekin du boost).
- **Pas d'attraction côte à côte** : antiparallèles en x à g²=0,03 : +0,12 ; à g²=0,02 : +0,27 ; g²=0 : +0,42 (valable pour g²≤0,03 ; à g²=0,1 les runs sont instables).
- **Un seul nœud tournant (g²=0) : +0,35**, dans l'incertitude du cas antiparallèle (+0,42). Le +1,05 des deux nœuds parallèles à g²=0 est un effet à deux corps non additif, de petite taille.
- Classement par Lz physique (et non par le nom sj, car le miroir x→−x retourne Lz et pas z→−z) : parallèles : +6,8 (x, annulé) et +6,3 (z, annulé), équivalents ; antiparallèles : ≈+0,1 à +0,3 en x contre +20,4 en z, très dépendant de l'orientation.

## 3. Retiré ou nuancé

- « L'hypothèse dipôle magnétique via la jauge » : non réfutée en bloc (la jauge contribue ≈60 % dans le cas z antiparallèle et la quasi-totalité en x), mais **non soutenue** : pas d'attraction des antiparallèles côte à côte, l'accélération sur l'axe était cinématique, et un effet existe sans jauge. Le signe et l'amplitude ne suivent pas une loi dipôle–dipôle simple.
- « Le retard est plus grand pour les rotations internes opposées dans les deux orientations » : invalide tel que formulé (le signe de J ne code pas le même spin selon l'orientation). Lecture par Lz uniquement, et comme hypothèse.
- L'anisotropie de base (x 15,07 / z 20,91 à g²=0 ; x 22,88 / z 19,90 à g²=0,02 : la jauge ajoute +7,8 en x et −1,0 en z) décrit deux configurations géométriques différentes (Z2 initial 3489 contre 3303), pas une propriété du nœud seul.
- En x la dépendance en g² du retard de base n'est pas monotone sur toute la plage (15,07 ; 19,92 ; 21,62 ; 22,88 ; 25,41 ; puis 8,40 à g²=0,1, instable).

## 4. Limites

Une réalisation par point ; images périodiques (boîte 48, séparation 20) ; seuil arbitraire sur Qabs ; le boost CANCEL_PZ n'est pas neutre ; X1 (part184o) vient d'une autre relaxation (Z2=3442) et ne doit pas être mélangé.

## 5. Suite

Contrôles manquants : annulation de Pz à g²=0 et pour les antiparallèles en x ; test à 45° de l'axe ; séparation plus grande ou nœuds tronqués contre les images ; réalisations multiples ; séparer le scalaire et la jauge en z antiparallèle (g²=0,01 et 0,015) ; mesurer la phase relative des deux nœuds (Z2) pendant l'approche pour tester un mécanisme de phase (le +7,9 sans jauge).
