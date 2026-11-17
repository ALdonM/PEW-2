# Note part184t : paire à 45° de l'axe du spin et contrôles à impulsion annulée

Session s35, 2026-11-17. Audit isolé (agent séparé, recalcul depuis les journaux) avant écriture ; ses corrections sont intégrées. Modèle Z classique, aucune identification à une particule (discipline D). Données : `resultats/part184/gh_part184t/` (T_*), comparées à part184p/r/s. Nœud relaxé N=96 (RELT=120), NB=96, SEPC=20 (SEPD=14 en xz), ν=0,12, T=45, une réalisation par point. `ORIENT=xz` ajouté (paire le long de (1,0,1)/√2, B = miroir x→−x, axes de spin des deux nœuds selon z). Suite des notes part184pq, r, s.

## 1. Chiffres (t_ann = Qabs<0,5 interpolé ; les 14 valeurs confirmées ; Econs conservé à ≤0,02 ; Pz initial nul avec CANCEL_PZ)

Écart au t_ann de ω=0 de même orientation et de même g², Pz annulé, g²=0,02, ω=0,25 :

| Spins (Lz t=0) | côte à côte (x) | 45° (xz) | sur l'axe (z) |
|---|---|---|---|
| parallèles (≈ −590 à −600) | +6,85 | +9,85 | +6,32 |
| antiparallèles (≈ 0) | +0,34 | +0,23 | +20,4 |
| référence ω=0 | 22,88 | 26,06 | 19,90 |

Les +0,23 et +0,34 ne se distinguent pas de 0 (selon le seuil sur Qabs : 0,00 à 0,34). Séparation initiale 19,83 en xz contre 20,05 en x et z.

Autres : x antiparallèles annulé 23,22 contre 23,15 non annulé ; z antiparallèles 40,32 annulé contre 40,36 non annulé (insensible à CANCEL_PZ, donc pas un effet d'impulsion). À g²=0, Pz annulé, sur l'axe : parallèles 27,76 (+6,85 vs 20,91), antiparallèles 31,36 (+10,45).

## 2. Ce que cela établit

- Le retard des spins parallèles a **même signe et même ordre de grandeur (6,3 à 9,85) pour les trois orientations**, avec un maximum à 45° (non monotone en angle ; écart 6,3→9,85 de 50 %, bien au-delà de l'incertitude de seuil de ±0,3). Je retire « indépendant de l'orientation ».
- Un **dipôle–dipôle pur ∝ (1−3cos²θ)** est **exclu** : il prédirait un changement de signe entre l'axe et le côte à côte et un effet faible à 45°. Non exclus : un dipôle plus un terme isotrope, ou un effet du réseau cubique (indiscernables avec 3 angles, une orientation de spin, un g²).
- Les spins antiparallèles ne retardent (≤0,3, compatible avec 0) que **hors axe** ; sur l'axe, +20,4.
- Sur l'axe, le retard des parallèles ne dépend presque pas de g² (+6,85 à g²=0 ; +6,32 à 0,02 ; −2,3 aux deux sans annulation). En côte à côte il en dépend fortement (+1,05 à g²=0 ; +9,4 à 0,02, sans annulation ; pas de contrôle annulé en x à g²=0, donc le contraste n'est valable qu'à conditions égales sans annulation).

## 3. Sur l'axe, antiparallèles, g² : barrière et quasi-lié (NON ÉTABLI comme loi)

Annulé : g²=0 : +10,45 (descente régulière, sans rebond) ; g²=0,01 : **pas d'annihilation à T=45** (séparation minimale 17,7 à t=18, rebond jusqu'à 19,2, Qabs ≥1,93, borne >+24) ; g²=0,02 : +20,4 (minimum 17,9, rebond jusqu'à 18,8, annihilation à 40,3). La non-monotonie est formelle : t_ann est ici une quantité de seuil (franchissement d'une barrière avec état quasi-lié), pas une mesure continue de la force, et très sensible aux conditions initiales. Le ω=0 de même g² traverse sans rebond : la barrière vient de ω antiparallèle avec g²>0 (à g²=0 la paire passe sans rebond, +10,45). Il faut T≥75 pour g²=0,01, une métrique indépendante du seuil (séparation minimale, nombre de passages) et 2 à 3 variantes de condition initiale.

## 4. Limites

Une réalisation par point, sans barre d'erreur sur la sensibilité aux conditions initiales. Les références ω=0 diffèrent beaucoup par orientation (g²=0 : x 15,07 / z 20,91 ; g²=0,02 : x 22,88 / xz 26,06 / z 19,90), l'écart de base (3 à 6) dépassant les retards parallèles, et la soustraction t(ω)−t(0) suppose un découplage non testé. CANCEL_PZ n'est pas neutre (J −6 à −8 %, Ekin −7 à −8 %, source de jauge modifiée ; en z parallèles il retire une vitesse d'approche et explique à lui seul ~9 de t_ann) ; en xz il n'annule que Pz (mélange axial et transverse, Px non annulé). Asymétrie de J de 2 % en z (miroir imparfait). Images périodiques : 28 (x, z) contre 36,8 (xz) dans une boîte de 48 ; aucun test de N, h, dt ; xz n'a pas la symétrie miroir perpendiculaire à l'axe. Runs Y_g01_* (g²=0,1) inutilisables (Econs diverge).

## 5. Bilan provisoire du mécanisme (part184j à t)

Établi : le retard vient de la phase d'approche ; il dépend du spin parallèle ou non (par Lz physique, pas par le nom sj) ; il existe sans champ de jauge sur l'axe et en dépend fortement côte à côte ; il n'est pas dipolaire pur ; il n'est pas dû à l'impulsion axiale ni à la durée de relaxation. Non établi : le mécanisme exact.

## 6. Suite possible

T plus long (≥80) pour g²=0,01 sur l'axe ; variantes de condition initiale pour la dispersion ; boîte plus grande (nœuds tronqués ou N=128) contre les images ; contrôle annulé en x à g²=0 ; métrique indépendante du seuil ; test d'un terme isotrope + dipolaire par un second angle de spin (axe de spin tourné).
