# part174 — test d'asymétrie v2 (PEW, modèle Z)

Corrige part171 : charge mesurée par le degré exact (entier, comptage signé des préimages, `degre.py`), à pleine résolution et
après moyenne par blocs 2×2×2 (sépare les vrais nœuds des défauts à l'échelle de la maille) ; biais KW(t) = KW₀/a sous le seuil
d'emballement (KW₀ = ±10), coupé à t = 20 pour tester si l'asymétrie reste figée ; 4 graines × (0, +10, −10), μ = 0,375.
Observable : Δdeg(t) = deg_KW − deg_0 (même graine), moyenne et dispersion sur les graines. Résultats dans resultats/part174/.
