# Palier 1 (part201b, CI) — dt=0.005, N=96, h=0.5, T=24, L0 sans jauge, ν=0
opp : t_ann KF0 = 5.5 (dt 0.0025 et 0.005 identiques) ; KF1 = 16.0 (dt 0.005) -> rapport 2.9. Qabs final KF1 = 0.084.
same : separation des pics en x de 11 a 28.5 (KF0, dt identiques) contre 18.5 (KF1 dt 0.005) a t=24 : deplacement 17.5 -> 7.5 (rapport 0.43).
Controle dt : KF0 dt 0.0025 vs 0.005 -> resultats identiques (t_ann, Ekm_max 1062, derive 35.4 %). KF1 : dt 0.0025 s'etait arrete a t=15 (budget).
A signaler : same_kf1_dt5 : cos_min_run_min = -0.87 et Ekm_max = 325 (vs 184 sur dt 0.0025 jusqu'a t=15) ; sens non elucide, a regarder avant de citer ce run.
Derive d'energie de diagnostic (Ekm+E2+E4+EV) : 3.9 % (opp KF1), 5.0 % (same KF1), 35 % (opp KF0 : annihilation + eponge), non interpretable en presence d'eponge.
