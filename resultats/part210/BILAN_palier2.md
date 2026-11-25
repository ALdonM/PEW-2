# Palier 2 (part210, CI) — h=0.4, N=120, dt=0.004, L0 sans jauge (nu=0), paire opp, separation 12 unites
Noeud de depart relaxe a h=0.4 (Newton-Krylov, boite L=32). Sorties : p210_opp_kf0_h04/ (T=9), p210_opp_kf1_h04/ (T=22), segments chaines par artefact.
| | h=0.5 (palier 1) | h=0.4 (palier 2) |
|---|---|---|
| t_ann KF=0 | 5.5 | 5.5 |
| t_ann KF=1 | 16.0 | 16.0 |
| Ekm_max KF=0 | 1062 | 1051 |
| Qabs final KF=1 (T=22 / 24) | 0.084 | 0.12 |
=> le rapport t_ann(KF1)/t_ann(KF0)=2.9 est inchange de h=0.5 a h=0.4 (pas de derive en h a la resolution de la mesure, pas 0.5).
KF=1 h=0.4 : cos_min_run_min=0.31 (le champ passe pres d'un zero de |phi| pendant l'annihilation). Energie Etot 1926 -> 1290 a T=22 (absorbee par l'eponge, non interpretable comme derive).
Non fait : same (memes chiralites) a h=0.4 ; dt non varie a h=0.4.
