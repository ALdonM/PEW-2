"""part210 -- regression de lagcov (KF=0 et KF=1, nu=0.12, theta=0) sur le noeud libre relaxe de part187e (L=64, N=128, h=0.5) ; attendu E_seul = 931.3220 au-dessus du vide."""
import sys, json, numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import lagcov
f = np.load(sys.argv[1]).astype(np.float64); print('champ', f.shape, f.dtype); N = f.shape[1]; h = 0.5; mu = 0.375; d2 = 1e-3
evac = mu**2 * (1 / (1 + d2) - 1) * (N * h)**3
out = dict(shape=list(f.shape), evac_V=evac)
for KF in (0.0, 1.0):
    S = lagcov.build(N, h=h, mu=mu, nu=0.12, g2=0.02, KF=KF)
    E = float(S['energy'](jnp.array(f), jnp.zeros((3, N, N, N)))); out[f'E_brut_lagcov_nu012_theta0']= E; out['E_rel_lagcov'] = E - evac
    print('E brut', E, 'E - e_vac V =', E - evac, ' (attendu 931.3220)', flush=True); break   # l'energie statique ne depend pas de KF
json.dump(out, open('regress187_210.json', 'w'))
