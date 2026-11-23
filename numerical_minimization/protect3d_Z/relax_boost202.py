"""part202 -- raffinement en h du test de boost. Usage : python relax_boost202.py H [TIME_BUDGET_s]
Relaxe wk_T0 (L0, boite L=32 fixe, N=L/h, float64, L-BFGS avec relances jusqu'a |grad/h^3|_inf < 2e-3) puis boost v = 0,05 ; 0,1 ; 0,2 selon x, y, z, KF=0 et KF=1. Ecrit boost202_h<H>.json."""
import sys, time, json, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '.'); jax.config.update("jax_enable_x64", True)
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
import lagcov
h = float(sys.argv[1]); budget = float(sys.argv[2]) if len(sys.argv) > 2 else 15000; N = int(round(32 / h)); t0 = time.time()
f0 = np.load('../../resultats/part179/wk_T0/wk_T0_champ_final.npy').astype(np.float64); c0 = 31.5; c = (N - 1) / 2
ix = (np.arange(N) - c) * h / 0.5 + c0; X, Y, Z = np.meshgrid(ix, ix, ix, indexing='ij')
x = np.stack([map_coordinates(f0[k], [X, Y, Z], order=3, mode='grid-wrap') for k in range(4)]).ravel()
S = lagcov.build(N, h=h, nu=0.0, g2=0.0, KF=0.0); th = jnp.zeros((3, N, N, N)); vg = jax.jit(jax.value_and_grad(lambda p: S['energy'](p, th)))
def fun(z):
    e, g = vg(jnp.array(z.reshape(4, N, N, N))); return float(e), np.array(g, dtype=np.float64).ravel()
log = []
while True:
    r = minimize(fun, x, jac=True, method='L-BFGS-B', options=dict(maxiter=1500, maxcor=50, ftol=1e-16, gtol=1e-12)); x = r.x
    e, g = fun(x); gi = float(np.abs(g).max() / h**3); log.append((round(time.time() - t0), e, gi, r.nit)); print('relance : t=%d s  E=%.4f  |g/h^3|_inf=%.5f  iters=%d' % log[-1], flush=True)
    if gi < 2e-3 or time.time() - t0 > budget or r.nit < 5: break
f = x.reshape(4, N, N, N); ix2 = np.arange(N, dtype=float); X, Y, Z = np.meshgrid(ix2, ix2, ix2, indexing='ij')
dxf = lambda a: (np.roll(a, -1, axis=1) - np.roll(a, 1, axis=1)) / (2 * h)
out = dict(h=h, N=N, E_rest=e, grad_inf=gi, relax_log=log, boost=[])
for v in (0.05, 0.1, 0.2):
    g_ = 1 / np.sqrt(1 - v * v)
    for KF in (0.0, 1.0):
        Sb = lagcov.build(N, h=h, nu=0.0, g2=0.0, KF=KF); w = jnp.zeros((3, N, N, N)); res = []
        for ax in (0, 1, 2):
            perm = list(range(3)); perm[0], perm[ax] = perm[ax], perm[0]; fp = np.stack([np.transpose(f[k], perm) for k in range(4)])
            phb = np.stack([map_coordinates(fp[k], [c + g_ * (X - c), Y, Z], order=3, mode='grid-wrap') for k in range(4)]); vel = -v * dxf(phb)
            q = (jnp.array(phb), th); vv = (jnp.array(vel), w); M = float(Sb['energy'](jnp.array(fp), th)); Hb = float(Sb['hamil'](q, vv)); p = Sb['pfun'](q, vv)[0]; P = float(-jnp.sum(p * jnp.array(dxf(phb))))
            res.append(dict(axis=ax, P_over_Mv=P / (M * v), H_over_M=Hb / M))
        out['boost'].append(dict(v=v, gamma=g_, KF=KF, M=M, res=res)); print('v=%g KF=%g :' % (v, KF), [(round(a['P_over_Mv'], 4), round(a['H_over_M'], 5)) for a in res], 'gamma=%.5f' % g_, flush=True)
json.dump(out, open(f'boost202_h{h}.json', 'w'), indent=1); print('FIN', round(time.time() - t0), 's')
