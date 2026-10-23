"""PEW 3D cartesien (sans axe, sans coeur epingle), chunks reprenables sur duree.
usage: python -u cart_gh.py N L A0 TAG ; env: TIME_BUDGET (s), TOTAL_MAXITER, CK (iterations entre checkpoints)
Depart : etat 2D relaxe (f2d_r128,d2d_r128) projete sur la grille cartesienne avec le rayon d'anneau A0 (pas de gel, le champ complet est libre).
Test : l'etat final doit etre independant de A0 ; viriel E2=E4 ; E(h->0,L->inf) a comparer a 193.5 (toroidal, sans 2pi)."""
import sys, os, json, time, numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
from cart3d import energy_cart, init_from_toroidal
N, L, A0, TAG = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
TB = float(os.environ.get('TIME_BUDGET', '18900')); TOTAL = int(os.environ.get('TOTAL_MAXITER', '200000')); CK = int(os.environ.get('CK', '200'))
h = 2*L/(N-1); NPZ = f'cart_{TAG}.npz'; HIST = f'cart_{TAG}_history.jsonl'
f2 = np.load('f2d_r128.npy'); d2 = np.load('d2d_r128.npy')
n0 = init_from_toroidal(f2, d2, 12.0, A0, N, L)
n0j = jnp.array(n0)
vg = jax.jit(jax.value_and_grad(lambda x: energy_cart(x, N, h, n0j)))
def parts(x):
    # E2 = (1/4)|dn|^2 , E4 = (1/4) sum |d_i n x d_j n|^2 ; meme schema que energy_cart (8 combinaisons)
    n = n0j.at[1:-1, 1:-1, 1:-1, :].set(jnp.array(x).reshape(N-2, N-2, N-2, 3))
    n = n/jnp.linalg.norm(n, axis=-1, keepdims=True); c = n[1:-1, 1:-1, 1:-1]
    d = [((n[2:, 1:-1, 1:-1]-c)/h, (c-n[:-2, 1:-1, 1:-1])/h), ((n[1:-1, 2:, 1:-1]-c)/h, (c-n[1:-1, :-2, 1:-1])/h), ((n[1:-1, 1:-1, 2:]-c)/h, (c-n[1:-1, 1:-1, :-2])/h)]
    X = (jnp.arange(N-2)-(N-3)/2)*h; XX, YY, ZZ = jnp.meshgrid(X, X, X, indexing='ij'); R2 = XX**2+YY**2+ZZ**2
    E2 = E4 = M2 = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for g in (0, 1):
                D = (d[0][a], d[1][b], d[2][g]); k = 0.25*sum(jnp.sum(v*v, -1) for v in D); q = 0.0
                for i in range(3):
                    for j in range(3):
                        if i != j: cr = jnp.cross(D[i], D[j]); q = q+jnp.sum(cr*cr, -1)
                E2 += jnp.sum(k)/8; E4 += jnp.sum(0.25*q)/8; M2 += jnp.sum((k+0.25*q)*R2)/8
    return float(E2*h**3), float(E4*h**3), float(M2*h**3)
def f(y):
    E, g = vg(jnp.array(y)); return float(E), np.asarray(g, float)
nit0 = 0
if os.path.exists(NPZ):
    z = np.load(NPZ); x0 = z['x']; nit0 = int(z['nit']); print('Reprise', NPZ, 'nit', nit0, flush=True)
else:
    x0 = n0[1:-1, 1:-1, 1:-1, :].ravel(); print('Initialisation A0 =', A0, flush=True)
T0 = time.time(); cnt = [0]; last = [x0]
def diag(x, nit, tag=''):
    E, g = f(x); E2, E4, M2 = parts(x)
    rec = dict(it=nit, E=E, rms=float(np.sqrt(np.mean(g**2))), E2=E2, E4=E4, E2sE4=E2/E4, r2moy=M2/E, a_eff=float(np.sqrt(E4/E2)), t=time.time())
    open(HIST, 'a').write(json.dumps(rec)+'\n'); print({k: (round(v, 8) if isinstance(v, float) else v) for k, v in rec.items() if k != 't'}, flush=True); return rec
class Stop(Exception): pass
def cb(xk):
    cnt[0] += 1; last[0] = xk.copy()
    if cnt[0] % CK == 0:
        np.savez(NPZ+'.tmp.npz', x=xk, nit=nit0+cnt[0]); os.replace(NPZ+'.tmp.npz', NPZ); diag(xk, nit0+cnt[0])
    if time.time()-T0 > TB: raise StopIteration
E0 = f(x0)[0]; print(f'N={N} L={L} h={h:.4f} A0={A0} E(depart)={E0:.5f}', flush=True)
try:
    r = minimize(f, x0, jac=True, method='L-BFGS-B', callback=cb, options=dict(maxiter=TOTAL-nit0, maxfun=3*(TOTAL-nit0), maxcor=30, ftol=1e-16, gtol=1e-11))
    xf = r.x; nitf = nit0+r.nit; msg = str(r.message)
except StopIteration:
    xf = last[0]; nitf = nit0+cnt[0]; msg = 'time'
np.savez(NPZ+'.tmp.npz', x=xf, nit=nitf); os.replace(NPZ+'.tmp.npz', NPZ)
rec = diag(xf, nitf)
fin = (rec['rms'] < 2e-8) or (nitf >= TOTAL)
json.dump(dict(nit_ckpt=nitf, E=rec['E'], rms=rec['rms'], finished=bool(fin), msg=msg), open(f'cart_{TAG}_state.json', 'w'))
print('FINAL', rec['E'], rec['rms'], 'nit', nitf, 'finished', fin, msg, flush=True)
