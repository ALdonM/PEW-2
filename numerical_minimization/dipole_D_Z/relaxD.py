"""part168 -- Dipôle de courant de phase D d'un nœud de Hopf du modèle Z (PEW).
Champ Z = e^{i k.x} Psi, Psi périodique ; gradient avec phase de Peierls : (Psi(x+e_i) e^{i k_i h} - Psi(x))/h.
E(k) = h^3 sum[ |D_k Psi|^2 + cF Faddeev(n(Psi)) + mu^2 (rho^2 + 1/(rho^2+d2) - 2) ]  (n et rho ne voient pas la phase commune).
Identité exacte sur réseau : dE/dk_i = 2 D_i,  D_i = h^2 sum Im( Psi^dag(x) Psi(x+e_i) e^{i k_i h} )  (-> int Im(Psi^dag d_i Psi)).
Relaxation L-BFGS en float64 ; reprise sur checkpoint.  usage : python relaxD.py TAG  (paramètres par variables d'environnement)"""
import os, sys, json, time, numpy as np
from zlat import nvec, cross, hopfion   # zlat règle x64 à False : on le rétablit APRÈS l'import
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.optimize import minimize

env = lambda k, d: type(d)(os.environ.get(k, d))
TAG = sys.argv[1]
N, h, mu, e0 = env('N', 64), env('H', 0.5), env('MU', 0.125), env('E0', 0.125)
k = np.array([env('KX', 0.0), env('KY', 0.0), env('KZ', 0.0)])
INIT, MAXIT, BUDGET, CK = env('INIT', 'fresh'), env('MAXITER', 20000), env('TIME_BUDGET', 19000.0), env('CK', 200)
d2, cF = 1e-3, 1.0 / (2 * e0**2)
ph = [complex(np.exp(1j * k[i] * h)) for i in range(3)]

def fields(x):
    p = x.reshape(4, N, N, N)
    return p[0] + 1j * p[1], p[2] + 1j * p[3], p

def parts(x):
    Z1, Z2, p = fields(x)
    g = 0.0
    for i, ax in enumerate((0, 1, 2)):
        g = g + jnp.abs(jnp.roll(Z1, -1, ax) * ph[i] - Z1)**2 + jnp.abs(jnp.roll(Z2, -1, ax) * ph[i] - Z2)**2
    g = g / h**2
    n = nvec(p)
    Df = [(jnp.roll(n, -1, a) - n) / h for a in (1, 2, 3)]
    Db = [(n - jnp.roll(n, 1, a)) / h for a in (1, 2, 3)]
    f = 0.0
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        for Di in (Df[i], Db[i]):
            for Dj in (Df[j], Db[j]):
                f = f + 0.25 * jnp.sum(n * cross(Di, Dj), axis=0)**2
    r2 = jnp.sum(p**2, axis=0)
    v = mu**2 * (r2 + 1.0 / (r2 + d2) - 2.0)
    return h**3 * jnp.array([jnp.sum(g), cF * jnp.sum(f), jnp.sum(v)])

E = jax.jit(lambda x: jnp.sum(parts(x)))
gE = jax.jit(jax.grad(lambda x: jnp.sum(parts(x))))
parts_j = jax.jit(parts)

@jax.jit
def dipole(x):
    Z1, Z2, p = fields(x)
    D, Phi_std = [], []
    for i, ax in enumerate((0, 1, 2)):
        jj = jnp.imag(jnp.conj(Z1) * jnp.roll(Z1, -1, ax) * ph[i] + jnp.conj(Z2) * jnp.roll(Z2, -1, ax) * ph[i]) / h
        D.append(h**3 * jnp.sum(jj))
        other = tuple(a for a in (0, 1, 2) if a != ax)
        Phi = h**2 * jnp.sum(jj, axis=other)          # flux à travers chaque plan normal à l'axe
        Phi_std.append(jnp.std(Phi))
    return jnp.array(D), jnp.array(Phi_std)

@jax.jit
def charge(x):
    _, _, p = fields(x)
    z = p / jnp.sqrt(jnp.sum(p**2, axis=0))
    cen = lambda f, a: (jnp.roll(f, -1, a) - jnp.roll(f, 1, a)) / (2 * h)
    M = jnp.moveaxis(jnp.stack([z] + [cen(z, a) for a in (1, 2, 3)]), (0, 1), (-2, -1))
    r = jnp.sqrt(jnp.sum(p**2, axis=0))
    return h**3 * jnp.sum(jnp.linalg.det(M)) / (2 * np.pi**2), jnp.min(r)

ck, stf, hist = f"ck_{TAG}.npz", f"{TAG}_state.json", f"{TAG}_history.jsonl"
if os.path.exists(ck):
    z = np.load(ck); x0 = z['x']; it0 = int(z['it']); print("reprise à l'itération", it0, flush=True)
else:
    x0 = (hopfion(N, h, 6.0) if INIT == 'fresh' else np.load(INIT)).astype(np.float64).ravel(); it0 = 0
    json.dump(dict(N=N, h=h, mu=mu, e0=e0, k=k.tolist(), INIT=INIT), open(f"{TAG}_params.json", "w"))

w0 = time.time(); cnt = [it0]; xs = [x0]
def log(x, it, extra=None):
    xj = jnp.array(x); P = np.array(parts_j(xj)); D, Ps = dipole(xj); Q, rmin = charge(xj)
    gn = float(jnp.linalg.norm(gE(xj)))
    m = dict(it=it, E=float(P.sum()), E2=float(P[0]), E4=float(P[1]), EV=float(P[2]), D=[float(a) for a in D],
             Phi_std=[float(a) for a in Ps], Q=float(Q), rho_min=float(rmin), gnorm=gn, wall=round(time.time() - w0))
    if extra: m.update(extra)
    open(hist, "a").write(json.dumps(m) + "\n"); print(json.dumps(m), flush=True)
class Stop(Exception): pass
def cb(x):
    cnt[0] += 1; xs[0] = x
    if cnt[0] % CK == 0:
        np.savez(ck, x=x, it=cnt[0]); log(x, cnt[0])
    if time.time() - w0 > BUDGET or cnt[0] - it0 >= MAXIT: raise Stop
fg = lambda x: (float(E(jnp.array(x))), np.array(gE(jnp.array(x))))
log(x0, it0)
finished = False
try:
    res = minimize(fg, x0, jac=True, method='L-BFGS-B', callback=cb,
                   options=dict(maxiter=10**7, maxcor=30, gtol=1e-9, ftol=1e-15))
    xs[0] = res.x; finished = True; msg = res.message
except Stop:
    msg = 'budget ou MAXITER'
    finished = cnt[0] - it0 >= MAXIT and os.environ.get('LAST_CHUNK', '0') == '1'
np.savez(ck, x=xs[0], it=cnt[0]); log(xs[0], cnt[0], dict(fin=str(msg)))
json.dump(dict(finished=bool(finished), it=cnt[0], message=str(msg)), open(stf, "w")); print("FIN", finished, msg)
