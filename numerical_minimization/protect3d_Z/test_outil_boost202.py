"""part202 -- validation de l'OUTIL de boost sur un cas exact : kink/antikink de sine-Gordon dans un reseau 3D (y,z uniformes).
L = h^3 sum[ phi_dot^2 ] - E,  E = h^3 sum[ sum_axes (phi(x+h)-phi)^2/h^2 + 2(1-cos phi) ].  Kink : phi = 4 atan(exp(x)), M = 16 par kink, 32 pour la paire.
Meme procedure que test_boost202.py : phi_b = phi_0(c+gamma(x-c)) (interp. cubique, grid-wrap), vel = -v d_x phi_b (diff. centree), P = -sum(p d_x phi_b) avec p = dK/dv (autodiff), H = K + E.
Covariance exacte (continu) : P/(M v) = H/M = gamma."""
import sys, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
def run(h, L=80.0, D=30.0, NY=4):
    N = int(round(L / h)); x = (np.arange(N) - (N - 1) / 2) * h
    ph1 = 4 * np.arctan(np.exp(x + D / 2)) - 4 * np.arctan(np.exp(x - D / 2))     # kink en -D/2, antikink en +D/2 (valeur 0 -> 2pi -> 0 : ph1 va de 0 a 2pi puis revient a 0)
    f0 = np.broadcast_to(ph1[:, None, None], (N, NY, NY)).copy()
    sh = lambda f, a: jnp.roll(f, -1, axis=a)
    E = lambda f: h**3 * jnp.sum(sum((sh(f, a) - f)**2 / h**2 for a in range(3)) + 2 * (1 - jnp.cos(f)))
    K = lambda f, v: h**3 * jnp.sum(v**2)
    vg = jax.jit(jax.value_and_grad(E))
    def fun(z): e, g = vg(jnp.array(z.reshape(N, NY, NY))); return float(e), np.array(g, dtype=np.float64).ravel()
    r = minimize(fun, f0.ravel(), jac=True, method='L-BFGS-B', options=dict(maxiter=5000, gtol=1e-13, ftol=1e-16, maxcor=30)); f = r.x.reshape(N, NY, NY)
    M = float(E(jnp.array(f))); gi = np.abs(fun(r.x)[1]).max() / h**3
    c = (N - 1) / 2; ix = [np.arange(s, dtype=float) for s in f.shape]; X, Y, Z = np.meshgrid(*ix, indexing='ij'); dxf = lambda a: (np.roll(a, -1, axis=0) - np.roll(a, 1, axis=0)) / (2 * h)
    out = []
    for v in (0.1, 0.2, 0.5):
        g = 1 / np.sqrt(1 - v * v); phb = map_coordinates(f, [c + g * (X - c), Y, Z], order=3, mode='grid-wrap'); vel = -v * dxf(phb)
        p = jax.grad(K, 1)(jnp.array(phb), jnp.array(vel)); P = float(-jnp.sum(p * jnp.array(dxf(phb)))); Hb = float(K(None, jnp.array(vel)) + E(jnp.array(phb)))
        out.append((v, g, P / (M * v) - g, Hb / M - g, np.sqrt(max(Hb**2 - P**2, 0)) / M - 1))
    return N, M, gi, out
print('kink+antikink SG : M exact = 32 (par unite de surface) ; colonnes : v, gamma, P/(Mv)-gamma, H/M-gamma, sqrt(H^2-P^2)/M-1')
for h in (0.5, 0.4, 0.32, 0.25):
    N, M, gi, out = run(h); print(f'h={h} N={N} M={M / (4 * 4 * h * h):.4f} (par unite de surface ; exact 32 + erreur de reseau) |g/h3|={gi:.1e}')
    for v, g, a, b, c in out: print(f'    v={v}: gamma={g:.4f}  P/(Mv)-gamma = {a:+.5f}   H/M-gamma = {b:+.5f}   sqrt(H^2-P^2)/M-1 = {c:+.5f}', flush=True)
