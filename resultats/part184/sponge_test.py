"""Test de reflexion d'une couche absorbante spherique (modele Z, vide zeta=(1,0)).
usage: N=64 python3 sponge_test.py MODE SIGM SIGG   MODE in {matiere, jauge}; SIGM amortissement max de v, SIGG de w (jauge)."""
import os, sys, json, numpy as np
mode, sigm, sigg = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
sys.argv = ['jauge.py', 'none']
sys.path.insert(0, os.getcwd())
import jax, jax.numpy as jnp
import jauge as J
N, h = J.N, J.h; nu, g2 = 0.3, 0.02; dt = 0.05
xx = J.x_(N); X, Y, Z = np.meshgrid(xx, xx, xx, indexing='ij'); R = np.sqrt(X**2 + Y**2 + Z**2)
half = N * h / 2; Rs = 0.5 * half
prof = np.clip((R - Rs) / (half - Rs), 0, 1)**2
Sv = jnp.asarray(np.exp(-sigm * prof * dt).astype(np.float32)); Sw = jnp.asarray(np.exp(-sigg * prof * dt).astype(np.float32))
s = 1.5; env = np.exp(-R**2 / (2 * s**2)).astype(np.float32); eps = 0.2
p = jnp.zeros((4, N, N, N), jnp.float32); t = jnp.zeros((3, N, N, N), jnp.float32); v = jnp.zeros_like(p); w = jnp.zeros_like(t)
if mode == 'matiere':   # onde de phase de Z1 (sans masse), vitesse 1
    p = jnp.stack([jnp.cos(eps * env), jnp.sin(eps * env), 0 * env, 0 * env])
else:                   # impulsion transverse de jauge (theta_y(x,z) -> rayonne), Z2=0
    p = jnp.stack([1 + 0 * env, 0 * env, 0 * env, 0 * env]); t = t.at[1].set(eps * 3 * jnp.asarray(env * (X > -1e9)))
    t = t.at[1].set(jnp.asarray((eps * 3 * np.exp(-(X**2 + Z**2) / (2 * s**2)) * np.exp(-(Y / 6)**2)).astype(np.float32)))
@jax.jit
def kdk(p, t, v, w, a, b):
    v = v + 0.5 * dt * a; w = w + 0.5 * dt * b; p = p + dt * v; t = t + dt * w
    a, b = J.accs(p, t, nu, g2); v = (v + 0.5 * dt * a) * Sv[None]; w = (w + 0.5 * dt * b) * Sw[None]
    return p, t, v, w, a, b
a, b = J.accs(p, t, nu, g2)
inn = jnp.asarray(R < Rs - 1.0)
def ener(p, t, v, w):
    return float(J.energy(p, t, nu, g2) + h**3 * jnp.sum(v**2) + h**3 * jnp.sum(w**2) / (2 * g2 * h**2))
pv = jnp.stack([jnp.ones((N,N,N)), jnp.zeros((N,N,N)), jnp.zeros((N,N,N)), jnp.zeros((N,N,N))]); Evac = float(J.energy(pv, t * 0, nu, g2))
E0 = ener(p, t, v, w) - Evac; out = []
for i in range(int(float(os.environ.get("TT", 50)) / dt) + 1):
    if i % int(5 / dt) == 0:
        Ein = float(h**3 * jnp.sum((jnp.sum(v**2, 0) + jnp.sum(w**2, 0) / (2 * g2 * h**2)) * inn)) * 2   # equipartition
        out.append((round(i * dt), round((ener(p, t, v, w) - Evac) / E0, 4), round(Ein / E0, 4)))
    p, t, v, w, a, b = kdk(p, t, v, w, a, b)
print(mode, 'sigm', sigm, 'sigg', sigg, 'E0', round(E0, 4)); print('t, E/E0, Ein/E0 :', out, flush=True)
