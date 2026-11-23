"""part202 -- tests de validite de lagcov.py (a passer AVANT tout run de production). Usage : python test_lagcov.py [N_gauge]   (float64)
T0 energie = jauge.energy ; T1 invariance de jauge exacte de L sous lambda(x,t) ; T2 Euler-Lagrange (CG) = Hessienne dense ; T3 Gauss et H conserves (RK4)."""
import os, sys, runpy, numpy as np
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zlat import hopfion
jax.config.update("jax_enable_x64", True)
import lagcov
D = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(7); H = 0.5
def rand_state(N, amp_th=0.3, amp_v=0.2):
    phi = jnp.array(hopfion(N, H, 1.2).astype(np.float64) + 0.05 * rng.normal(size=(4, N, N, N)))
    th = jnp.array(amp_th * rng.normal(size=(3, N, N, N))); v = jnp.array(amp_v * rng.normal(size=(4, N, N, N))); w = jnp.array(amp_v * rng.normal(size=(3, N, N, N)))
    return (phi, th), (v, w)
tr = lambda f, *a: jax.tree_util.tree_map(f, *a)
ok = True
def report(name, val, tol):
    global ok; good = val < tol; ok &= good; print(f"{'OK  ' if good else 'ECHEC'} {name} : {val:.3e} (seuil {tol:.0e})", flush=True)
# ---- T0 : copie de l'energie
N = 6; S = lagcov.build(N, g2=0.05, KF=1.0); q, (v, w) = rand_state(N)
sys.argv = ['jauge.py', 'none']; os.environ['N'] = str(N); jg = runpy.run_path(os.path.join(D, 'jauge.py'), run_name='jauge')
Ej = float(jg['energy'](q[0], q[1], 0.12, 0.05)); Ec = float(S['energy'](*q)); report("T0 E(lagcov)/E(jauge.py) - 1", abs(Ec / Ej - 1), 1e-9)
# ---- T1 : invariance de jauge de L sous lambda(x,t)
a0 = jnp.array(0.2 * rng.normal(size=(N, N, N))); lam = jnp.array(rng.uniform(-3, 3, size=(N, N, N))); lamd = jnp.array(rng.normal(size=(N, N, N)))
(phi, th) = q; c2 = (phi[2] + 1j * phi[3]) * jnp.exp(1j * lam); phi2 = phi.at[2].set(c2.real).at[3].set(c2.imag)
z2 = phi[2] + 1j * phi[3]; vz2 = (v[2] + 1j * v[3]); vz2n = jnp.exp(1j * lam) * (vz2 + 1j * lamd * z2); v2 = v.at[2].set(vz2n.real).at[3].set(vz2n.imag)
th2 = jnp.stack([th[a] + jnp.roll(lam, -1, axis=a) - lam for a in range(3)]); w2 = jnp.stack([w[a] + jnp.roll(lamd, -1, axis=a) - lamd for a in range(3)]); a02 = a0 + lamd
L1 = float(S['lagr']((phi, th), (v, w), a0)); L2 = float(S['lagr']((phi2, th2), (v2, w2), a02)); report("T1 invariance de jauge |L'-L|/|L|", abs(L2 - L1) / abs(L1), 1e-9)
S0 = lagcov.build(N, g2=0.05, KF=0.0); k1 = float(S0['kin'](phi, th, v, w, a0)); k2 = float(S0['kin'](phi2, th2, v2, w2, a02)); report("T1' (KF=0) invariance de K", abs(k2 - k1) / abs(k1), 1e-9)
# ---- T2 : Euler-Lagrange par CG contre Hessienne dense (N=4)
n4 = 4; S4 = lagcov.build(n4, g2=0.05, KF=1.0); q4, (v4, w4) = rand_state(n4)
flat = lambda t: jnp.concatenate([x.ravel() for x in t]); shp = [x.shape for x in q4]
def unflat(x):
    out, i = [], 0
    for s in shp: m = int(np.prod(s)); out.append(x[i:i + m].reshape(s)); i += m
    return tuple(out)
a0z = jnp.zeros((n4, n4, n4))
Lf = lambda qf, vf: S4['kin'](*unflat(qf), *unflat(vf), a0z) - S4['energy'](*unflat(qf))
qf, vf = flat(q4), flat((v4, w4))
Hvv = jax.hessian(Lf, 1)(qf, vf); Lq = jax.grad(Lf, 0)(qf, vf); Lvq = jax.jacfwd(jax.grad(Lf, 1), 0)(qf, vf)
a_dense = jnp.linalg.solve(Hvv, Lq - Lvq @ vf)
a_cg = flat(S4['acc'](q4, (v4, w4)))
report("T2 EL (CG) vs Hessienne dense : ecart relatif", float(jnp.linalg.norm(a_cg - a_dense) / jnp.linalg.norm(a_dense)), 1e-8)
ev = np.linalg.eigvalsh(np.array(Hvv)); print(f"     M : valeur propre min {ev.min():.3e} (doit etre > 0), max {ev.max():.3e}")
# ---- T3 : conservation de Gauss et de H sous RK4
Nt = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
St = lagcov.build(Nt, g2=0.05, KF=1.0); qt, vt = rand_state(Nt, amp_th=0.1, amp_v=0.1)
deriv = lambda q, v: (v, St['acc'](q, v))
def rk4(q, v, dt):
    k1 = deriv(q, v); k2 = deriv(tr(lambda a, b: a + 0.5 * dt * b, q, k1[0]), tr(lambda a, b: a + 0.5 * dt * b, v, k1[1]))
    k3 = deriv(tr(lambda a, b: a + 0.5 * dt * b, q, k2[0]), tr(lambda a, b: a + 0.5 * dt * b, v, k2[1])); k4 = deriv(tr(lambda a, b: a + dt * b, q, k3[0]), tr(lambda a, b: a + dt * b, v, k3[1]))
    comb = lambda x, a, b, c, d: x + dt / 6 * (a + 2 * b + 2 * c + d)
    return tr(comb, q, k1[0], k2[0], k3[0], k4[0]), tr(comb, v, k1[1], k2[1], k3[1], k4[1])
G0 = St['gauss'](qt, vt); H0 = float(St['hamil'](qt, vt)); q_, v_ = qt, vt; dt = 0.005
for i in range(40): q_, v_ = rk4(q_, v_, dt)
G1 = St['gauss'](q_, v_); H1 = float(St['hamil'](q_, v_))
report("T3 Gauss : |G(t)-G(0)|/|G(0)| (40 pas RK4)", float(jnp.linalg.norm(G1 - G0) / jnp.linalg.norm(G0)), 1e-6)
report("T3 energie H : |H(t)-H(0)|/H(0)", abs(H1 - H0) / H0, 1e-6)
# controle de puissance : sans le couplage (n3-1)E ni KF, Gauss serait-il encore conserve ? (valeur informative)
print("TOUS LES TESTS PASSENT" if ok else "AU MOINS UN TEST ECHOUE")
