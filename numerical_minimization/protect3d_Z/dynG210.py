"""part210 -- dynamique jaugee (lagcov, a0=0) avec absorbeur compatible Gauss (absorb210) : kick-drift-kick, CG preconditionne a depart a chaud.
Env : TAG(arg1) N H NU G2 KF DT T_END MEAS SPR0 SPRB SPSIG INIT_FILE(phi 4xN^3, theta=0, v=0) TIME_BUDGET TOL PREC_EVERY
Eponge : v <- v e^{-sig dt}; phi <- vac+(phi-vac)e^{-sig dt}; theta <- theta e^{-sig dt}; puis reprojection de v sur G=0.
Reprise : ck_{TAG}.npz (phi, th, vphi, vth, t). Sorties : {TAG}_history.jsonl, {TAG}_state.json, {TAG}_champ_final.npy"""
import os, sys, time, json, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, '.'); jax.config.update("jax_enable_x64", True)
import lagcov, absorb210
from zlat import make
env = lambda k, d: type(d)(os.environ.get(k, d))
TAG = sys.argv[1]; N = env('N', 64); h = env('H', 0.5); NU = env('NU', 0.12); G2 = env('G2', 0.02); KF = env('KF', 1.0)
dt = env('DT', 0.02); T_END = env('T_END', 10.0); MEAS = env('MEAS', 0.5); SPR0, SPRB, SPSIG = env('SPR0', 0.0), env('SPRB', 0.0), env('SPSIG', 0.3)
BUDGET = env('TIME_BUDGET', 18900.0); TOL = env('TOL', 1e-8); PREC_EVERY = env('PREC_EVERY', 5); INIT_FILE = os.environ.get('INIT_FILE', '')
MU, E0, D2 = 0.375, 0.25, 1e-3
S = lagcov.build(N, h=h, mu=MU, e0=E0, d2=D2, nu=NU, g2=G2, KF=KF); pfun = S['pfun']; cF = S['cF']
project, damp, Gam = absorb210.make(S, N, h, 1.0)
a0z = jnp.zeros((N, N, N)); tm = jax.tree_util.tree_map; zeros_like = jnp.zeros_like
Kz = lambda q, v: S['kin'](q[0], q[1], v[0], v[1], a0z)
dKq = jax.jit(jax.grad(Kz, 0)); dEq = jax.jit(jax.grad(lambda q: S['energy'](q[0], q[1])))
jp = jax.jit(lambda q, v: jax.jvp(lambda qq: pfun(qq, v), (q,), (v,))[1]); Mv = jax.jit(lambda q, v, x: jax.jvp(lambda vv: pfun(q, vv), (v,), (x,))[1])
rhs = jax.jit(lambda q, v: tm(lambda a, b, c: a - b - c, dKq(q, v), dEq(q), jp(q, v)))
@jax.jit
def build_prec(q, v):
    ones = lambda c: (jnp.zeros((4, N, N, N)).at[c].set(1.0), jnp.zeros((3, N, N, N)))
    B = jnp.stack([Mv(q, v, ones(c))[0] for c in range(4)], 0); Binv = jnp.linalg.inv(jnp.moveaxis(B, (0, 1), (-1, -2)))
    dth = jnp.stack([Mv(q, v, (jnp.zeros((4, N, N, N)), jnp.zeros((3, N, N, N)).at[a].set(1.0)))[1][a] for a in range(3)]); return Binv, dth
dot = lambda a, b: sum(jnp.vdot(x, y) for x, y in zip(a, b))
def acc(q, v, x0, P):
    Binv, dth = P; pr = lambda r: (jnp.einsum('...ij,j...->i...', Binv, r[0]), r[1] / dth)
    b = rhs(q, v); x = x0; r = tm(lambda a, c: a - c, b, Mv(q, v, x)); z = pr(r); p = z; rz = float(dot(r, z)); b2 = float(dot(b, b)); it = 0
    while float(dot(r, r)) > TOL**2 * b2 and it < 200:
        Ap = Mv(q, v, p); al = rz / float(dot(p, Ap)); x = tm(lambda a, c: a + al * c, x, p); r = tm(lambda a, c: a - al * c, r, Ap); z = pr(r); rzn = float(dot(r, z)); p = tm(lambda a, c: a + (rzn / rz) * c, z, p); rz = rzn; it += 1
    return x, it
hamil = jax.jit(lambda q, v: S['hamil'](q, v)); gauss = jax.jit(lambda q, v: S['gauss'](q, v)); energy = jax.jit(lambda q: S['energy'](q[0], q[1]))
project_j = jax.jit(project)
_, _, _, qd = make(h, MU, cF, D2); qd = jax.jit(qd)
KW = float(np.sqrt(1.0 - D2)); vac = jnp.array([KW, 0, 0, 0])[:, None, None, None]
SIG = absorb210.sponge(N, h, SPR0, SPRB, SPSIG) if SPR0 > 0 else None
ck, stf, hist = f"ck_{TAG}.npz", f"{TAG}_state.json", f"{TAG}_history.jsonl"
if os.path.exists(ck):
    z = np.load(ck); q = (jnp.array(z['phi']), jnp.array(z['th'])); v = (jnp.array(z['vphi']), jnp.array(z['vth'])); t = float(z['t']); print('reprise t=', t, flush=True)
else:
    phi0 = np.load(INIT_FILE).astype(np.float64); q = (jnp.array(phi0), jnp.zeros((3, N, N, N))); v = (jnp.zeros((4, N, N, N)), jnp.zeros((3, N, N, N))); t = 0.0
    open(hist, 'w').close()
def measure(q, v, t, its, n):
    qq = np.array(qd(q[0])) * h**3; H = float(hamil(q, v)); Ep = float(energy(q))
    r = dict(t=round(t, 4), H=H, E=Ep, K=H - Ep, Gmax=float(jnp.abs(gauss(q, v)).max()), Q=round(float(qq.sum()), 3), Qabs=round(float(np.abs(qq).sum()), 3),
             rho_min=float(jnp.sqrt(jnp.sum(q[0]**2, 0)).min()), cg=its / max(n, 1)); open(hist, 'a').write(json.dumps(r) + '\n'); print(json.dumps(r), flush=True)
P = build_prec(q, v); a, _ = acc(q, v, tm(zeros_like, v), P); t00 = time.time(); its = 0; n = 0; next_meas = (np.floor(t / MEAS + 1e-9) + (0 if os.path.exists(hist) and os.path.getsize(hist) > 0 else 0)) * MEAS
if os.path.getsize(hist) == 0: measure(q, v, t, 0, 1); next_meas = t + MEAS
else: next_meas = t + MEAS
fin = False
while t < T_END - 1e-9:
    if time.time() - t00 > BUDGET: break
    if n % PREC_EVERY == 0: P = build_prec(q, v)
    vh = tm(lambda x, y: x + 0.5 * dt * y, v, a); q = tm(lambda x, y: x + dt * y, q, vh); a, it1 = acc(q, vh, a, P); v = tm(lambda x, y: x + 0.5 * dt * y, vh, a); its += it1
    if SIG is not None:
        fs = jnp.exp(-SIG * dt); v = (v[0] * fs[None], v[1] * fs[None]); q = (vac + (q[0] - vac) * fs[None], q[1] * fs[None]); v = project_j(q, v)
    t += dt; n += 1
    if t >= next_meas - 1e-9: measure(q, v, t, its, n); next_meas += MEAS
    if n % 500 == 0: np.savez(ck, phi=np.array(q[0]), th=np.array(q[1]), vphi=np.array(v[0]), vth=np.array(v[1]), t=t)
fin = t >= T_END - 1e-9
np.savez(ck, phi=np.array(q[0]), th=np.array(q[1]), vphi=np.array(v[0]), vth=np.array(v[1]), t=t)
np.save(f"{TAG}_champ_final.npy", np.array(q[0], dtype=np.float32)); json.dump(dict(finished=fin, t=t, wall=time.time() - t00, steps=n), open(stf, 'w')); print('FIN', fin, t, flush=True)
