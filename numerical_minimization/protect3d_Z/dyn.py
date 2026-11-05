"""part176 -- Protection de la charge de Hopf sur réseau (note de cadrage part175) : barrière d'admissibilité
P = EPSB * sum_{7 directions de Kuhn} B(cos theta), B = (c*-c)^2/c pour 0<c<c*, 0 sinon (BARRIER=1) ; témoin BARRIER=0.
Pas de biais (KW forcé à 0). Mesures : cos_min sur les arêtes de Kuhn, fraction d'arêtes sous c*, énergie de barrière,
minimum courant de cos entre deux mesures (contrôle du pas fini), degré exact, degré par blocs, amas.
part174 -- (v2 de part171) biais KW(t) = KW0/a pour t < T_OFF, coupé ensuite ; charge mesurée aussi par le degré EXACT (entier,
comptage des préimages, degre.py) à pleine résolution et après moyenne par blocs 2x2x2 (sépare les vrais nœuds des défauts à l'échelle de la maille).
part171 -- (dérivé de part166) Génération d'asymétrie nœuds/anti-nœuds : terme -KW * Q_H ajouté à l'énergie
(potentiel chimique de Hopf kappa*omega, Sakharov 2), chaos initial symétrique.
part166 -- Dynamique réelle d'un chaos d'ondes du modèle Z sur réseau 3D, avec expansion (ère de rayonnement)
ou friction constante, checkpoints et reprise (GitHub Actions, tranches < 6 h).
Coordonnées comobiles, a(t) = sqrt(t/t0), H = 1/(2t) (mode 'rad') ; mode 'none' : a = 1, gamma = G0.
  phi'' + 3H phi' = -(1/2) [ a^-2 dE2/dphi + a^-4 dE4/dphi + dEV/dphi ] / h^3
E2 = |grad Z|^2, E4 = Faddeev(n) (schéma anti-damier), EV = mu^2 (rho^2 + 1/(rho^2+d2) - 2)  (voir zlat.py).
Terme de Faddeev temporel omis [D]. Schéma : saut-de-mouton avec friction exacte exp(-3H dt).
usage : python dyn.py TAG   (paramètres par variables d'environnement)"""
import os, sys, json, time, numpy as np, jax, jax.numpy as jnp
from scipy import ndimage
from zlat import energy_density, random_field, make

env = lambda k, d: type(d)(os.environ.get(k, d))
TAG = sys.argv[1]
N, h, mu, e0, xi0, seed = env('N', 64), env('H', 0.5), env('MU', 0.125), env('E0', 0.125), env('XI0', 4.0), env('SEED', 1)
MODE, G0, t0, dt, T_END = env('MODE', 'rad'), env('G0', 0.05), env('T0', 1.0), env('DT', 0.005), env('T_END', 200.0)
MEAS, SNAP, BUDGET = env('MEAS', 2.0), env('SNAP', 20.0), env('TIME_BUDGET', 18900.0)
INIT, WARM = env('INIT', 'chaos'), env('WARM', 0.2)
KW = 0.0
BARRIER, EPSB, CSTAR = env('BARRIER', 0), env('EPSB', 0.5), env('CSTAR', 0.5)
KNOT_SIGN = env('KNOT_SIGN', 1)
CHECK = env('CHECK', 10)
T_OFF = env('T_OFF', 20.0)
from degre import degree   # potentiel chimique de Hopf (kappa * omega) en unités du modèle   # INIT = chaos | knot_in_chaos ; WARM : durée à dt/5 au départ
d2, cF = 1e-3, 1.0 / (2 * e0**2)

def Qlat(phi):
    z = phi / jnp.sqrt(jnp.sum(phi**2, axis=0))
    cen = lambda f, a: (jnp.roll(f, -1, a) - jnp.roll(f, 1, a)) / (2 * h)
    M = jnp.moveaxis(jnp.stack([z] + [cen(z, a) for a in (1, 2, 3)]), (0, 1), (-2, -1))
    return h**3 * jnp.sum(jnp.linalg.det(M)) / (2 * np.pi**2)

DIRS = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
def shift(f, d):
    for ax, s in enumerate(d):
        if s: f = jnp.roll(f, -s, 1 + ax)
    return f
def coss(phi):
    u = phi / jnp.sqrt(jnp.sum(phi**2, axis=0))
    return jnp.stack([jnp.sum(u * shift(u, d), axis=0) for d in DIRS])
def Pbar(phi):
    c = coss(phi)
    cc = jnp.clip(c, 1e-6, CSTAR)
    return EPSB * jnp.sum(jnp.where(c < CSTAR, (CSTAR - cc)**2 / cc, 0.0))
cosmin_j = jax.jit(lambda phi: jnp.min(coss(phi)))

def Ew(phi, w2, w4, kw):
    g, f, p = energy_density(phi, h, mu, cF, d2)
    return h**3 * jnp.sum(w2 * g + w4 * f + p) + (Pbar(phi) if BARRIER else 0.0)
gEw = jax.grad(Ew)

@jax.jit
def kdk(phi, v, acc, a_new, damp, dt, kw):
    # saut-de-mouton (une seule évaluation du gradient par pas) avec friction exacte exp(-3H dt/2) de part et d'autre
    v = v * damp + 0.5 * dt * acc
    phi = phi + dt * v
    acc = -0.5 * gEw(phi, a_new**-2, a_new**-4, kw) / h**3
    v = (v + 0.5 * dt * acc) * damp
    return phi, v, acc

acc_of = jax.jit(lambda phi, a, kw: -0.5 * gEw(phi, a**-2, a**-4, kw) / h**3)

_, _, parts, qd = make(h, mu, cF, d2)
def a_of(t): return np.sqrt(t / t0) if MODE == 'rad' else 1.0
def gam(t): return 1.5 / t if MODE == 'rad' else G0

ck, stf, hist = f"ck_{TAG}.npz", f"{TAG}_state.json", f"{TAG}_history.jsonl"
if os.path.exists(ck):
    z = np.load(ck); phi, v, t = jnp.array(z['phi']), jnp.array(z['v']), float(z['t'])
    print("reprise à t =", t, flush=True)
else:
    phi = random_field(N, h, xi0, 1.0, seed).astype(np.float64)
    if INIT == 'knot_in_chaos':   # produit de quaternions : degré du nœud (|Q| = 1) + degré du chaos
        from zlat import hopfion
        a_, b_ = hopfion(N, h, 6.0).astype(np.float64), phi
        if KNOT_SIGN < 0: a_ = a_ * np.array([1, -1, -1, -1])[:, None, None, None]
        a0, a1, a2, a3 = a_; b0, b1, b2, b3 = b_
        phi = np.stack([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2, a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0])
    phi = jnp.array(phi.astype(np.float32)); v = jnp.zeros_like(phi); t = t0
    json.dump(dict(N=N, h=h, mu=mu, e0=e0, xi0=xi0, seed=seed, MODE=MODE, G0=G0, t0=t0, dt=dt, INIT=INIT, T_END=T_END, BARRIER=BARRIER, EPSB=EPSB, CSTAR=CSTAR, KNOT_SIGN=KNOT_SIGN), open(f"{TAG}_params.json", "w"))

def measure(phi, v, t):
    a = a_of(t); E = np.array(parts(phi)); q = np.array(qd(phi)) * h**3
    r = np.sqrt(np.array(jnp.sum(phi**2, 0))); zz = np.array(phi) / r
    g, f, p = energy_density(phi, h, mu, cF, d2)
    e = np.array(g / a**2 + f / a**4 + p) * h**3
    es = ndimage.gaussian_filter(e, 2, mode='wrap')
    lab, nl = ndimage.label(es > 5 * es.mean(), structure=np.ones((3, 3, 3)))
    El = ndimage.sum(e, lab, np.arange(1, nl + 1)) if nl else np.array([])
    Ql = ndimage.sum(q, lab, np.arange(1, nl + 1)) if nl else np.array([])
    o = np.argsort(-El)[:8] if nl else []
    cc = coss(phi); cmin = jnp.min(cc)
    return dict(t=round(t, 3), a=round(a, 4), Ek=round(float(jnp.sum(v**2)) * h**3, 2),
                E2=round(E[0] / a**2, 2), E4=round(E[1] / a**4, 2), EV=round(E[2], 2),
                Q=round(float(q.sum()), 3), Qabs=round(float(np.abs(q).sum()), 3),
                rho_min=round(float(r.min()), 3), rho_med=round(float(np.median(r)), 3),
                order=round(float(np.linalg.norm(zz.reshape(4, -1).mean(1))), 4),
                clusters=[[round(float(El[i]), 1), round(float(Ql[i]), 2)] for i in o],
                deg=degree(np.array(phi, dtype=np.float64)),
                deg_bloc=degree(np.array(phi, dtype=np.float64).reshape(4, N//2, 2, N//2, 2, N//2, 2).mean((2, 4, 6))),
                cos_min=round(float(cmin), 4), frac_below=round(float(jnp.mean(cc < CSTAR)), 5),
                cos_min_run=round(float(RUNMIN[0]), 4),
                Ebar=round(float(Pbar(phi)), 3), Etot=round(float(jnp.sum(v**2)) * h**3 + E[0]/a**2 + E[1]/a**4 + E[2] + float(Pbar(phi) if BARRIER else 0.0), 3))

def kw_of(t): return (KW / a_of(t)) if t < T_OFF else 0.0
RUNMIN = [2.0]
acc = acc_of(phi, a_of(t), kw_of(t))
w0 = time.time(); next_meas = t; next_snap = (np.floor(t / SNAP) + 1) * SNAP; finished = False
with open(hist, "a") as H_:
    while True:
        if t >= next_meas - 1e-9:
            m = measure(phi, v, t); m['wall'] = round(time.time() - w0)
            H_.write(json.dumps(m, default=float) + "\n"); H_.flush(); print(json.dumps(m, default=float), flush=True)
            RUNMIN[0] = 2.0
            if not np.isfinite(m['E2']): print("DIVERGENCE"); break
            next_meas += MEAS
        if t >= next_snap - 1e-9:
            np.save(f"snap_{TAG}_t{int(round(t)):04d}.npy", np.array(phi, dtype=np.float16)); next_snap += SNAP
        if t >= T_END: finished = True; break
        if time.time() - w0 > BUDGET: break
        dte = dt / 5 if t < t0 + WARM else dt
        tm = t + dte / 2
        phi, v, acc = kdk(phi, v, acc, a_of(t + dte), float(np.exp(-gam(tm) * dte / 2)), dte, kw_of(t + dte))
        t += dte; nstep = locals().get('nstep', 0) + 1
        if nstep % CHECK == 0: RUNMIN[0] = min(RUNMIN[0], float(cosmin_j(phi)))
        if nstep % 2000 == 0:
            np.savez(ck, phi=np.array(phi), v=np.array(v), t=t)
np.savez(ck, phi=np.array(phi), v=np.array(v), t=t)
np.save(f"{TAG}_champ_final.npy", np.array(phi, dtype=np.float16))
json.dump(dict(finished=finished, t=t), open(stf, "w")); print("FIN", finished, t)
