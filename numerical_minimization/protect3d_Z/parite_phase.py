"""part184 -- (A) parité / conjugaison de la charge de Hopf et de la charge de phase J=Im(Z^dag Zdot) dans le modèle Z ;
(B) un nœud portant une charge de phase LOCALISEE reste-t-il localisé (conservatif, G0=0) ?
usage : python parite_phase.py A    |    python parite_phase.py B OMEGA RW TEND
Champ phi = (Re Z1, Im Z1, Re Z2, Im Z2). Paramètres du modèle : mu=0.375, e0=0.25, h=0.5, N=64, d2=1e-3."""
import sys, json, time, numpy as np, jax, jax.numpy as jnp
from zlat import energy_density, hopfion
from degre import degree
import os
N, h, mu, e0, d2 = int(os.environ.get('N', 64)), 0.5, 0.375, 0.25, 1e-3; cF = 1 / (2 * e0**2)
E = lambda phi: h**3 * sum(jnp.sum(a) for a in energy_density(phi, h, mu, cF, d2))
Eparts = lambda phi: [float(h**3 * jnp.sum(a)) for a in energy_density(phi, h, mu, cF, d2)]
gE = jax.jit(jax.grad(E)); acc_of = jax.jit(lambda phi: -0.5 * gE(phi) / h**3)
def Qlat(phi):
    z = phi / jnp.sqrt(jnp.sum(phi**2, axis=0))
    cen = lambda f, a: (jnp.roll(f, -1, a) - jnp.roll(f, 1, a)) / (2 * h)
    M = jnp.moveaxis(jnp.stack([z] + [cen(z, a) for a in (1, 2, 3)]), (0, 1), (-2, -1))
    return float(h**3 * jnp.sum(jnp.linalg.det(M)) / (2 * np.pi**2))
Imul = lambda p: jnp.stack([-p[1], p[0], -p[3], p[2]])          # multiplication par i
qdens = lambda p, v: p[0]*v[1] - p[1]*v[0] + p[2]*v[3] - p[3]*v[2]   # Im(Z^dag Zdot)
x = (np.arange(N) - N / 2 + 0.5) * h
R = np.sqrt(sum(a**2 for a in np.meshgrid(x, x, x, indexing='ij')))

if sys.argv[1] == 'A':
    phi0 = jnp.array(hopfion(N, h, 6.0))
    P = lambda p: p[:, ::-1, :, :]                       # x -> -x (grille symétrique)
    P3 = lambda p: p[:, ::-1, ::-1, ::-1]               # inversion totale
    C = lambda p: p * jnp.array([1, -1, 1, -1])[:, None, None, None]    # Z -> Z* (det +1 sur R^4)
    D = lambda p: p * jnp.array([1, -1, 1, 1])[:, None, None, None]     # Z1 -> Z1* seul (det -1 sur R^4)
    ident = lambda p: p
    cas = [('id', ident), ('P (x->-x)', P), ('P3 (x->-x,y->-y,z->-z)', P3), ('C (Z->Z*)', C), ('P.C', lambda p: P(C(p))),
           ('D (Z1->Z1*, det -1)', D), ('P.D', lambda p: P(D(p)))]
    for nom, f in cas:
        p = f(phi0); ep = Eparts(p)
        print(json.dumps(dict(cas=nom, Qlat=round(Qlat(p), 4), degre_exact=degree(np.array(p, dtype=np.float64)),
                              E2=round(ep[0], 3), E4=round(ep[1], 3), EV=round(ep[2], 4), Etot=round(sum(ep), 3))))
    # charge de phase sous les transformations : champ de vitesse v = omega*i*Z localisé
    v0 = 0.2 * Imul(phi0) * jnp.exp(-(R / 8.0)**4)[None]
    J = lambda p, v: float(h**3 * jnp.sum(qdens(p, v)))
    print('J(id)=', round(J(phi0, v0), 3), ' J(P)=', round(J(P(phi0), P(v0)), 3), ' J(C)=', round(J(C(phi0), C(v0)), 3),
          ' J(D)=', round(J(D(phi0), D(v0)), 3))
else:
    om, Rw, TEND = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]); dt = float(os.environ.get('DT', 0.005))
    phi = jnp.array(hopfion(N, h, 6.0)); v = jnp.zeros_like(phi)
    step = jax.jit(lambda p, v, a, g: (lambda v1: (p + dt * v1, v1))(v * g + 0.5 * dt * a))
    @jax.jit
    def kdk(p, v, a, g):
        v = v * g + 0.5 * dt * a; p = p + dt * v; a = -0.5 * gE(p) / h**3; v = (v + 0.5 * dt * a) * g
        return p, v, a
    a = acc_of(phi); t0 = time.time()
    g = np.exp(-0.5 * dt)   # friction G0=1 pendant la relaxation
    for i in range(int(20 / dt)): phi, v, a = kdk(phi, v, a, g)
    print('relaxé : E=', round(float(E(phi)), 3), 'Q=', round(Qlat(phi), 4), 'tmps', round(time.time() - t0), flush=True)
    NB_ = int(os.environ.get('NBIG', 0))
    if NB_ > N:   # plongement du nœud relaxé dans une boîte plus grande (vide zeta=(1,0) autour)
        big = np.zeros((4, NB_, NB_, NB_), dtype=np.float32); big[0] = 1.0; o = (NB_ - N) // 2
        big[:, o:o+N, o:o+N, o:o+N] = np.array(phi); phi = jnp.array(big); N = NB_
        x = (np.arange(N) - N / 2 + 0.5) * h; R = np.sqrt(sum(a**2 for a in np.meshgrid(x, x, x, indexing='ij')))
        gE = jax.jit(jax.grad(E)); acc_of = jax.jit(lambda p_: -0.5 * gE(p_) / h**3)
        kdk = jax.jit(lambda p, v, a, g: (lambda v1: (lambda p1: (lambda a1: (p1, (v1 + 0.5 * dt * a1) * g, a1))(-0.5 * gE(p1) / h**3))(p + dt * v1))(v * g + 0.5 * dt * a))
        print('embarqué dans N =', N, flush=True)
    phi_s = phi
    w = jnp.exp(-(R / Rw)**4)[None]
    v = om * Imul(phi) * w; a = acc_of(phi)
    msk = lambda r: jnp.asarray(R < r)
    out = []
    for i in range(int(TEND / dt) + 1):
        if i % int(2.0 / dt) == 0:
            q = qdens(phi, v); Jt = float(h**3 * jnp.sum(q)); Jb = [float(h**3 * jnp.sum(q * msk(r))) for r in (6, 9, 11, 15, 19)]
            g_, f_, p_ = energy_density(phi, h, mu, cF, d2); e = h**3 * (g_ + f_ + p_)
            Ek = float(h**3 * jnp.sum(jnp.sum(v**2, axis=0)))
            r2 = jnp.sum(phi**2, axis=0)
            rec = dict(t=round(i * dt, 2), Q=round(Qlat(phi), 3), J=round(Jt, 3), J6=round(Jb[0], 3), J9=round(Jb[1], 3), J11=round(Jb[2], 3), J15=round(Jb[3], 3), J19=round(Jb[4], 3),
                       Etot=round(float(jnp.sum(e)) + Ek, 3), Eb9=round(float(jnp.sum(e * msk(9))), 3), Ek=round(Ek, 3),
                       rho_min=round(float(jnp.sqrt(r2.min())), 3), rho_max=round(float(jnp.sqrt(r2.max())), 3),
                       dist_statique=round(float(jnp.sqrt(jnp.sum((phi - phi_s)**2) * h**3)), 3))
            print(json.dumps(rec), flush=True)
        phi, v, a = kdk(phi, v, a, 1.0)
