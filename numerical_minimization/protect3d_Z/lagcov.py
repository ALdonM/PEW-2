"""part202 -- Lagrangien COVARIANT complet du modele Z (L1 + jauge U(1) sur Z2) avec terme de Faddeev temporel jauge et champ A0 (pour la loi de Gauss).
L = K(phi, th, v, w, a0) - E(phi, th),  v = phi_dot, w = theta_dot (variables de lien), a0 = composante temporelle du champ de jauge (site).
K = h^3 sum_x [ |Z1_dot|^2 + |Z2_dot - i a0 Z2|^2 + (0,5/g2) sum_a E_a^2 + KF cF * 0,5 sum_a (Om0a_avant^2 + Om0a_arriere^2) ]
E_a = (w_a - (a0(x+a) - a0(x))) / h ;  Om0a = n . (D0 n x D_a n) + (n3 - 1) E_a ;  D0 n = n_dot - a0 (e3 x n).
E(phi, th) = energie de jauge.py (copie verbatim). Meme convention : L = |phi_dot|^2 - E  (c = 1 dans tous les secteurs si KF = 1).
Les equations du mouvement sont tirees de L par autodiff (pas ecrites a la main) ; M a = rhs est resolu par gradient conjugue sans matrice."""
import os, numpy as np, jax, jax.numpy as jnp
from zlat import cross

def build(N, h=0.5, mu=0.375, e0=0.25, d2=1e-3, nu=0.12, g2=0.02, KF=1.0, GUARD=1e-6):
    cF = 1 / (2 * e0**2)
    sq = lambda z: z.real**2 + z.imag**2
    def rot(n, al):
        c, s = jnp.cos(al), jnp.sin(al)
        return jnp.stack([n[0] * c - n[1] * s, n[0] * s + n[1] * c, n[2]])
    def nv(phi):
        r2 = jnp.sum(phi**2, axis=0); a_, b_, c_, d_ = phi
        return jnp.stack([2 * (a_ * c_ + b_ * d_), 2 * (a_ * d_ - b_ * c_), a_**2 + b_**2 - c_**2 - d_**2]) / (r2 + GUARD)
    def plaq(th, i, j):
        return (th[i] + jnp.roll(th[j], -1, axis=i) - jnp.roll(th[i], -1, axis=j) - th[j]) / h**2
    def energy(phi, th):   # copie de jauge.energy
        c1, c2 = phi[0] + 1j * phi[1], phi[2] + 1j * phi[3]
        sh = lambda f, a: jnp.roll(f, -1, axis=a)
        grad = 0.0
        for a in range(3):
            grad = grad + sq(sh(c1, a) - c1) / h**2 + sq(jnp.exp(-1j * th[a]) * sh(c2, a) - c2) / h**2
        r2 = jnp.sum(phi**2, axis=0); n = nv(phi)
        Df = [(rot(jnp.roll(n, -1, axis=1 + a), -th[a]) - n) / h for a in range(3)]
        Db = [(n - rot(jnp.roll(n, 1, axis=1 + a), jnp.roll(th[a], 1, axis=a))) / h for a in range(3)]
        fad = 0.0; B2 = 0.0
        for (i, j) in ((0, 1), (0, 2), (1, 2)):
            Fp = plaq(th, i, j)
            for di, Di in ((0, Df[i]), (1, Db[i])):
                for dj, Dj in ((0, Df[j]), (1, Db[j])):
                    F = jnp.roll(Fp, (di, dj), axis=(i, j)); n3 = n[2]
                    oi = [-di, -di + 1]; oj = [-dj, -dj + 1]
                    n3m = sum(jnp.roll(n3, (-a_, -b_), axis=(i, j)) for a_ in oi for b_ in oj) / 4
                    Om = jnp.sum(n * cross(Di, Dj), axis=0) + (n3m - 1.0) * F
                    fad = fad + 0.25 * Om**2
            B2 = B2 + Fp**2
        pot = mu**2 * (r2 + 1.0 / (r2 + d2) - 2.0) + 2 * nu**2 * sq(c2)
        return h**3 * jnp.sum(grad + cF * fad + pot + ((0.5 / g2) * B2 if g2 > 0 else 0.0))
    def kin(phi, th, v, w, a0):
        T = jnp.sum(v[0]**2 + v[1]**2 + (v[2] + a0 * phi[3])**2 + (v[3] - a0 * phi[2])**2)
        E = [(w[a] - (jnp.roll(a0, -1, axis=a) - a0)) / h for a in range(3)]
        if g2 > 0: T = T + (0.5 / g2) * sum(jnp.sum(e**2) for e in E)
        if KF != 0.0:
            n, ndot = jax.jvp(nv, (phi,), (v,))
            D0n = ndot - a0 * jnp.stack([-n[1], n[0], 0 * n[2]])
            n3 = n[2]
            for a in range(3):
                Df = (rot(jnp.roll(n, -1, axis=1 + a), -th[a]) - n) / h
                Db = (n - rot(jnp.roll(n, 1, axis=1 + a), jnp.roll(th[a], 1, axis=a))) / h
                Of = jnp.sum(n * cross(D0n, Df), axis=0) + (0.5 * (n3 + jnp.roll(n3, -1, axis=a)) - 1.0) * E[a]
                Ob = jnp.sum(n * cross(D0n, Db), axis=0) + (0.5 * (n3 + jnp.roll(n3, 1, axis=a)) - 1.0) * jnp.roll(E[a], 1, axis=a)
                T = T + KF * cF * 0.5 * jnp.sum(Of**2 + Ob**2)
        return h**3 * T
    def lagr(q, v, a0): return kin(q[0], q[1], v[0], v[1], a0) - energy(q[0], q[1])
    a0z = jnp.zeros((N, N, N))
    Kz = lambda q, v: kin(q[0], q[1], v[0], v[1], a0z)
    Ez = lambda q: energy(q[0], q[1])
    pfun = jax.grad(Kz, argnums=1)                  # p = dK/dv (dans a0 = 0)
    def acc(q, v, x0=None, tol=1e-12, maxit=400):
        dKq = jax.grad(Kz, 0)(q, v); dEq = jax.grad(Ez)(q)
        _, jp = jax.jvp(lambda qq: pfun(qq, v), (q,), (v,))
        rhs = jax.tree_util.tree_map(lambda a, b, c: a - b - c, dKq, dEq, jp)
        Mv = lambda x: jax.jvp(lambda vv: pfun(q, vv), (v,), (x,))[1]
        a, _ = jax.scipy.sparse.linalg.cg(Mv, rhs, x0=x0, tol=tol, maxiter=maxit)
        return a
    gauss = lambda q, v: jax.grad(lambda a0: kin(q[0], q[1], v[0], v[1], a0))(a0z)   # G = dL/da0 en a0=0 (conservee si L est invariant de jauge)
    hamil = lambda q, v: Kz(q, v) + Ez(q)           # K quadratique en v : H = K + E
    return dict(energy=energy, kin=kin, lagr=lagr, acc=acc, gauss=gauss, hamil=hamil, nv=nv, rot=rot, pfun=pfun, cF=cF)
