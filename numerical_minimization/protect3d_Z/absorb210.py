"""part210 -- absorbeur compatible Gauss pour lagcov (jauge temporelle a0=0).
Principe : la contrainte de Gauss est G = dK/da0 (lineaire en v). On amortit les vitesses (taux sigma(r), eponge spherique),
puis on re-projette sur G=0 : a0* = argmin_a0 K(q,v,a0) (CG sur un scalaire N^3), v' = v + Gamma(a0*), ou Gamma est le deplacement
de vitesse le long de l'orbite de jauge. Dissipation ne violant pas la contrainte de Gauss. Test : python absorb210.py"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

def sponge(N, h, spr0, sprb, spsig):
    x = (np.arange(N) - N / 2 + 0.5) * h
    r = np.sqrt(x[:, None, None]**2 + x[None, :, None]**2 + x[None, None, :]**2)
    return jnp.asarray(spsig * np.clip((r - spr0) / (sprb - spr0), 0, 1)**2 + 50.0 * (r > sprb))

def make(L, N, h, sgn=1.0):
    kin = L['kin']
    def Gam(q, a0):   # deplacement de vitesse par une transformation de jauge de parametre a0
        phi = q[0]; z = jnp.zeros_like(phi[0])
        gphi = jnp.stack([z, z, a0 * phi[3], -a0 * phi[2]])
        gw = jnp.stack([-(jnp.roll(a0, -1, axis=a) - a0) for a in range(3)])
        return (sgn * gphi, sgn * gw)
    def project(q, v):
        a00 = jnp.zeros((N, N, N))
        f = lambda a0: kin(q[0], q[1], v[0], v[1], a0)
        g = jax.grad(f)(a00)
        Hv = lambda x: jax.jvp(jax.grad(f), (a00,), (x,))[1]
        a0, _ = jax.scipy.sparse.linalg.cg(Hv, -g, tol=1e-12, maxiter=500)
        d = Gam(q, a0)
        return (v[0] + d[0], v[1] + d[1])
    def damp(v, sig, dt):
        fac = jnp.exp(-sig * dt)
        return (v[0] * fac, v[1] * fac)
    return project, damp, Gam

if __name__ == '__main__':
    import lagcov
    N, h = 16, 0.5
    L = lagcov.build(N, h, nu=0.12, g2=0.02, KF=1.0)
    key = jax.random.PRNGKey(0); k = jax.random.split(key, 4)
    phi = jnp.stack([jnp.ones((N, N, N)), 0*jnp.ones((N, N, N)), 0*jnp.ones((N, N, N)), 0*jnp.ones((N, N, N))]) + 0.15 * jax.random.normal(k[0], (4, N, N, N))
    phi = phi / jnp.sqrt(jnp.sum(phi**2, 0))
    th = 0.05 * jax.random.normal(k[1], (3, N, N, N)); q = (phi, th)
    v = (0.1 * jax.random.normal(k[2], (4, N, N, N)), 0.1 * jax.random.normal(k[3], (3, N, N, N)))
    a0t = 0.07 * jax.random.normal(jax.random.PRNGKey(5), (N, N, N))
    for sgn in (1.0, -1.0):   # verifie le sens du deplacement de jauge : kin(v,a0) == kin(v+Gamma(a0),0)
        project, damp, Gam = make(L, N, h, sgn)
        d = Gam(q, a0t); lhs = L['kin'](q[0], q[1], v[0], v[1], a0t); rhs = L['kin'](q[0], q[1], v[0] + d[0], v[1] + d[1], jnp.zeros((N, N, N)))
        print('sgn', sgn, 'kin(v,a0)', float(lhs), 'kin(v+Gam,0)', float(rhs), 'ecart rel', abs(float(lhs - rhs)) / abs(float(lhs)))
