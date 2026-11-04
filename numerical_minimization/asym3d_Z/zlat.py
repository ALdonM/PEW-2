"""
part165 -- Réseau 3D périodique du modèle Z (convention Z, alpha' = 1).
Champ : phi (4, N, N, N) réel, Z1 = phi0 + i phi1, Z2 = phi2 + i phi3, rho = |Z|, zeta = Z/rho.
Densité d'énergie statique (mêmes coefficients que part122) :
  e = |grad Z|^2 + cF * sum_{i<j} (n . D_i n x D_j n)^2 + mu^2 (rho^2 + 1/(rho^2 + d2) - 2)
  cF = 1/(2 e0^2) ; d2 = régulateur [D] (évite 1/0 sur réseau ; rho ne s'annule pas génériquement en 3D).
Dynamique [D] : L = sum |dZ/dt|^2 - E_stat  =>  phi'' = -(1/2) dE/dphi - gamma(t) phi'.
  (le terme de Faddeev temporel -- inertie quartique -- est omis : il ne change ni l'énergie statique,
   ni les minima, ni la conservation de l'énergie ; il change la cinétique des ondes.)
Charge de Hopf de n = degré de zeta : S^3 -> S^3,  Q = (1/2 pi^2) int det(zeta, d_x zeta, d_y zeta, d_z zeta).
"""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", False)


def fwd(f, ax, h):
    return (jnp.roll(f, -1, axis=ax) - f) / h


def cen(f, ax, h):
    return (jnp.roll(f, -1, axis=ax) - jnp.roll(f, 1, axis=ax)) / (2 * h)


def nvec(phi):
    r2 = jnp.sum(phi**2, axis=0)
    a, b, c, d = phi
    # Z1* Z2 = (a - ib)(c + id) = (ac + bd) + i(ad - bc)
    n1 = 2 * (a * c + b * d)
    n2 = 2 * (a * d - b * c)
    n3 = a**2 + b**2 - c**2 - d**2
    return jnp.stack([n1, n2, n3]) / r2


def cross(u, v):
    return jnp.stack([u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]])


def energy_density(phi, h, mu, cF, d2):
    if mu < 0:   # amplitude gelée (rho = 1) : contrôle de résolution (modèle de Faddeev pur + phase chi_g)
        phi = phi / jnp.sqrt(jnp.sum(phi**2, axis=0))
    grad2 = sum(jnp.sum(fwd(phi, ax, h)**2, axis=0) for ax in (1, 2, 3))
    n = nvec(phi)
    # schéma anti-damier : moyenne de F^2 sur les 4 combinaisons avant/arrière (comme part121/energy_jax6)
    Df = [fwd(n, ax, h) for ax in (1, 2, 3)]
    Db = [(n - jnp.roll(n, 1, axis=ax)) / h for ax in (1, 2, 3)]
    fad = 0.0
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        for Di in (Df[i], Db[i]):
            for Dj in (Df[j], Db[j]):
                fad = fad + 0.25 * jnp.sum(n * cross(Di, Dj), axis=0)**2
    r2 = jnp.sum(phi**2, axis=0)
    pot = mu**2 * (r2 + 1.0 / (r2 + d2) - 2.0) if mu >= 0 else 0.0 * r2
    return grad2, cF * fad, pot


def make(h, mu, cF, d2):
    def E(phi):
        g, f, p = energy_density(phi, h, mu, cF, d2)
        return h**3 * jnp.sum(g + f + p)
    gE = jax.jit(jax.grad(E))
    Ej = jax.jit(E)

    @jax.jit
    def parts(phi):
        g, f, p = energy_density(phi, h, mu, cF, d2)
        return h**3 * jnp.array([jnp.sum(g), jnp.sum(f), jnp.sum(p)])

    @jax.jit
    def qdens(phi):
        z = phi / jnp.sqrt(jnp.sum(phi**2, axis=0))
        M = jnp.stack([z] + [cen(z, ax, h) for ax in (1, 2, 3)])  # (4 lignes, 4 comps, ...)
        M = jnp.moveaxis(M, (0, 1), (-2, -1))
        return jnp.linalg.det(M) / (2 * np.pi**2)

    return Ej, gE, parts, qdens


def hopfion(N, h, R, x0=(0, 0, 0)):
    """Hérisson S^3 de degré 1 (charge de Hopf 1 pour n), profil f = pi exp(-(r/R)^2), vide zeta = (1,0) au bord :
    Z1 = cos f + i sin f u_z,  Z2 = sin f (u_x + i u_y),  u = x/r.  (Périodique sans discontinuité.)"""
    x = (np.arange(N) - N / 2 + 0.5) * h
    X, Y, Zc = np.meshgrid(x - x0[0], x - x0[1], x - x0[2], indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Zc**2) + 1e-12
    f = np.pi * np.exp(-(r / R)**2)
    s = np.sin(f) / r
    return np.stack([np.cos(f), s * Zc, s * X, s * Y]).astype(np.float32)


def hopfion_stereo(N, h, R, x0=(0, 0, 0)):
    """Application de degré 1 R^3 -> S^3 (u = x/R) : zeta1 = (2u_z + i(u^2-1))/(u^2+1), zeta2 = 2(u_x + i u_y)/(u^2+1)."""
    x = (np.arange(N) - N / 2 + 0.5) * h
    X, Y, Zc = np.meshgrid(x - x0[0], x - x0[1], x - x0[2], indexing='ij')
    ux, uy, uz = X / R, Y / R, Zc / R
    u2 = ux**2 + uy**2 + uz**2
    phi = np.stack([2 * uz, u2 - 1, 2 * ux, 2 * uy]) / (u2 + 1)
    return phi.astype(np.float32)


def random_field(N, h, xi0, rms_rho2, seed):
    rng = np.random.default_rng(seed)
    k = 2 * np.pi * np.fft.fftfreq(N, d=h)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    filt = np.exp(-0.25 * (KX**2 + KY**2 + KZ**2) * xi0**2)
    phi = np.empty((4, N, N, N))
    for a in range(4):
        w = rng.standard_normal((N, N, N))
        f = np.real(np.fft.ifftn(np.fft.fftn(w) * filt))
        phi[a] = f / f.std() * np.sqrt(rms_rho2 / 4)
    return phi.astype(np.float32)
