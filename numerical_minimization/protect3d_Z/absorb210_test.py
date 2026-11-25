import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import lagcov, absorb210
N, h = 16, 0.5
for KF in (0.0, 1.0):
    L = lagcov.build(N, h, nu=0.12, g2=0.02, KF=KF); project, damp, Gam = absorb210.make(L, N, h, 1.0)
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    phi = jnp.concatenate([jnp.ones((1, N, N, N)), 0.15 * jax.random.normal(k[0], (3, N, N, N))]); phi = phi / jnp.sqrt(jnp.sum(phi**2, 0))
    th = 0.05 * jax.random.normal(k[1], (3, N, N, N)); q = (phi, th)
    v = (0.1 * jax.random.normal(k[2], (4, N, N, N)), 0.1 * jax.random.normal(k[3], (3, N, N, N)))
    nG = lambda v: float(jnp.max(jnp.abs(L['gauss'](q, v))))
    K = lambda v: float(L['hamil'](q, v) - L['hamil'](q, jax.tree_util.tree_map(jnp.zeros_like, v)))
    v = project(q, v); print(f'KF={KF}: apres projection initiale  max|G|={nG(v):.2e}  K={K(v):.4f}')
    sig = absorb210.sponge(N, h, 1.5, 3.5, 0.5); v2 = damp(v, sig, 0.2)
    print(f'   apres amortissement seul   max|G|={nG(v2):.2e}  K={K(v2):.4f}')
    v3 = project(q, v2); print(f'   apres re-projection        max|G|={nG(v3):.2e}  K={K(v3):.4f}')
