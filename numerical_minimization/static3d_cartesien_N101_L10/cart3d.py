"""PEW2 statique, controle INDEPENDANT sur grille cartesienne 3D (pas d'axe, pas de coeur epingle).
E = int [ (f0^2/4) |dn|^2 + (1/(4 e0^2)) sum_{ij} |d_i n x d_j n|^2 ] d3x   (f0=e0=1)  ; sans facteur 2pi => comparer a 2*pi*G.
Sommets sur grille N^3 dans [-L,L]^3 (N impair : l'axe passe par des sommets). Bord : n = +e3 (vide). Variables : v in R^3 (n = v/|v|).
Schema retenu (le gradient 'moyenne des 4 aretes paralleles' produisait des modes quasi nuls et un effondrement artificiel) : differences avant/arriere par direction aux sommets, densite d'energie moyennee sur les 8 combinaisons (comme le code toroidal)."""
import sys, time, numpy as np, jax, jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
jax.config.update("jax_enable_x64", True)

def energy_cart(vint, N, h, bnd=None):
    n_int = vint.reshape(N-2, N-2, N-2, 3)
    v = (jnp.zeros((N, N, N, 3)).at[..., 2].set(1.0)) if bnd is None else jnp.asarray(bnd)
    v = v.at[1:-1, 1:-1, 1:-1, :].set(n_int)
    n = v/jnp.linalg.norm(v, axis=-1, keepdims=True)
    c = n[1:-1, 1:-1, 1:-1]
    # differences avant/arriere par direction aux sommets interieurs ; densite moyennee sur les 8 combinaisons (schema 'staggered', sans mode damier)
    d = [((n[2:, 1:-1, 1:-1]-c)/h, (c-n[:-2, 1:-1, 1:-1])/h),
         ((n[1:-1, 2:, 1:-1]-c)/h, (c-n[1:-1, :-2, 1:-1])/h),
         ((n[1:-1, 1:-1, 2:]-c)/h, (c-n[1:-1, 1:-1, :-2])/h)]
    E = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for g in (0, 1):
                D = (d[0][a], d[1][b], d[2][g])
                kin = 0.25*sum(jnp.sum(x*x, -1) for x in D)
                q = 0.0
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            cr = jnp.cross(D[i], D[j]); q = q + jnp.sum(cr*cr, -1)
                E = E + kin + 0.25*q
    return jnp.sum(E/8.0)*h**3

def init_from_toroidal(f2, d2, EM, a, N, L):
    Ne, Nx = f2.shape
    deta = EM/Ne; dxi = 2*np.pi/Nx
    eta = (np.arange(Ne)+0.5)*deta; xi = (np.arange(Nx)+0.5)*dxi
    eta_e = np.concatenate([[0.0], eta]); xi_e = np.concatenate([[xi[-1]-2*np.pi], xi, [xi[0]+2*np.pi]])
    fe = np.concatenate([np.zeros((1, Nx)), f2], 0); de = np.concatenate([d2[:1], d2], 0)
    fe = np.concatenate([fe[:, -1:], fe, fe[:, :1]], 1); de = np.concatenate([de[:, -1:], de, de[:, :1]], 1)
    fI = RegularGridInterpolator((eta_e, xi_e), fe); dI = RegularGridInterpolator((eta_e, xi_e), de)
    x = np.linspace(-L, L, N); X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    rho = np.hypot(X, Y); r2 = rho**2 + Z**2
    ETA = np.arctanh(np.clip(2*a*rho/(r2 + a*a), 0, 1-1e-15)); XI = np.mod(np.arctan2(2*a*Z, r2 - a*a), 2*np.pi)
    pts = np.stack([np.clip(ETA, 0, EM), XI], -1)
    f = fI(pts); dch = dI(pts); phi = np.arctan2(Y, X)
    th = XI + dch + phi
    n = np.stack([np.sin(f)*np.cos(th), np.sin(f)*np.sin(th), np.cos(f)], -1)
    return n

if __name__ == '__main__':
    N, L, MAXIT, tag = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    a = float(sys.argv[5]) if len(sys.argv) > 5 else 1.743
    h = 2*L/(N-1)
    f2 = np.load('f2d_r128.npy'); d2 = np.load('d2d_r128.npy')
    n0 = init_from_toroidal(f2, d2, 12.0, a, N, L)
    x0 = n0[1:-1, 1:-1, 1:-1, :].ravel()
    _nz = float(__import__('os').environ.get('NOISE','0'))
    if _nz > 0:
        _r = np.random.RandomState(1); x0 = x0 + _nz*_r.randn(*x0.shape)
    vg = jax.jit(jax.value_and_grad(lambda x: energy_cart(x, N, h, n0)))
    E0, g0 = vg(jnp.array(x0)); print(f'N={N} L={L} h={h:.4f} E(depart)={float(E0):.5f}  2pi*G(toro,N128)={2*np.pi*30.800869/ (1):.4f}?? (a=1.743 fixe)  rms g={float(jnp.sqrt(jnp.mean(g0**2))):.2e}', flush=True)
    from scipy.optimize import minimize
    t = time.time(); x = x0; it = 0
    CH=int(__import__("os").environ.get("CH","2000"))
    for ch in range(MAXIT//CH):
        r = minimize(lambda y: tuple(np.asarray(v, float) for v in vg(jnp.array(y))), x, jac=True, method='L-BFGS-B',
                     options=dict(maxiter=CH, maxfun=2*CH, ftol=1e-15, gtol=1e-11, maxcor=30))
        x = r.x; it += r.nit
        print(f'  it={it} E={r.fun:.6f} rms={np.sqrt(np.mean(r.jac**2)):.2e} t={time.time()-t:.0f}s', flush=True)
        np.save(f'cart_{tag}.npy', x)
        if 'CONVERGENCE' in str(r.message): break
