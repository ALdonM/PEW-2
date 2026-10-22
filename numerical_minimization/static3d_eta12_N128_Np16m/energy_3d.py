"""PEW 3D statique : extension de energy_native_staggered.py avec dependance en phi.
Champs f(eta,xi,phi), dchi(eta,xi,phi) ; theta = m*xi + n*phi + dchi (m=n=1 : Q_H = 1).
Meme metrique toroidale, meme schema 'staggered' etendu aux 3 directions (8 densites avant/arriere moyennees).
Normalisation : E3D = (1/Np) * sum sqrt(g) * dens * deta * dxi  -> pour un champ independant de phi, E3D == E2D (energy_native_staggered).
Conditions aux bords : eta=0 : f miroir impair (f=0), dchi Neumann ; eta=eta_max : f=f_core, dchi Neumann ; xi, phi periodiques.
"""
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)

def build_grid3(Ne, Nx, eta_max):
    deta = eta_max/Ne; dxi = 2*np.pi/Nx
    eta = (np.arange(Ne)+0.5)*deta; xi = (np.arange(Nx)+0.5)*dxi
    ETA, XI = np.meshgrid(eta, xi, indexing='ij')
    return eta, xi, deta, dxi, jnp.array(ETA)[:, :, None], jnp.array(XI)[:, :, None]

def total_energy_3d(params, ETA, XI, deta, dxi, ar, f0, e0, f_core, Ne, Nx, Np, m=1, n=1):
    V = Ne*Nx*Np
    f = params[:V].reshape(Ne, Nx, Np); c = params[V:].reshape(Ne, Nx, Np)
    f_ext = jnp.concatenate([-f[0:1], f, 2*f_core - f[-1:]], axis=0)
    c_ext = jnp.concatenate([c[0:1], c, c[-1:]], axis=0)
    dphi = 2*np.pi/Np
    fe = ((f_ext[2:]-f_ext[1:-1])/deta, (f_ext[1:-1]-f_ext[:-2])/deta)
    ce = ((c_ext[2:]-c_ext[1:-1])/deta, (c_ext[1:-1]-c_ext[:-2])/deta)
    fx = ((jnp.roll(f, -1, 1)-f)/dxi, (f-jnp.roll(f, 1, 1))/dxi)
    cx = ((jnp.roll(c, -1, 1)-c)/dxi, (c-jnp.roll(c, 1, 1))/dxi)
    fp = ((jnp.roll(f, -1, 2)-f)/dphi, (f-jnp.roll(f, 1, 2))/dphi)
    cp = ((jnp.roll(c, -1, 2)-c)/dphi, (c-jnp.roll(c, 1, 2))/dphi)
    Om = ar/(jnp.cosh(ETA)-jnp.cos(XI)); sqrtg = Om**3*jnp.sinh(ETA)
    gee = Om**2; gxx = Om**2; gpp = Om**2*jnp.sinh(ETA)**2
    s = jnp.sin(f)
    def dens(a, b, d):
        fE, fX, fP = fe[a], fx[b], fp[d]
        tE, tX, tP = ce[a], m+cx[b], n+cp[d]
        kin = (f0**2/4.0)*(fE**2/gee+fX**2/gxx+fP**2/gpp + s**2*(tE**2/gee+tX**2/gxx+tP**2/gpp))
        Hex = s*(fE*tX-fX*tE); Hep = s*(fE*tP-fP*tE); Hxp = s*(fX*tP-fP*tX)
        HH = 2.0*(Hex**2/(gee*gxx)+Hep**2/(gee*gpp)+Hxp**2/(gxx*gpp))
        return kin+HH/(4.0*e0**2)
    E = 0.0
    for a in (0, 1):
        for b in (0, 1):
            for d in (0, 1):
                E = E + dens(a, b, d)
    return jnp.sum(sqrtg*E/8.0)*deta*dxi/Np

energy_and_grad_3d = jax.jit(jax.value_and_grad(total_energy_3d), static_argnums=(9, 10, 11, 12, 13))
