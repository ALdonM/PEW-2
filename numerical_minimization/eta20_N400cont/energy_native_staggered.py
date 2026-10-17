"""
Fusion des deux corrections deja validees separement dans le Projet PEW :

  (1) reformulation "native-f" (AUDIT_NOTE_addendum4_reformulation_native_f.md,
      2026-08-09) : substituer Atilde=tan(f/2) ANALYTIQUEMENT dans le
      Lagrangien avant toute derivation, ce qui donne des expressions
      entierement bornees en sin(f), cos(f) -- plus aucun facteur
      (1+Atilde^2)^n explosant pres de f_core.

  (2) schema "staggered" anti-damier (energy_jax5.py, deja la reference
      validee pour la chaine JAX principale) : chaque derivee (eta et xi)
      est evaluee separement en avant et en arriere, puis les 4 densites
      resultantes (avant/avant, avant/arriere, arriere/avant, arriere/
      arriere) sont moyennees -- ceci supprime l'angle mort au damier du
      stencil centre pur.

Ces deux corrections n'avaient jamais ete combinees : le solveur natif
(mg_solver_native.py / relax_native_f.py) a (1) mais pas (2) ; la chaine
JAX (energy_jax5/6.py) a (2) mais pas (1) (elle reste en variable Atilde).

Meme convention de grille que relax_sor.build_grid / energy_jax5.py
(grille centree sur les cellules, deta=eta_max/N_eta, eta=(i+0.5)*deta),
memes conditions au bord (Dirichlet miroir f=0 en eta=0 / f=f_core en
eta=eta_max, Neumann/copie en dchi, periodique en xi) -- comparaison
directe possible avec les resultats deja publies.

Derivation de la densite native (verifiee a la main, coherente avec
AUDIT_NOTE_addendum4 et noether_theta_current.py) :
  A=tan(f/2) => 1+A^2 = 1/cos^2(f/2)
  A/(1+A^2)      = (1/2) sin f
  A_eta/(1+A^2)  = (1/2) f_eta      (car dA/df = (1/2) sec^2(f/2) = (1/2)(1+A^2))
d'ou:
  kin_native = (f0^2/4) [ f_eta^2/g_ee + f_xi^2/g_xx
                          + sin^2(f) (theta_eta^2/g_ee + theta_xi^2/g_xx + 1/g_pp) ]
  H_ex = sin(f) (f_eta*theta_xi - f_xi*theta_eta)
  H_ep = sin(f) f_eta
  H_xp = sin(f) f_xi
  HH   = 2 sin^2(f) [ (f_eta theta_xi - f_xi theta_eta)^2/(g_ee g_xx)
                        + f_eta^2/(g_ee g_pp) + f_xi^2/(g_xx g_pp) ]

Session 2026-09-06 : gradient JAX verifie par differences finies
(relerr max ~3e-7 sur points aleatoires). Minimise via scipy L-BFGS-B
(pas Newton-LM) depuis rampe lineaire f: 0->f_core, dchi=0, f_core=pi,
eta_max=3.0 fixe. Resultat (voir AUDIT_NOTE_session_20260906_native_
staggered_independant_pas_de_plateau.md pour le detail) : convergence
propre a precision quasi-machine a N=30, 60 ET 90 (|grad|_rms descend a
~1e-8, ratio derniere-ligne/mediane ~1, AUCUN plateau, AUCUNE trace des
deux zones "ligne externe"/"bosse symetrique" documentees pour
mg_solver_native.py). E(N=30)=32.845906, E(N=60)=32.823521,
E(N=90)=32.815523 -- convergence en N propre (increments decroissants).
"""
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)


def build_grid(N_eta, N_xi, eta_max):
    deta = eta_max / N_eta
    eta = (np.arange(N_eta) + 0.5) * deta
    dxi = 2 * np.pi / N_xi
    xi = (np.arange(N_xi) + 0.5) * dxi
    ETA, XI = np.meshgrid(eta, xi, indexing='ij')
    return eta, xi, deta, dxi, jnp.array(ETA), jnp.array(XI)


def total_energy_native_staggered(params, ETA, XI, deta, dxi, ar, f0, e0, f_core, N_eta, N_xi):
    f = params[:N_eta * N_xi].reshape(N_eta, N_xi)
    dchi = params[N_eta * N_xi:].reshape(N_eta, N_xi)

    # ghosts eta : meme convention que relax_sor.extend_field_dirichlet /
    # energy_jax5 (miroir f=0 en eta=0, miroir f=f_core en eta=eta_max)
    left_f = -f[0:1, :]
    right_f = 2 * f_core - f[-1:, :]
    f_ext = jnp.concatenate([left_f, f, right_f], axis=0)

    left_c = dchi[0:1, :]
    right_c = dchi[-1:, :]
    c_ext = jnp.concatenate([left_c, dchi, right_c], axis=0)

    f_eta_fwd = (f_ext[2:, :] - f_ext[1:-1, :]) / deta
    f_eta_bwd = (f_ext[1:-1, :] - f_ext[:-2, :]) / deta
    c_eta_fwd = (c_ext[2:, :] - c_ext[1:-1, :]) / deta
    c_eta_bwd = (c_ext[1:-1, :] - c_ext[:-2, :]) / deta

    f_xi_fwd = (jnp.roll(f, -1, axis=1) - f) / dxi
    f_xi_bwd = (f - jnp.roll(f, 1, axis=1)) / dxi
    c_xi_fwd = (jnp.roll(dchi, -1, axis=1) - dchi) / dxi
    c_xi_bwd = (dchi - jnp.roll(dchi, 1, axis=1)) / dxi

    Omega = ar / (jnp.cosh(ETA) - jnp.cos(XI))
    sqrtg = Omega**3 * jnp.sinh(ETA)
    g_ee = Omega**2
    g_xx = Omega**2
    g_pp = Omega**2 * jnp.sinh(ETA)**2

    sinf = jnp.sin(f)

    def density(f_eta, f_xi, c_eta, c_xi):
        theta_eta = c_eta
        theta_xi = 1.0 + c_xi
        kin = (f0**2 / 4.0) * (
            f_eta**2 / g_ee + f_xi**2 / g_xx
            + sinf**2 * (theta_eta**2 / g_ee + theta_xi**2 / g_xx + 1.0 / g_pp)
        )
        H_ex = sinf * (f_eta * theta_xi - f_xi * theta_eta)
        H_ep = sinf * f_eta
        H_xp = sinf * f_xi
        HH = 2.0 * (H_ex**2 / (g_ee * g_xx) + H_ep**2 / (g_ee * g_pp) + H_xp**2 / (g_xx * g_pp))
        return kin + HH / (4.0 * e0**2)

    d_ff = density(f_eta_fwd, f_xi_fwd, c_eta_fwd, c_xi_fwd)
    d_fb = density(f_eta_fwd, f_xi_bwd, c_eta_fwd, c_xi_bwd)
    d_bf = density(f_eta_bwd, f_xi_fwd, c_eta_bwd, c_xi_fwd)
    d_bb = density(f_eta_bwd, f_xi_bwd, c_eta_bwd, c_xi_bwd)
    Edens = 0.25 * (d_ff + d_fb + d_bf + d_bb)

    return jnp.sum(sqrtg * Edens) * deta * dxi


energy_and_grad_native_staggered = jax.jit(
    jax.value_and_grad(total_energy_native_staggered), static_argnums=(9, 10)
)


def total_energy_native_centered(params, ETA, XI, deta, dxi, ar, f0, e0, f_core, N_eta, N_xi):
    """Meme densite native (bornee), mais difference centree pure (pas de
    moyenne avant/arriere) -- pour comparaison directe avec
    relax_native_f.energy_reference_native (numpy), sert de test de
    coherence croisee independant de energy_jax5."""
    f = params[:N_eta * N_xi].reshape(N_eta, N_xi)
    dchi = params[N_eta * N_xi:].reshape(N_eta, N_xi)

    left_f = -f[0:1, :]
    right_f = 2 * f_core - f[-1:, :]
    f_ext = jnp.concatenate([left_f, f, right_f], axis=0)
    left_c = dchi[0:1, :]
    right_c = dchi[-1:, :]
    c_ext = jnp.concatenate([left_c, dchi, right_c], axis=0)

    f_eta = (f_ext[2:, :] - f_ext[:-2, :]) / (2 * deta)
    c_eta = (c_ext[2:, :] - c_ext[:-2, :]) / (2 * deta)
    f_xi = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dxi)
    c_xi = (jnp.roll(dchi, -1, axis=1) - jnp.roll(dchi, 1, axis=1)) / (2 * dxi)

    Omega = ar / (jnp.cosh(ETA) - jnp.cos(XI))
    sqrtg = Omega**3 * jnp.sinh(ETA)
    g_ee = Omega**2
    g_xx = Omega**2
    g_pp = Omega**2 * jnp.sinh(ETA)**2
    sinf = jnp.sin(f)
    theta_eta = c_eta
    theta_xi = 1.0 + c_xi
    kin = (f0**2 / 4.0) * (
        f_eta**2 / g_ee + f_xi**2 / g_xx
        + sinf**2 * (theta_eta**2 / g_ee + theta_xi**2 / g_xx + 1.0 / g_pp)
    )
    H_ex = sinf * (f_eta * theta_xi - f_xi * theta_eta)
    H_ep = sinf * f_eta
    H_xp = sinf * f_xi
    HH = 2.0 * (H_ex**2 / (g_ee * g_xx) + H_ep**2 / (g_ee * g_pp) + H_xp**2 / (g_xx * g_pp))
    Edens = kin + HH / (4.0 * e0**2)
    return jnp.sum(sqrtg * Edens) * deta * dxi
