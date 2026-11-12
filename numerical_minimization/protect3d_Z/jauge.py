"""part184b -- modèle Z avec (i) terme de masse nu^2*2|Z2|^2 pour les modes chargés, (ii) champ de jauge U(1) agissant sur Z2 seul (Z2 -> e^{i lam} Z2),
le vide zeta=(1,0) laisse ce U(1) intact (photon sans masse). Faddeev jaugé : Omega_ij = n.(D_i n x D_j n) + (n3 - 1) F_ij  (fermé, jauge-invariant).
Variables : phi (4,N,N,N) = (Re Z1, Im Z1, Re Z2, Im Z2) ; theta (3,N,N,N) phase de lien x -> x+e_a ; A = theta/h ; D Z2 = (e^{-i theta} Z2(x+a) - Z2(x))/h.
Dynamique : phi'' = -dE/dphi /(2 h^3) ; theta'' = -(g2/h) dE/dtheta.   [D] modèle d'essai, aucune identification physique.
usage : python jauge.py check | python jauge.py run NU G2 OMEGA TEND TAG"""
import os, sys, json, time, numpy as np, jax, jax.numpy as jnp
from zlat import nvec, cross, hopfion
N, h, mu, e0, d2 = int(os.environ.get('N', 48)), 0.5, 0.375, 0.25, 1e-3; cF = 1 / (2 * e0**2)
fwd1 = lambda f, a: jnp.roll(f, -1, axis=1 + a)       # f(x+e_a)
bwd1 = lambda f, a: jnp.roll(f, 1, axis=1 + a)        # f(x-e_a)
def rot(n, al):   # rotation de n autour de e3 de l'angle al (n_perp -> e^{i al} n_perp)
    c, s = jnp.cos(al), jnp.sin(al)
    return jnp.stack([n[0] * c - n[1] * s, n[0] * s + n[1] * c, n[2]])
def plaq(th, i, j):   # F_ij au site x (plaquette x, x+i, x+i+j, x+j)
    return (th[i] + jnp.roll(th[j], -1, axis=i) - jnp.roll(th[i], -1, axis=j) - th[j]) / h**2
sq = lambda z: z.real**2 + z.imag**2   # |z|^2 sans gradient singulier en 0
def energy(phi, th, nu, g2):
    c1, c2 = phi[0] + 1j * phi[1], phi[2] + 1j * phi[3]
    sh = lambda f, a: jnp.roll(f, -1, axis=a)
    grad = 0.0
    for a in range(3):
        grad = grad + sq(sh(c1, a) - c1) / h**2 + sq(jnp.exp(-1j * th[a]) * sh(c2, a) - c2) / h**2
    r2 = jnp.sum(phi**2, axis=0); a_, b_, c_, d_ = phi
    n = jnp.stack([2 * (a_ * c_ + b_ * d_), 2 * (a_ * d_ - b_ * c_), a_**2 + b_**2 - c_**2 - d_**2]) / (r2 + 1e-6)   # garde-fou numérique près de rho=0 (n non défini)
    # différences covariantes de n : avant (transport du voisin x+a vers x), arrière
    Df = [(rot(jnp.roll(n, -1, axis=1 + a), -th[a][None][0]) - n) / h for a in range(3)]
    Db = [(n - rot(jnp.roll(n, 1, axis=1 + a), jnp.roll(th[a], 1, axis=a))) / h for a in range(3)]
    fad = 0.0; B2 = 0.0
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        Fp = plaq(th, i, j)
        for di, Di in ((0, Df[i]), (1, Db[i])):
            for dj, Dj in ((0, Df[j]), (1, Db[j])):
                F = jnp.roll(Fp, (di, dj), axis=(i, j))   # plaquette décalée de -di*e_i - dj*e_j
                n3 = n[2]                                   # n3 moyen des 4 coins de la plaquette
                n3c = 0.25 * (n3 + jnp.roll(n3, (-1 + 2 * di, 0), axis=(i, j)) * 0 + 0) if False else None
                # coins de la plaquette (x - di e_i - dj e_j ... + e_i + e_j) vus depuis x : indices relatifs
                oi = [-di, -di + 1]; oj = [-dj, -dj + 1]
                n3m = sum(jnp.roll(n3, (-a_, -b_), axis=(i, j)) for a_ in oi for b_ in oj) / 4
                Om = jnp.sum(n * cross(Di, Dj), axis=0) + (n3m - 1.0) * F
                fad = fad + 0.25 * Om**2
        B2 = B2 + Fp**2
    pot = mu**2 * (r2 + 1.0 / (r2 + d2) - 2.0) + 2 * nu**2 * sq(c2)
    return h**3 * jnp.sum(grad + cF * fad + pot + (0.5 / g2) * B2 if g2 > 0 else grad + cF * fad + pot)
gE = jax.jit(jax.grad(energy, argnums=(0, 1)), static_argnums=(2, 3))
def accs(phi, th, nu, g2):
    gp, gt = gE(phi, th, nu, g2)
    return -0.5 * gp / h**3, (-(g2 / h) * gt if g2 > 0 else jnp.zeros_like(th))
x_ = lambda NN: (np.arange(NN) - NN / 2 + 0.5) * h
if sys.argv[1] == 'check':
    rng = np.random.default_rng(3); phi = jnp.array(hopfion(N, h, 6.0) + 0.05 * rng.normal(size=(4, N, N, N)).astype(np.float32))
    th = jnp.array(0.3 * rng.normal(size=(3, N, N, N)).astype(np.float32)); nu, g2 = 0.3, 0.05
    E0 = float(energy(phi, th, nu, g2))
    lam = jnp.array(rng.uniform(-3, 3, size=(N, N, N)).astype(np.float32))
    c2 = (phi[2] + 1j * phi[3]) * jnp.exp(1j * lam); phi2 = phi.at[2].set(c2.real).at[3].set(c2.imag)
    th2 = jnp.stack([th[a] + jnp.roll(lam, -1, axis=a) - lam for a in range(3)])
    print('invariance de jauge : E, E(transformé), écart relatif =', E0, float(energy(phi2, th2, nu, g2)), abs(E0 - float(energy(phi2, th2, nu, g2))) / E0)
    print('A=0 : E(g2) vs E(sans jauge)', float(energy(phi, th * 0, nu, g2)))
    # fermeture de Omega (div B_eff -> 0 quand h->0) : champ lisse
    for hh in (0.5, 0.25):
        pass
elif sys.argv[1] == 'relax':   # relaxe un nœud isolé (N=48) et le sauvegarde : python jauge.py relax NU FICHIER
    nu = float(sys.argv[2]); dt = 0.02; p = jnp.array(hopfion(N, h, 6.0)); t = jnp.zeros((3, N, N, N), dtype=p.dtype); v = jnp.zeros_like(p)
    a, _ = accs(p, t, nu, 0.0); g = np.exp(-0.5 * dt)
    for i in range(int(20 / dt)):
        v = v * g + 0.5 * dt * a; p = p + dt * v; a, _ = accs(p, t, nu, 0.0); v = (v + 0.5 * dt * a) * g
    np.save(sys.argv[3], np.array(p)); print('relaxé, E =', float(energy(p, t, nu, 0.0)))
elif sys.argv[1] == 'pair':   # python jauge.py pair NU G2 OM TEND TAG SQ SJ   (SQ, SJ dans {opp, same}) ; nécessite RELAXED=fichier.npy ; N=NBIG=80
    from zlat import energy_density
    nu, g2, om, TEND, TAG, SQ, SJ = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7], sys.argv[8]
    dt = float(os.environ.get('DT', 0.02)); CS = 2.0; NB = 48; SEP = int(os.environ.get('SEPC', 12))   # demi-séparation en mailles (12 -> 6 unités)
    k1 = np.load(os.environ['RELAXED'])
    def embed(k, shift, refl):
        big = np.zeros((4, N, N, N), dtype=np.float32); big[0] = 1.0; o = (N - NB) // 2
        big[:, o:o+NB, o:o+NB, o:o+NB] = k
        if refl: big = big[:, ::-1]                       # réflexion x -> -x (grille symétrique) : Q -> -Q exactement
        return np.roll(big, shift, axis=1)
    A = embed(k1, -SEP, False); B = embed(k1, +SEP, SQ == 'opp')
    a0, a1, a2, a3 = A; b0, b1, b2, b3 = B   # produit de quaternions (comme dyn.py)
    p = jnp.array(np.stack([a0*b0-a1*b1-a2*b2-a3*b3, a0*b1+a1*b0+a2*b3-a3*b2, a0*b2-a1*b3+a2*b0+a3*b1, a0*b3+a1*b2-a2*b1+a3*b0]).astype(np.float32))
    xx = x_(N); X, Y, Zc = np.meshgrid(xx, xx, xx, indexing='ij'); R0 = np.sqrt(X**2 + Y**2 + Zc**2)
    c1x, c2x = (-SEP) * h, SEP * h   # centres initiaux (la grille centrée sur 0 est décalée d'une demi-maille ; erreur négligeable)
    w1 = np.exp(-(np.sqrt((X - c1x)**2 + Y**2 + Zc**2) / 8.0)**4); w2 = np.exp(-(np.sqrt((X - c2x)**2 + Y**2 + Zc**2) / 8.0)**4)
    s1, s2 = om, (om if SJ == 'same' else -om)
    wgt = jnp.array((s1 * w1 + s2 * w2).astype(np.float32))[None]
    v = jnp.stack([0 * p[0], 0 * p[1], -p[3], p[2]]) * wgt
    t = jnp.zeros((3, N, N, N), dtype=p.dtype); w = jnp.zeros_like(t)
    kk = 2 * np.pi * np.fft.fftfreq(N, h); lap = -sum((2 - 2 * np.cos(a_ * h)) / h**2 for a_ in np.meshgrid(kk, kk, kk, indexing='ij'))
    def poisson(rhs):
        f = np.fft.fftn(np.array(rhs)); lp = lap.copy(); lp[0, 0, 0] = 1.0; f = f / lp; f[0, 0, 0] = 0
        return jnp.array(np.fft.ifftn(f).real)
    qd = lambda p, v: p[2] * v[3] - p[3] * v[2]
    q = qd(p, v); psi = poisson(g2 * CS * (q - q.mean())); w = jnp.stack([jnp.roll(psi, -1, axis=a_) - psi for a_ in range(3)])
    @jax.jit
    def kdk(p, t, v, w, a, b):
        v = v + 0.5 * dt * a; w = w + 0.5 * dt * b; p = p + dt * v; t = t + dt * w
        a, b = accs(p, t, nu, g2); return p, t, v + 0.5 * dt * a, w + 0.5 * dt * b, a, b
    a, b = accs(p, t, nu, g2)
    Qh = lambda p: float(h**3 * jnp.sum(jnp.linalg.det(jnp.moveaxis(jnp.stack([p / jnp.sqrt(jnp.sum(p**2, 0))] + [(jnp.roll(p / jnp.sqrt(jnp.sum(p**2, 0)), -1, k_) - jnp.roll(p / jnp.sqrt(jnp.sum(p**2, 0)), 1, k_)) / (2 * h) for k_ in (1, 2, 3)]), (0, 1), (-2, -1)))) / (2 * np.pi**2))
    def Qa(p):
        u = p / jnp.sqrt(jnp.sum(p**2, 0) + 1e-12)
        M = jnp.moveaxis(jnp.stack([u] + [(jnp.roll(u, -1, k_) - jnp.roll(u, 1, k_)) / (2 * h) for k_ in (1, 2, 3)]), (0, 1), (-2, -1))
        return float(h**3 * jnp.sum(jnp.abs(jnp.linalg.det(M))) / (2 * np.pi**2))
    left = jnp.asarray(X < 0); Rjx = jnp.asarray(R0 > float(os.environ.get('REXT', 16.0)))
    for i in range(int(TEND / dt) + 1):
        if i % int(1.0 / dt) == 0:
            q = qd(p, v); gd, fd, pd = energy_density(p, h, mu, cF, d2); e = h**3 * (gd + fd + pd + jnp.sum(v**2, 0))
            gg = jnp.asarray(gd); 
            def cen(m):
                ww = jnp.where(m, gg, 0.0); ww = jnp.where(ww > 0.5 * ww.max(), ww, 0.0)
                return float(jnp.sum(ww * jnp.asarray(X)) / jnp.sum(ww))
            rec = dict(t=round(i * dt, 2), Q=round(Qh(p), 3), Qabs=round(Qa(p), 3), qL=round(float(h**3 * jnp.sum(q * left)), 2), qR=round(float(h**3 * jnp.sum(q * ~left)), 2),
                       xL=round(cen(left), 2), xR=round(cen(~left), 2), Etot=round(float(jnp.sum(e)), 2), Eext=round(float(jnp.sum(e * Rjx)), 2),
                       Ekin=round(float(h**3 * jnp.sum(jnp.sum(v**2, 0))), 2), Z2=round(float(h**3 * jnp.sum(p[2]**2 + p[3]**2)), 1))
            print(json.dumps(rec), flush=True)
        p, t, v, w, a, b = kdk(p, t, v, w, a, b)
elif sys.argv[1] in ('run', 'gauss'):
    nu, g2 = float(sys.argv[2]), float(sys.argv[3]); dt = float(os.environ.get('DT', 0.02)); CS = float(os.environ.get('CS', 2.0))
    xx = x_(N); R = np.sqrt(sum(a**2 for a in np.meshgrid(xx, xx, xx, indexing='ij')))
    kk = 2 * np.pi * np.fft.fftfreq(N, h); lap = -sum((2 - 2 * np.cos(a * h)) / h**2 for a in np.meshgrid(kk, kk, kk, indexing='ij'))
    def poisson(rhs):   # laplacien de réseau psi = rhs (rhs de moyenne nulle)
        f = np.fft.fftn(np.array(rhs)); lp = lap.copy(); lp[0, 0, 0] = 1.0; f = f / lp; f[0, 0, 0] = 0
        return jnp.array(np.fft.ifftn(f).real)
    qd = lambda p, v: p[2] * v[3] - p[3] * v[2]
    def gauss_res(p, v, w):   # (1/g2) div(Adot) - CS*q  avec Adot = w/h
        div = sum((w[a] - jnp.roll(w[a], 1, axis=a)) / h for a in range(3)) / h
        return div / g2 - CS * qd(p, v)
    def kdk(p, t, v, w, a, b, g):
        v = v * g + 0.5 * dt * a; w = w * g + 0.5 * dt * b
        p = p + dt * v; t = t + dt * w
        a, b = accs(p, t, nu, g2)
        v = (v + 0.5 * dt * a) * g; w = (w + 0.5 * dt * b) * g
        return p, t, v, w, a, b
    kdk = jax.jit(kdk)
    if sys.argv[1] == 'gauss':
        rng = np.random.default_rng(5); p = jnp.array(hopfion(N, h, 6.0)); t = jnp.zeros((3, N, N, N), dtype=p.dtype)
        v = jnp.array(0.1 * rng.normal(size=p.shape).astype(np.float32)) * jnp.exp(-(R / 8.0)**4)[None]
        q = qd(p, v); w = jnp.stack([jnp.roll(psi := poisson(g2 * CS * (q - q.mean())), -1, axis=a) - psi for a in range(3)])
        a, b = accs(p, t, nu, g2); G0 = gauss_res(p, v, w); print('résidu initial rms (hors fond uniforme):', float(jnp.sqrt(jnp.mean((G0 - G0.mean())**2))))
        for i in range(100): p, t, v, w, a, b = kdk(p, t, v, w, a, b, 1.0)
        G1 = gauss_res(p, v, w); print('CS =', CS, ' variation du résidu après 100 pas : rms', float(jnp.sqrt(jnp.mean((G1 - G0)**2))), ' ~ échelle de q :', float(jnp.sqrt(jnp.mean((CS * qd(p, v))**2))))
        sys.exit()
    om, TEND, TAG = float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
    p = jnp.array(hopfion(N, h, 6.0)); t = jnp.zeros((3, N, N, N), dtype=p.dtype); v = jnp.zeros_like(p); w = jnp.zeros_like(t)
    a, b = accs(p, t, nu, g2); g = np.exp(-0.5 * dt); t0 = time.time()
    for i in range(int(20 / dt)):
        p, t, v, w, a, b = kdk(p, t * 0, v, w * 0, a, b, g)
    print('relaxé : E =', round(float(energy(p, t, nu, g2)), 3), 'tmps', round(time.time() - t0), flush=True)
    wgt = jnp.exp(-(R / 8.0)**4)[None]
    v = jnp.stack([0 * p[0], 0 * p[1], -om * p[3], om * p[2]]) * wgt
    q = qd(p, v); w = jnp.zeros_like(t)
    if g2 > 0:
        psi = poisson(g2 * CS * (q - q.mean())); w = jnp.stack([jnp.roll(psi, -1, axis=a_) - psi for a_ in range(3)])
    a, b = accs(p, t, nu, g2); Gref = gauss_res(p, v, w) if g2 > 0 else None
    msk = lambda r: jnp.asarray(R < r)
    Qh = lambda p: float(h**3 * jnp.sum(jnp.linalg.det(jnp.moveaxis(jnp.stack([p / jnp.sqrt(jnp.sum(p**2, 0))] + [(jnp.roll(p / jnp.sqrt(jnp.sum(p**2, 0)), -1, k) - jnp.roll(p / jnp.sqrt(jnp.sum(p**2, 0)), 1, k)) / (2 * h) for k in (1, 2, 3)]), (0, 1), (-2, -1)))) / (2 * np.pi**2))
    for i in range(int(TEND / dt) + 1):
        if i % int(2.0 / dt) == 0:
            q = qd(p, v); Ek = float(h**3 * jnp.sum(jnp.sum(v**2, 0))) + (float(h**3 * jnp.sum(0.5 / g2 * (w / h)**2)) if g2 > 0 else 0.0)
            rec = dict(t=round(i * dt, 2), Q=round(Qh(p), 3), qtot=round(float(h**3 * jnp.sum(q)), 3), q6=round(float(h**3 * jnp.sum(q * msk(6))), 3),
                       q9=round(float(h**3 * jnp.sum(q * msk(9))), 3), q11=round(float(h**3 * jnp.sum(q * msk(11))), 3),
                       Etot=round(float(energy(p, t, nu, g2)) + Ek, 3), Z2=round(float(h**3 * jnp.sum(p[2]**2 + p[3]**2)), 3),
                       gauss_dev=round(float(jnp.sqrt(jnp.mean((gauss_res(p, v, w) - Gref)**2))), 4) if g2 > 0 else None,
                       Bmax=round(float(jnp.max(jnp.abs(jnp.stack([plaq(t, 0, 1), plaq(t, 0, 2), plaq(t, 1, 2)])))), 4))
            print(json.dumps(rec), flush=True)
        p, t, v, w, a, b = kdk(p, t, v, w, a, b, 1.0)
