"""Analyse part195b. usage : python analyse_part195b.py [racine]. Pour chaque run : masse M, taille, charge, profil delta rho/rho_vac, G_eff(r_d) = Delta(r_d) * r_d / M."""
import numpy as np, json, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax; jax.config.update('jax_enable_x64', False)
from zlat import energy_density, fwd
R = sys.argv[1] if len(sys.argv) > 1 else '../../resultats/part195b'
N = 96; h = 0.5; cF = 1 / (2 * 0.25**2); d2 = 1e-3; xs = (np.arange(N) - N / 2 + .5) * h
r = np.sqrt(xs[:, None, None]**2 + xs[None, :, None]**2 + xs[None, None, :]**2)
for d in sorted(glob.glob(f'{R}/p195b_*')):
    t = os.path.basename(d); f = f'{d}/{t}_champ_final.npy'
    if not os.path.exists(f): print(t, 'absent'); continue
    P = np.load(f).astype(np.float32); pj = json.load(open(f'{d}/{t}_params.json')); mu = pj['mu']; H = [json.loads(l) for l in open(f'{d}/{t}_history.jsonl')]
    rv = H[-1]['rv']; rho = np.sqrt((P**2).sum(0)); rp = rho / rv
    sx = 2; WA = rv**-2; WR = rv**-sx
    g, fq, p = energy_density(P / rv, h, mu, cF, d2); gr = sum(np.array(fwd(rp, ax, h))**2 for ax in (0, 1, 2)); g = (WR * gr + WA * (np.array(g) - gr)) * rv**2 * rp**2; fq = np.array(fq); p = np.array(p) * rv**2 * rv**-2
    e = g + fq + p; M = e.sum() * h**3; Rrms = np.sqrt((e * r**2).sum() / e.sum())
    shells = {a: rp[(r >= a - 1) & (r < a + 1)].mean() for a in (5, 9, 11, 13, 17, 22)}
    far = 1.0
    print(f'--- {t}: rv={rv} mu={mu} t={H[-1]["t"]:.0f} | M={M:.1f} (E2 {g.sum()*h**3:.1f} E4 {fq.sum()*h**3:.1f} EV {p.sum()*h**3:.2f}) R_rms={Rrms:.2f} | deg {H[-1]["deg"]} Q {H[-1]["Q"]} rho_min/rv {rho.min()/rv:.3f} cos_min {H[-1]["cos_min"]} | rho_mean {H[-1]["rho_mean"]}')
    print('   rho/rv-1 par couche r=5,9,11,13,17,22 :', [round(v - 1, 4) for v in shells.values()])
    print('   G_eff = (rho/rv(r_d)-rho/rv(22)) * r_d / M  pour r_d=9,11,13 :', [round((shells[a] - far) * a / M, 6) for a in (9, 11, 13)], ' pente locale 9->13 de (rho/rv - far):', round(np.log(abs((shells[13]-far)/(shells[9]-far)))/np.log(13/9), 2))
    print('   rho_mean(t):', [(round(x['t']), round(x['rho_mean'], 3)) for x in H[::3]])
