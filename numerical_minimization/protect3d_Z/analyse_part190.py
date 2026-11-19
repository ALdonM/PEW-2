"""Analyse part190 : profil rho(r)-1 du noeud relaxe, pentes locales, ajustement rho_inf + A/r. usage : python analyse_part190.py [racine resultats/part190]"""
import numpy as np, json, sys, glob, os
R = sys.argv[1] if len(sys.argv) > 1 else '../../resultats/part190'
N = 96; h = 0.5; xs = (np.arange(N) - N / 2 + 0.5) * h
r = np.sqrt(xs[:, None, None]**2 + xs[None, :, None]**2 + xs[None, None, :]**2)
for d in sorted(glob.glob(f'{R}/p190_*')):
    t = os.path.basename(d); f = f'{d}/{t}_champ_final.npy'
    if not os.path.exists(f): print(t, 'absent'); continue
    P = np.load(f).astype(float); rho = np.sqrt((P**2).sum(0)); H = [json.loads(l) for l in open(f'{d}/{t}_history.jsonl')]
    print(f'--- {t}  deg final {H[-1].get("deg")}  Q {H[-1].get("Q")}  t {H[-1].get("t")}  rho_min {rho.min():.3f}')
    edges = np.arange(0, 24, 2); m = [rho[(r >= a) & (r < a + 2)].mean() for a in edges]; rc = edges + 1
    print(' r:', list(rc)); print(' rho-1:', [round(x - 1, 4) for x in m])
    # pente locale sur les couches (rho-rho_far) : rho_far = valeur moyenne pour r in [20,24[
    far = m[-1]
    s = [np.log(abs((m[i + 1] - far) / (m[i] - far))) / np.log(rc[i + 1] / rc[i]) if abs(m[i] - far) > 1e-5 and abs(m[i + 1] - far) > 1e-5 else np.nan for i in range(3, 8)]
    print(' pente locale de (rho-rho_loin), r=7..17 :', [round(x, 2) for x in s])
    sel = (rc >= 6) & (rc <= 20); A = np.vstack([np.ones(sel.sum()), 1 / rc[sel]]).T; c = np.linalg.lstsq(A, np.array(m)[sel], rcond=None)[0]
    res = np.array(m)[sel] - A @ c; print(f' ajustement rho_inf + A/r (r 6-20) : rho_inf={c[0]:.4f} A={c[1]:+.4f} rms residu {res.std():.5f}')
    A4 = np.vstack([np.ones(sel.sum()), 1 / rc[sel]**4]).T; c4 = np.linalg.lstsq(A4, np.array(m)[sel], rcond=None)[0]; print(f' ajustement rho_inf + B/r^4 : rms residu {(np.array(m)[sel]-A4@c4).std():.5f}')
