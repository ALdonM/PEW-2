"""Minimisation PEW, reprise et sauvegardes longues.
MAXITER = plafond par job ; TOTAL_MAXITER = plafond cumulatif ; CHECKPOINT_EVERY = frequence de sauvegarde.
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize
from energy_native_staggered import build_grid, energy_and_grad_native_staggered

ETA_MAX, N, INIT, TAG = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
MAXITER = int(os.environ.get('MAXITER', '100000'))
TOTAL_MAXITER = int(os.environ.get('TOTAL_MAXITER', '1000000'))
CK = int(os.environ.get('CHECKPOINT_EVERY', '10000'))
if min(MAXITER, TOTAL_MAXITER, CK) < 1:
    raise ValueError('MAXITER, TOTAL_MAXITER et CHECKPOINT_EVERY doivent etre >= 1')
f0 = e0 = 1.0
f_core = np.pi
eta, xi, deta, dxi, ETA, XI = build_grid(N, N, ETA_MAX)
n = N*N
PC = int(os.environ.get('PC','1')); ETA0 = float(os.environ.get('PC_ETA0','8.0'))
# Preconditionnement exponentiel statique : f = pi - s(eta)*u_f, s = exp(-(eta-ETA0)) pour eta>ETA0 (sinon 1) ; dchi inchange.
S_rows = np.where(eta > ETA0, np.exp(-(eta-ETA0)), 1.0) if PC else np.ones_like(eta)
S2 = np.repeat(S_rows, N)  # indexation [eta ligne, xi colonne] aplatie ligne par ligne
def to_u(x):
    u = x.copy(); u[:n] = (np.pi - x[:n]) / S2; return u
def to_x(u):
    x = u.copy(); x[:n] = np.pi - S2*u[:n]; return x
ST, H = f'free_{TAG}_state.json', f'free_{TAG}_history.jsonl'
FCK, DCK, NPZ = f'f_free_{TAG}_ckpt.npy', f'dchi_free_{TAG}_ckpt.npy', f'free_{TAG}_checkpoint.npz'

def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh); fh.write('\n'); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)

def atomic_npy(path, arr):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        np.save(fh, arr); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)

def atomic_npz(path, **arrays):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)

# NPZ est le checkpoint de reference, car il contient les deux champs et le compteur ensemble.
if os.path.exists(NPZ):
    with np.load(NPZ) as z:
        f, d = np.asarray(z['f'], float), np.asarray(z['dchi'], float)
        total_nit = int(z['nit_ckpt'])
    print(f'Reprise depuis {NPZ}: nit_cumulative={total_nit}', flush=True)
else:
    hf, hd = os.path.exists(FCK), os.path.exists(DCK)
    if hf != hd:
        raise RuntimeError(f'Checkpoint incomplet: verifier {FCK} et {DCK}.')
    if hf:
        if not os.path.exists(ST): raise RuntimeError(f'{ST} absent avec checkpoints NPY.')
        f, d = np.load(FCK), np.load(DCK)
        with open(ST, encoding='utf-8') as fh: total_nit = int(json.load(fh)['nit_ckpt'])
        print(f'Reprise depuis NPY: nit_cumulative={total_nit}', flush=True)
    else:
        fi, di = f'f_{INIT}_ckpt.npy', f'dchi_{INIT}_ckpt.npy'
        if not (os.path.exists(fi) and os.path.exists(di)):
            raise FileNotFoundError(f'Checkpoint absent et fichiers initiaux manquants: {fi}, {di}')
        f, d, total_nit = np.load(fi), np.load(di), 0
        print(f'Initialisation depuis {fi} et {di}', flush=True)
if f.shape != (N,N) or d.shape != (N,N): raise ValueError(f'Dimensions inattendues: {f.shape}, {d.shape}; attendu {(N,N)}')
if not 0 <= total_nit <= TOTAL_MAXITER: raise ValueError(f'Compteur cumulatif incoherent: {total_nit}')
xcur = to_u(np.concatenate([f.ravel(), d.ravel()]))
info, run_nit, last_saved_nit, last_history_it = {}, 0, None, -1
if os.path.exists(H):
    with open(H, encoding='utf-8') as fh:
        for line in fh:
            try: last_history_it = int(json.loads(line)['it'])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError): pass

def fg_x(x):
    E1,g1 = energy_and_grad_native_staggered(x, ETA, XI, deta, dxi, 1.0, f0, e0, f_core, N, N)
    Ea,ga = energy_and_grad_native_staggered(x, ETA, XI, deta, dxi, 2.0, f0, e0, f_core, N, N)
    E1,Ea = float(E1),float(Ea); g1,ga = np.asarray(g1,float),np.asarray(ga,float)
    E4=(2.0*E1-Ea)/1.5; E2=E1-E4
    dE4=(2.0*g1-ga)/1.5; dE2=g1-dE4
    if not (np.isfinite(E2) and np.isfinite(E4)) or E2<=1e-12 or E4<=1e-12: return 1e6,np.zeros_like(g1)
    G=2.0*np.sqrt(E2*E4); grad=np.sqrt(E4/E2)*dE2+np.sqrt(E2/E4)*dE4
    info.update(E2=E2,E4=E4,astar=np.sqrt(E4/E2),E1=E1)
    return float(G),grad

def fg(u):
    x = to_x(u); G, g = fg_x(x); gu = np.asarray(g, float).copy(); gu[:n] *= -S2
    return G, gu

def save_checkpoint(u, write_history=True):
    x = to_x(u)
    global last_saved_nit,last_history_it
    G,grad=fg_x(x); ff=x[:n].reshape(N,N); dd=x[n:].reshape(N,N)
    atomic_npz(NPZ,f=ff,dchi=dd,nit_ckpt=np.asarray(total_nit,dtype=np.int64))
    atomic_npy(FCK,ff); atomic_npy(DCK,dd)
    atomic_json(ST,{'nit_ckpt':int(total_nit),'total_maxiter':TOTAL_MAXITER,'N':N,'eta_max':ETA_MAX,'tag':TAG})
    if write_history and total_nit>last_history_it:
        rec={'it':int(total_nit),'G':float(G),'E2':float(info['E2']),'E4':float(info['E4']),'astar':float(info['astar']),'E_ar1':float(info['E1']),'rms':float(np.sqrt(np.mean(np.asarray(grad)**2))),'t':time.time()}
        with open(H,'a',encoding='utf-8') as fh: fh.write(json.dumps(rec)+'\n'); fh.flush(); os.fsync(fh.fileno())
        last_history_it=total_nit; print(rec,flush=True)
    last_saved_nit=total_nit

def callback(xk):
    global run_nit,total_nit
    run_nit+=1; total_nit+=1
    if total_nit % CK == 0: save_checkpoint(xk)

start=time.time(); budget=min(MAXITER,TOTAL_MAXITER-total_nit)
if budget<=0:
    print(f'Plafond cumulatif deja atteint: {total_nit}/{TOTAL_MAXITER}; aucun calcul lance.',flush=True)
else:
    print(f'Debut job: cumul={total_nit}, budget_job={budget}, plafond={TOTAL_MAXITER}, checkpoint_every={CK}',flush=True)
    short=0
    while run_nit<budget:
        remaining=budget-run_nit
        res=minimize(fg,xcur,jac=True,method='L-BFGS-B',callback=callback,options=dict(maxiter=remaining,maxfun=max(2*remaining,100),gtol=1e-14,ftol=1e-17,maxcor=20,maxls=100))
        print('L-BFGS-B stop:',res.message,'nit_this_call',res.nit,'nit_this_job',run_nit,'nit_cumulative',total_nit,'nfev',res.nfev,'fun',res.fun,flush=True)
        xcur=res.x
        if run_nit>=budget: break
        if res.nit<5:
            fx,gx=fg(xcur); step=1e-3/max(np.linalg.norm(gx),1e-30); accepted=0
            for _ in range(60):
                xn=xcur-step*gx; fn,gn=fg(xn)
                if np.isfinite(fn) and fn<fx: xcur,fx,gx=xn,fn,gn; step*=1.5; accepted+=1
                else: step*=0.5
            print('kick GD ->',float(fx),'accepted_steps',accepted,flush=True)
            save_checkpoint(xcur,write_history=False); short+=1
            if short>=3:
                print('Arret de ce job: trois appels L-BFGS-B successifs ont avance de moins de 5 iterations.',flush=True); break
        else: short=0
if last_saved_nit!=total_nit or not os.path.exists(NPZ): save_checkpoint(xcur)
Gfinal,gfinal=fg_x(to_x(xcur))
print('FINAL',{'nit_this_job':int(run_nit),'nit_cumulative':int(total_nit),'total_maxiter':TOTAL_MAXITER,'G':float(Gfinal),'E2':float(info['E2']),'E4':float(info['E4']),'astar':float(info['astar']),'E_ar1':float(info['E1']),'rms':float(np.sqrt(np.mean(np.asarray(gfinal)**2))),'elapsed_s':float(time.time()-start)},flush=True)
