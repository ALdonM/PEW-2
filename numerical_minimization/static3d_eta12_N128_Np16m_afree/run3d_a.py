"""PEW 3D statique, a_r LIBRE (derniere variable de x), axe gele, chunks reprenables.
usage: python -u run3d.py ETA_MAX N Np TAG   ; env: AR, AMP, MAXITER, TOTAL_MAXITER, CHECKPOINT_EVERY
Depart (1er job) : f2d_start.npy, d2d_start.npy (etat 2D N x N) repliques en phi + bruit lisse en (eta,xi) et blanc en phi, d'amplitude AMP.
Diagnostics : E - E2D (E2D = energie de l'etat 2D replique, meme a_r), rms gradient, spectre de Fourier en phi de f et dchi."""
import sys, os, json, time, numpy as np, jax.numpy as jnp
from scipy.optimize import minimize
from energy_3d import *
import jax
egA=jax.jit(jax.value_and_grad(total_energy_3d,argnums=(0,5)),static_argnums=(9,10,11,12,13))
EM, N, Np, TAG = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
AR=float(os.environ.get('AR','1.743')); AMP=float(os.environ.get('AMP','0.01'))
MAXITER=int(os.environ.get('MAXITER','50000')); TOTAL=int(os.environ.get('TOTAL_MAXITER','500000')); CK=int(os.environ.get('CHECKPOINT_EVERY','5000'))
eta,xi,deta,dxi,ETA,XI=build_grid3(N,N,EM); V=N*N*Np
args=(ETA,XI,deta,dxi)
TAIL=(1.0,1.0,np.pi,N,N,Np,1,1)
MASK=float(os.environ.get('MASK_ETA','0.6'))   # lignes eta < MASK : dependance en phi GELEE (= etat 2D) - evite l'effondrement de reseau pres de l'axe
rowsm=eta<MASK
mk=np.zeros((N,N,Np),bool); mk[rowsm]=True; mk=np.concatenate([mk.ravel(),mk.ravel()])
def fun(x):
    E,(g,ga)=egA(jnp.array(x[:-1]),*args,float(x[-1]),*TAIL); g=np.array(g,float); g[mk]=0.0
    return float(E),np.concatenate([g,[float(ga)]])
f2=np.load('f2d_start.npy'); d2=np.load('d2d_start.npy')
x2=np.concatenate([np.repeat(f2[:,:,None],Np,2).ravel(),np.repeat(d2[:,:,None],Np,2).ravel(),[1.743]])
E2D,_=fun(x2)   # reference : etat 2D relaxe, a=1.743, replique en phi
NPZ=f'x3d_{TAG}.npz'; HIST=f'x3d_{TAG}_history.jsonl'
if os.path.exists(NPZ):
    z=np.load(NPZ); x0=z['x']; nit0=int(z['nit']); print('Reprise',NPZ,'nit',nit0,flush=True)
else:
    rng=np.random.default_rng(0)
    from scipy.ndimage import gaussian_filter
    sm=lambda: (lambda a: a/a.std())(gaussian_filter(rng.standard_normal((N,N,Np)),sigma=(4,4,0),mode=('nearest','wrap','wrap')))
    pf=AMP*sm()*np.exp(-0.5*((eta[:,None,None]-2.0)/1.5)**2)   # bruit lisse (largeur ~4 cellules), blanc en phi
    pd=AMP*sm()
    f3=np.clip(np.repeat(f2[:,:,None],Np,2)+pf,1e-3,np.pi-1e-6); d3=np.repeat(d2[:,:,None],Np,2)+pd
    f3=np.array(f3); d3=np.array(d3); f3[rowsm]=np.repeat(f2[rowsm][:,:,None],Np,2); d3[rowsm]=np.repeat(d2[rowsm][:,:,None],Np,2)
    x0=np.concatenate([f3.ravel(),d3.ravel(),[AR]]); nit0=0; print('Initialisation (2D replique + bruit phi AMP=%g)'%AMP,flush=True)
def diag(x,nit):
    E,g=fun(x); f=x[:V].reshape(N,N,Np); d=x[V:2*V].reshape(N,N,Np)
    sp=lambda a:[float(np.sqrt(np.sum(np.abs(np.fft.rfft(a,axis=2)[:,:,k])**2)/Np)) for k in range(Np//2+1)]
    rec=dict(it=nit,E=E,a=float(x[-1]),ga=float(g[-1]),dE=E-E2D,rms=float(np.sqrt(np.mean(g[:-1]**2))),f_k=sp(f),d_k=sp(d),t=time.time())
    open(HIST,'a').write(json.dumps(rec)+'\n'); print({k:(round(v,9) if isinstance(v,float) else v) for k,v in rec.items() if k not in('f_k','d_k','t')},flush=True); return rec
print('AR initial',AR,'E2D (a initial, replique)',E2D,flush=True)
cnt=[0]; last=[x0]
def cb(xk):
    cnt[0]+=1; last[0]=xk.copy()
    if cnt[0]%CK==0:
        np.savez(NPZ+'.tmp.npz',x=xk,nit=nit0+cnt[0]); os.replace(NPZ+'.tmp.npz',NPZ); diag(xk,nit0+cnt[0])
budget=min(MAXITER,TOTAL-nit0)
r=minimize(fun,x0,jac=True,method='L-BFGS-B',callback=cb,options=dict(maxiter=budget,maxfun=3*budget,maxcor=30,ftol=1e-16,gtol=1e-12))
np.savez(NPZ+'.tmp.npz',x=r.x,nit=nit0+r.nit); os.replace(NPZ+'.tmp.npz',NPZ)
rec=diag(r.x,nit0+r.nit); json.dump(dict(nit_ckpt=nit0+r.nit,dE=rec['dE'],rms=rec['rms'],a=rec['a'],converged=('CONVERGENCE' in str(r.message)),msg=str(r.message)),open(f'x3d_{TAG}_state.json','w'))
print('FINAL',rec['dE'],rec['rms'],'nit_cumulative',nit0+r.nit,flush=True)
