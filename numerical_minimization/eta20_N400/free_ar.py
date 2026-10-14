"""Minimisation sur a_r (taille d'anneau libre) : G(champ)=min_a E = 2*sqrt(E2*E4), E(a)=E2*a+E4/a exact.
Gradient par theoreme de l'enveloppe, E2/E4 obtenus de E(a=1), E(a=2). Usage: free_ar.py ETA N INIT_TAG OUT_TAG"""
import sys, os, json, time
import numpy as np
from scipy.optimize import minimize
from energy_native_staggered import build_grid, energy_and_grad_native_staggered
ETA_MAX=float(sys.argv[1]); N=int(sys.argv[2]); INIT=sys.argv[3]; TAG=sys.argv[4]
MAXITER=int(os.environ.get("MAXITER","200000")); CK=250
f0=e0=1.0; f_core=np.pi
eta,xi,deta,dxi,ETA,XI=build_grid(N,N,ETA_MAX); n=N*N
ST=f"free_{TAG}_state.json"; H=f"free_{TAG}_history.jsonl"
if os.path.exists(f"f_free_{TAG}_ckpt.npy"):
    f=np.load(f"f_free_{TAG}_ckpt.npy"); d=np.load(f"dchi_free_{TAG}_ckpt.npy")
    nit0=json.load(open(ST))["nit_ckpt"]
else:
    f=np.load(f"f_{INIT}_ckpt.npy"); d=np.load(f"dchi_{INIT}_ckpt.npy"); nit0=0
x0=np.concatenate([f.ravel(),d.ravel()])
info={}
def fg(x):
    E1,g1=energy_and_grad_native_staggered(x,ETA,XI,deta,dxi,1.0,f0,e0,f_core,N,N)
    Ea,ga=energy_and_grad_native_staggered(x,ETA,XI,deta,dxi,2.0,f0,e0,f_core,N,N)
    E1=float(E1);Ea=float(Ea);g1=np.asarray(g1,float);ga=np.asarray(ga,float)
    E4=(2*E1-Ea)/1.5; E2=E1-E4; dE4=(2*g1-ga)/1.5; dE2=g1-dE4
    if not (np.isfinite(E2) and np.isfinite(E4)) or E2<=1e-12 or E4<=1e-12:
        return 1e6, np.zeros_like(g1)
    G=2*np.sqrt(E2*E4); g=np.sqrt(E4/E2)*dE2+np.sqrt(E2/E4)*dE4
    info.update(E2=E2,E4=E4,astar=np.sqrt(E4/E2),E1=E1)
    return G,g
st={"nit":nit0,"t0":time.time()}
def cb(xk):
    st["nit"]+=1
    if st["nit"]%CK==0:
        np.save(f"f_free_{TAG}_ckpt.npy",xk[:n].reshape(N,N)); np.save(f"dchi_free_{TAG}_ckpt.npy",xk[n:].reshape(N,N))
        G,g=fg(xk); rms=float(np.sqrt(np.mean(g**2)))
        json.dump({"nit_ckpt":st["nit"]},open(ST,"w"))
        r=dict(it=st["nit"],G=float(G),E2=info["E2"],E4=info["E4"],astar=info["astar"],E_ar1=info["E1"],rms=rms,t=time.time())
        open(H,"a").write(json.dumps(r)+"\n")
        print(r,flush=True)
xcur=x0; short=0
while True:
    res=minimize(fg,xcur,jac=True,method="L-BFGS-B",callback=cb,options=dict(maxiter=MAXITER,maxfun=2*MAXITER,gtol=1e-14,ftol=1e-17,maxcor=20,maxls=100))
    print("L-BFGS-B stop:",res.message,"nit",res.nit,"nfev",res.nfev,"fun",res.fun,flush=True)
    xcur=res.x
    if res.nit<5:
        # kick: descente de gradient avec backtracking, pour sortir d'un echec de line search L-BFGS
        fx,gx=fg(xcur); step=1e-3/max(np.linalg.norm(gx),1e-30)
        for k in range(60):
            xn=xcur-step*gx; fn,gn=fg(xn)
            if fn<fx: xcur,fx,gx=xn,fn,gn; step*=1.5
            else: step*=0.5
        print("kick GD ->",fx,flush=True)
        st["nit"]+=1
    short = short+1 if res.nit<5 else 0
    if short>=3 or st["nit"]>=MAXITER: break

