import numpy as np, time, jax.numpy as jnp, sys
import zlat, degre, pair
h=0.5; mu=0.375; e0=0.25; cF=1/(2*e0**2); d2=1e-3
knot=np.load('../../resultats/part179/wk_T0/wk_T0_champ_final.npy').astype(np.float64)
N=128
Ej,gE,parts,qdens=zlat.make(h,mu,cF,d2)
E1=float(Ej(jnp.array(pair.embed(knot,N,((N-1)/2,)*3).astype(np.float32)))); print('E1 (seul, N=128)=%.1f'%E1, flush=True)
# rayon d'énergie du noeud seul
g,f,p=zlat.energy_density(jnp.array(knot.astype(np.float32)),h,mu,cF,d2)
e=np.array(g+f+p); x=(np.arange(64)-31.5)*h; X,Y,Z=np.meshgrid(x,x,x,indexing='ij'); r=np.sqrt(X**2+Y**2+Z**2)
etot=e.sum()*h**3; e0r=e-0  # densité nette de vide ~ 0
for R in (2,4,6,8,10,12): print('énergie dans r<%d : %.1f%%'%(R,100*e[r<R].sum()*h**3/etot))
print('r_energie moyen = %.2f'%((e*r).sum()*h**3/etot), flush=True)
res=[]
for beta_name,g_ in [('beta0',None),('pi_x',pair.unit_q([1,0,0],np.pi)),('pi_z',pair.unit_q([0,0,1],np.pi)),('pi2_x',pair.unit_q([1,0,0],np.pi/2))]:
    for sep in (16,24,32,40):   # mailles (h=0.5 -> 8,12,16,20 unités)
        P=pair.pair(knot,N,sep,g_)
        Pj=jnp.array(P.astype(np.float32)); E=float(Ej(Pj))
        Q=degre.degree(P) if sep in (24,40) else None
        print('%-6s sep=%2d mailles (%.0f u) : E=%.1f  E-2E1=%+.1f  Q=%s'%(beta_name,sep,sep*h,E,E-2*E1,Q), flush=True)
