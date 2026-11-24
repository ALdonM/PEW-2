"""part205 : profil cumule du viriel de la taille de boite sur l'exces d'inertie P/(Mv) du noeud KF=1.
usage : python -u relax_boost203.py L [h=0.5] [budget_s=18000] [start.npy]
Relaxation Newton-Krylov (MINRES sur H+lam I) du noeud L1-sans-jauge (nu=0,g2=0) dans une boite L^3 (remplissage du noeud L=32 par le bord),
puis mesure du boost (v=0.1, axes x,y,z, KF=0 et 1). Sauvegarde du champ a chaque iteration -> reprise automatique."""
import sys, os, time, json, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.'); jax.config.update("jax_enable_x64", True)
from scipy.sparse.linalg import minres, LinearOperator
from scipy.ndimage import map_coordinates
import lagcov
L=float(sys.argv[1]); h=float(sys.argv[2]) if len(sys.argv)>2 else 0.5; budget=float(sys.argv[3]) if len(sys.argv)>3 else 18000
start=sys.argv[4] if len(sys.argv)>4 else '../../resultats/part203/knot_L32_h0.5_newton.npy'
N=int(round(L/h)); NU=float(os.environ.get('NU','0.12')); tag=f'L{int(L)}_h{h}_nu{NU}'; ck=f'ck_{tag}.npy'; t0=time.time()
if os.path.exists(ck): f=np.load(ck).astype(float); print('reprise',ck,flush=True)
else:
    f=np.load(start).astype(float); pad=(N-f.shape[1])//2; assert 2*pad+f.shape[1]==N,(N,f.shape)
    f=np.pad(f,((0,0),(pad,pad),(pad,pad),(pad,pad)),mode='edge')
S=lagcov.build(N,h=h,nu=NU,g2=0.0,KF=0.0); th=jnp.zeros((3,N,N,N))
E=lambda p:S['energy'](p.reshape(4,N,N,N),th); gr=jax.jit(jax.grad(E)); hv=jax.jit(lambda p,u:jax.jvp(gr,(p,),(u,))[1])
x=jnp.array(f.ravel()); lam=1e-3; hist=[]
for it in range(400):
    g=gr(x); gn=float(jnp.linalg.norm(g)); gi=float(jnp.abs(g).max())/h**3
    if gi<1e-3 or time.time()-t0>budget: break
    A=LinearOperator((x.size,x.size),matvec=lambda u:np.array(hv(x,jnp.array(u))+lam*jnp.array(u)),dtype=np.float64)
    d,info=minres(A,-np.array(g),maxiter=300,rtol=1e-4); d=jnp.array(d); a=1.0
    while a>1e-6:
        xn=x+a*d; gnn=float(jnp.linalg.norm(gr(xn)))
        if gnn<gn*(1-1e-4*a): break
        a/=2
    x=xn; np.save(ck,np.array(x).reshape(4,N,N,N)); hist.append((it,gnn,float(E(x))))
    print(f'it {it}: |g|_2={gnn:.3e} E={hist[-1][2]:.4f} t={time.time()-t0:.0f}s',flush=True)
g=gr(x); gi=float(jnp.abs(g).max())/h**3; conv=gi<1e-3
print(f'FIN relaxation : |g/h3|_inf={gi:.5f} |g|_2={float(jnp.linalg.norm(g)):.3e} E={float(E(x)):.4f} converge={conv}',flush=True)
f=np.array(x).reshape(4,N,N,N); np.save(f'knot_{tag}_final.npy',f)

from zlat import cross
mu,e0,d2=0.375,0.25,1e-3; cF=1/(2*e0**2); Sg=lagcov.build(N,h=h,nu=NU,g2=0.0,KF=0.0); n=Sg['nv'](jnp.array(f)); fj=jnp.array(f); r2=jnp.sum(fj**2,0)
e2=[h**3*jnp.sum((jnp.roll(fj,-1,axis=1+a)-fj)**2,axis=0)/h**2 for a in range(3)]
Df=[(jnp.roll(n,-1,axis=1+a)-n)/h for a in range(3)]; Db=[(n-jnp.roll(n,1,axis=1+a))/h for a in range(3)]
e4={}
for (i,j) in ((0,1),(0,2),(1,2)):
    q=0.0
    for Di in (Df[i],Db[i]):
        for Dj in (Df[j],Db[j]): q=q+0.25*jnp.sum(n*cross(Di,Dj),0)**2
    e4[(i,j)]=cF*h**3*q
vac=mu**2*((1-d2)+1/((1-d2)+d2)-2)
ev=h**3*(mu**2*(r2+1/(r2+d2)-2)-vac)+h**3*2*NU**2*(fj[2]**2+fj[3]**2)
tot=sum(e2)+sum(e4.values())+ev
out={}
for ax,nm in ((0,'x'),(2,'z')):
    others=[a for a in range(3) if a!=ax]; pair_o=e4[tuple(others)]
    pair_a=sum(e4[k] for k in e4 if ax in k)
    dens=(e2[ax]+pair_a)-(sum(e2[a] for a in others)+pair_o+ev)
    out[nm]=np.array(dens)
tot=np.array(tot); c=(N-1)/2; ix=np.arange(N)-c; X,Y,Z=np.meshgrid(ix,ix,ix,indexing='ij'); R=np.sqrt(X**2+Y**2+Z**2)*h
Rg=np.arange(2.0,L/2+0.01,2.0); prof={'R':Rg.tolist(),'M_rel':[float(tot[R<r].sum()) for r in Rg],'Sx':[float(out['x'][R<r].sum()) for r in Rg],'Sz':[float(out['z'][R<r].sum()) for r in Rg]}
prof.update(L=L,h=h,nu=NU,converged=bool(conv),Mtot=float(tot.sum()),Sx_tot=float(out['x'].sum()),Sz_tot=float(out['z'].sum()))
json.dump(prof,open(f'stress205_{tag}.json','w'),indent=1)
print('R   M_rel(<R)   S_x(<R)   S_z(<R)  [S=P/v-M_rel integre] ; totaux',prof['Mtot'],prof['Sx_tot'],prof['Sz_tot'])
for r,m,a,b in zip(prof['R'],prof['M_rel'],prof['Sx'],prof['Sz']): print(f'{r:5.1f} {m:10.2f} {a:9.2f} {b:9.2f}')
