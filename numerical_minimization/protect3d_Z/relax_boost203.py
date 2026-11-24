"""part203 : effet de la taille de boite sur l'exces d'inertie P/(Mv) du noeud KF=1.
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
N=int(round(L/h)); tag=f'L{int(L)}_h{h}'; ck=f'ck_{tag}.npy'; t0=time.time()
if os.path.exists(ck): f=np.load(ck).astype(float); print('reprise',ck,flush=True)
else:
    f=np.load(start).astype(float); pad=(N-f.shape[1])//2; assert 2*pad+f.shape[1]==N,(N,f.shape)
    f=np.pad(f,((0,0),(pad,pad),(pad,pad),(pad,pad)),mode='edge')
S=lagcov.build(N,h=h,nu=0.0,g2=0.0,KF=0.0); th=jnp.zeros((3,N,N,N))
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
dxf=lambda a:(np.roll(a,-1,axis=1)-np.roll(a,1,axis=1))/(2*h)
v=0.1; c=(N-1)/2; ix=np.arange(N,dtype=float); X,Y,Z=np.meshgrid(ix,ix,ix,indexing='ij'); gam=1/np.sqrt(1-v*v); res={'L':L,'h':h,'N':N,'converged':bool(conv),'gi':gi,'E':float(E(x))}
for KF in (0.0,1.0):
    Sk=lagcov.build(N,h=h,nu=0.0,g2=0.0,KF=KF); w=jnp.zeros((3,N,N,N)); out=[]
    for ax in (0,1,2):
        perm=list(range(3)); perm[0],perm[ax]=perm[ax],perm[0]; fp=np.stack([np.transpose(f[k],perm) for k in range(4)])
        phb=np.stack([map_coordinates(fp[k],[c+gam*(X-c),Y,Z],order=3,mode='grid-wrap') for k in range(4)]); vel=-v*dxf(phb)
        q=(jnp.array(phb),th); vv=(jnp.array(vel),w); M=float(Sk['energy'](jnp.array(fp),th)); Hb=float(Sk['hamil'](q,vv)); p=Sk['pfun'](q,vv)[0]; P=float(-jnp.sum(p*jnp.array(dxf(phb))))
        out.append((P/(M*v),Hb/M))
    res[f'KF{KF:g}']={'M':M,'P_over_Mv_xyz':[a for a,_ in out],'H_over_M_xyz':[b for _,b in out]}
    print(f'KF={KF:g} M={M:.3f} P/(Mv) x,y,z = '+', '.join(f'{a:.4f}' for a,_ in out),flush=True)
json.dump(res,open(f'boost203_{tag}.json','w'),indent=1); print('covariant attendu :',round(gam,5))
