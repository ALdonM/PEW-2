"""part209 : test de tolerance en dt de l'integrateur saut-de-mouton sur lagcov (jauge g2=0.02, nu=0.12, KF=1), CG preconditionne a depart a chaud.
Depart : noeud relaxe L=32 (h=0.5), boost v=0.1 sur phi seul (theta=0, w=0), T=2, boite periodique sans eponge. usage : python dt_test209.py dt T"""
import sys,time,json,numpy as np,jax,jax.numpy as jnp
sys.path.insert(0,'.'); jax.config.update("jax_enable_x64",True)
from scipy.ndimage import map_coordinates
import lagcov
dt=float(sys.argv[1]); T=float(sys.argv[2]); N=64; h=0.5; v0=0.1; TOL=1e-8; PREC_EVERY=5
S=lagcov.build(N,h=h,nu=0.12,g2=0.02,KF=1.0); pfun=S['pfun']; a0z=jnp.zeros((N,N,N))
Kz=lambda q,v:S['kin'](q[0],q[1],v[0],v[1],a0z)
f=np.load('../../resultats/part203/knot_L32_h0.5_newton.npy').astype(float); c=(N-1)/2; ix=np.arange(N,dtype=float); X,Y,Z=np.meshgrid(ix,ix,ix,indexing='ij'); gam=1/np.sqrt(1-v0**2)
phb=np.stack([map_coordinates(f[k],[c+gam*(X-c),Y,Z],order=3,mode='grid-wrap') for k in range(4)]); dxf=lambda a:(np.roll(a,-1,axis=1)-np.roll(a,1,axis=1))/(2*h)
q=(jnp.array(phb),jnp.zeros((3,N,N,N))); v=(jnp.array(-v0*dxf(phb)),jnp.zeros((3,N,N,N)))
dKq=jax.jit(jax.grad(Kz,0)); dEq=jax.jit(jax.grad(lambda q:S['energy'](q[0],q[1])))
jp=jax.jit(lambda q,v:jax.jvp(lambda qq:pfun(qq,v),(q,),(v,))[1]); Mv=jax.jit(lambda q,v,x:jax.jvp(lambda vv:pfun(q,vv),(v,),(x,))[1])
rhs=jax.jit(lambda q,v:jax.tree_util.tree_map(lambda a,b,c:a-b-c,dKq(q,v),dEq(q),jp(q,v)))
@jax.jit
def build_prec(q,v):
    ones=lambda c:(jnp.zeros((4,N,N,N)).at[c].set(1.0),jnp.zeros((3,N,N,N)))
    B=jnp.stack([Mv(q,v,ones(c))[0] for c in range(4)],0); Binv=jnp.linalg.inv(jnp.moveaxis(B,(0,1),(-1,-2)))
    dth=jnp.stack([Mv(q,v,(jnp.zeros((4,N,N,N)),jnp.zeros((3,N,N,N)).at[a].set(1.0)))[1][a] for a in range(3)]); return Binv,dth
dot=lambda a,b:sum(jnp.vdot(x,y) for x,y in zip(a,b)); tm=jax.tree_util.tree_map
def acc(q,v,x0,P):
    Binv,dth=P; pr=lambda r:(jnp.einsum('...ij,j...->i...',Binv,r[0]),r[1]/dth)
    b=rhs(q,v); x=x0; r=tm(lambda a,c:a-c,b,Mv(q,v,x)); z=pr(r); p=z; rz=float(dot(r,z)); b2=float(dot(b,b)); it=0
    while float(dot(r,r))>TOL**2*b2 and it<100:
        Ap=Mv(q,v,p); al=rz/float(dot(p,Ap)); x=tm(lambda a,c:a+al*c,x,p); r=tm(lambda a,c:a-al*c,r,Ap); z=pr(r); rzn=float(dot(r,z)); p=tm(lambda a,c:a+(rzn/rz)*c,z,p); rz=rzn; it+=1
    return x,it
hamil=jax.jit(lambda q,v:S['hamil'](q,v)); gauss=jax.jit(lambda q,v:S['gauss'](q,v))
P=build_prec(q,v); a,_=acc(q,v,tm(jnp.zeros_like,v),P); nst=int(round(T/dt)); t0=time.time(); its=0; H0=float(hamil(q,v)); log=[]
for n in range(nst):
    if n%PREC_EVERY==0: P=build_prec(q,v)
    vh=tm(lambda x,y:x+0.5*dt*y,v,a); q=tm(lambda x,y:x+dt*y,q,vh); a,it1=acc(q,vh,a,P); v=tm(lambda x,y:x+0.5*dt*y,vh,a); its+=it1
    if (n+1)%max(1,nst//8)==0: log.append((round((n+1)*dt,4),float(hamil(q,v))/H0-1)); print(f'dt={dt} t={(n+1)*dt:.3f} dH/H={log[-1][1]:+.2e} CG moy={its/(n+1):.1f} {time.time()-t0:.0f}s',flush=True)
np.save(f'dt209_phi_dt{dt}.npy',np.array(q[0])); np.save(f'dt209_th_dt{dt}.npy',np.array(q[1]))
json.dump(dict(dt=dt,T=T,H0=H0,log=log,cg_mean=its/nst,wall=time.time()-t0,gauss=float(jnp.abs(gauss(q,v)).max())),open(f'dt209_dt{dt}.json','w'))
