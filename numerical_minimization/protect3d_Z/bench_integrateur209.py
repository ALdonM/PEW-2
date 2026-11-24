import sys,time,numpy as np,jax,jax.numpy as jnp
sys.path.insert(0,'.'); jax.config.update("jax_enable_x64",True)
import lagcov
N=64;h=0.5; S=lagcov.build(N,h=h,nu=0.12,g2=0.02,KF=1.0)
f=np.load('../../resultats/part203/knot_L32_h0.5_newton.npy').astype(float); th=jnp.zeros((3,N,N,N)); q=(jnp.array(f),th)
rng=np.random.default_rng(0); sm=lambda a:np.real(np.fft.ifftn(np.fft.fftn(a)*np.exp(-(np.fft.fftfreq(N)[:,None,None]**2+np.fft.fftfreq(N)[None,:,None]**2+np.fft.fftfreq(N)[None,None,:]**2)*60)))
v=(jnp.array(np.stack([0.02*sm(rng.standard_normal((N,N,N))) for _ in range(4)])),jnp.array(np.stack([0.02*sm(rng.standard_normal((N,N,N))) for _ in range(3)])))
Kz=lambda q,v:S['kin'](q[0],q[1],v[0],v[1],jnp.zeros((N,N,N))); pfun=S['pfun']
dKq=jax.jit(jax.grad(Kz,0)); dEq=jax.jit(jax.grad(lambda q:S['energy'](q[0],q[1])))
jp=jax.jit(lambda q,v:jax.jvp(lambda qq:pfun(qq,v),(q,),(v,))[1]); Mv=jax.jit(lambda q,v,x:jax.jvp(lambda vv:pfun(q,vv),(v,),(x,))[1])
def rhs(q,v): return jax.tree_util.tree_map(lambda a,b,c:a-b-c,dKq(q,v),dEq(q),jp(q,v))
def tm(f,*a,n=3):
    r=f(*a); jax.block_until_ready(r); t=time.time()
    for _ in range(n): r=f(*a); jax.block_until_ready(r)
    return (time.time()-t)/n
print('dK/dq',round(tm(dKq,q,v),3),'dE/dq',round(tm(dEq,q),3),'jvp(dp/dq)',round(tm(jp,q,v),3),'Mv',round(tm(Mv,q,v,v),3),flush=True)
b=rhs(q,v)
def cg(tol,x0=None,maxit=300):
    x=x0 if x0 is not None else jax.tree_util.tree_map(jnp.zeros_like,b)
    r=jax.tree_util.tree_map(lambda a,c:a-c,b,Mv(q,v,x)); p=r; rs=sum(float(jnp.vdot(a,a)) for a in r); b2=sum(float(jnp.vdot(a,a)) for a in b)
    for it in range(maxit):
        if rs<tol**2*b2: return x,it
        Ap=Mv(q,v,p); al=rs/sum(float(jnp.vdot(a,c)) for a,c in zip(p,Ap))
        x=jax.tree_util.tree_map(lambda a,c:a+al*c,x,p); r=jax.tree_util.tree_map(lambda a,c:a-al*c,r,Ap); rn=sum(float(jnp.vdot(a,a)) for a in r)
        p=jax.tree_util.tree_map(lambda a,c:a+(rn/rs)*c,r,p); rs=rn
    return x,maxit
xref,it=cg(1e-13); print('CG tol 1e-13 :',it,'iterations')
for tol in (1e-4,1e-6,1e-8,1e-10): x,it=cg(tol); err=max(float(jnp.abs(a-c).max()) for a,c in zip(x,xref))/max(float(jnp.abs(c).max()) for c in xref); print(f'tol {tol:g}: {it} it, erreur rel sur a = {err:.1e}',flush=True)
print('--- CG preconditionne (bloc 4x4 par site pour phi, diagonale pour theta)')
ones=lambda c: (jnp.zeros((4,N,N,N)).at[c].set(1.0),jnp.zeros((3,N,N,N)))
B=jnp.stack([Mv(q,v,ones(c))[0] for c in range(4)],axis=0)        # B[c,c',x] = M_{c' c}(x)
Bm=jnp.moveaxis(B,(0,1),(-1,-2))                                  # (N,N,N,c',c)
Binv=jnp.linalg.inv(Bm)
dth=jnp.stack([Mv(q,v,(jnp.zeros((4,N,N,N)),jnp.zeros((3,N,N,N)).at[a].set(1.0)))[1][a] for a in range(3)])
def prec(r):
    rp,rt=r; yp=jnp.einsum('...ij,j...->i...',Binv,rp); return (yp,rt/dth)
def pcg(tol,maxit=300):
    x=jax.tree_util.tree_map(jnp.zeros_like,b); r=b; z=prec(r); p=z; rz=sum(float(jnp.vdot(a,c)) for a,c in zip(r,z)); b2=sum(float(jnp.vdot(a,a)) for a in b)
    for it in range(maxit):
        rr=sum(float(jnp.vdot(a,a)) for a in r)
        if rr<tol**2*b2: return x,it
        Ap=Mv(q,v,p); al=rz/sum(float(jnp.vdot(a,c)) for a,c in zip(p,Ap))
        x=jax.tree_util.tree_map(lambda a,c:a+al*c,x,p); r=jax.tree_util.tree_map(lambda a,c:a-al*c,r,Ap); z=prec(r); rzn=sum(float(jnp.vdot(a,c)) for a,c in zip(r,z))
        p=jax.tree_util.tree_map(lambda a,c:a+(rzn/rz)*c,z,p); rz=rzn
    return x,maxit
for tol in (1e-4,1e-6,1e-8,1e-10):
    x,it=pcg(tol); err=max(float(jnp.abs(a-c).max()) for a,c in zip(x,xref))/max(float(jnp.abs(c).max()) for c in xref); print(f'PCG tol {tol:g}: {it} it, erreur rel = {err:.1e}',flush=True)
