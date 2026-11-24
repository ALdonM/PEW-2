"""part206 : viriel sur le reseau lui-meme. Energie L1 sans jauge avec pas anisotrope (hx,hy,hz) ; champ phi fixe (valeurs aux noeuds).
Changer hx a champ fixe = etirer le champ discret selon x SANS interpolation ni couture. Pour un champ relaxe, dE/dln(hx) = -Sxx (viriel exact du reseau).
Identite validee (part204b) : P/v - M_rel = -dE/dlnhx  (+ gam-1 petit)."""
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.'); jax.config.update("jax_enable_x64", True)
import lagcov
from zlat import cross
mu,e0,d2,GUARD=0.375,0.25,1e-3,1e-6; cF=1/(2*e0**2)
def nv(phi):
    r2=jnp.sum(phi**2,0); a,b,c,d=phi
    return jnp.stack([2*(a*c+b*d),2*(a*d-b*c),a*a+b*b-c*c-d*d])/(r2+GUARD)
def energy_a(phi,hs,nu=0.0):
    hx,hy,hz=hs; hh=(hx,hy,hz); vol=hx*hy*hz; r2=jnp.sum(phi**2,0); n=nv(phi)
    grad=sum(jnp.sum((jnp.roll(phi,-1,axis=1+a)-phi)**2,0)/hh[a]**2 for a in range(3))
    Df=[(jnp.roll(n,-1,axis=1+a)-n)/hh[a] for a in range(3)]; Db=[(n-jnp.roll(n,1,axis=1+a))/hh[a] for a in range(3)]
    fad=0.0
    for (i,j) in ((0,1),(0,2),(1,2)):
        for Di in (Df[i],Db[i]):
            for Dj in (Df[j],Db[j]): fad=fad+0.25*jnp.sum(n*cross(Di,Dj),0)**2
    pot=mu**2*(r2+1/(r2+d2)-2)+2*nu**2*(phi[2]**2+phi[3]**2)
    return vol*jnp.sum(grad+cF*fad+pot)
if __name__=='__main__':
    fn=sys.argv[1]; h=float(sys.argv[2]); nu=float(sys.argv[3]); f=jnp.array(np.load(fn).astype(float)); N=f.shape[1]
    S=lagcov.build(N,h=h,nu=nu,g2=0.0,KF=0.0); E0=float(S['energy'](f,jnp.zeros((3,N,N,N)))); Ea=float(energy_a(f,(h,h,h),nu))
    print(f'controle isotrope : lagcov={E0:.6f} aniso={Ea:.6f} diff={Ea-E0:.2e}')
    vac=mu**2*((1-d2)+1/1.0-2); Evac=vac*(N*h)**3; M=E0-Evac; print(f'M_rel={M:.3f}')
    for ax,nm in ((0,'x'),(2,'z')):
        def Eh(lam):
            hs=[h,h,h]; hs[ax]=h*lam; return float(energy_a(f,tuple(hs),nu))
        for e in (0.02,0.005):
            d=(Eh(1+e)-Eh(1-e))/(2*e); print(f'axe {nm} eps={e}: dE/dln h = {d:+.3f}  -> -dE/dlnh = {-d:+.3f}  ({-d/M*100:+.2f} % de M)')
    # decomposition par morceaux (derivees numeriques en hx)
    n=nv(f)
