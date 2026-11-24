"""part204b : test exact de la covariance de lagcov (outil de mesure + Lagrangien discret), sur un champ STATIQUE QUELCONQUE a support compact.
Identite exacte pour une theorie covariante (indep. de toute relaxation / stationnarite) : pour phi0 statique et phi_b(x)=phi0(c+g(x-c)), phidot=-v phi_x,
   H = gam*(M + v^2 Sxx),  P = gam*v*(M + Sxx),  avec Sxx = int T^xx = -dE/dln(lambda_x)   (lambda_x = etirement de phi0 selon x).
Champ analytique -> aucune erreur d'interpolation. KF=1 doit verifier l'identite (erreur ~h^2), KF=0 (theorie non covariante) non."""
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.'); jax.config.update("jax_enable_x64", True)
import lagcov
SX,SY,SZ=2.5,3.5,3.0
def field(x,y,z,A=(-0.3,0.6,0.5,0.4)):
    g=np.exp(-(x**2/(2*SX**2)+y**2/(2*SY**2)+z**2/(2*SZ**2)))
    return np.stack([1+A[0]*g, A[1]*g, A[2]*g*x/SX, A[3]*g*y/SY])
def run(h,v=0.1,L=32.0,nu=0.0,KF=1.0,axis=0,e0=0.25):
    N=int(round(L/h)); c=(N-1)/2; x=(np.arange(N)-c)*h; X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
    S=lagcov.build(N,h=h,nu=nu,g2=0.0,KF=KF,e0=e0); th=jnp.zeros((3,N,N,N)); w=jnp.zeros((3,N,N,N)); gam=1/np.sqrt(1-v*v)
    def f0(lam=1.0,gb=1.0):
        co=[X,Y,Z]; co[axis]=co[axis]*gb/lam; return field(*co)          # phi0(x_axis -> gb*x/lam)
    dxf=lambda a:(np.roll(a,-1,axis=axis+1)-np.roll(a,1,axis=axis+1))/(2*h)
    En=lambda lam:float(S['energy'](jnp.array(f0(lam)),th)); M=En(1.0); e=0.005
    Evac=float(S['energy'](jnp.array(np.stack([np.full((N,N,N),np.sqrt(1-1e-3)),*[np.zeros((N,N,N))]*3])),th)); M=M-Evac   # M = energie relative au vide uniforme (le vide a une densite d'energie -d2*mu^2 non nulle, invariante de Lorentz : elle se compense dans P)
    dE=(En(1+e)-En(1-e))/(2*e)                                           # dE/dln(lambda) (centre, O(e^2))
    phb=f0(1.0,gam); vel=-v*dxf(phb); q=(jnp.array(phb),th); vv=(jnp.array(vel),w)
    Hb=float(S['hamil'](q,vv)); p=S['pfun'](q,vv)[0]; P=float(-jnp.sum(p*jnp.array(dxf(phb))))
    Hp=gam*(M-v**2*dE); Pp=gam*v*(M-dE)
    return dict(h=h,N=N,KF=KF,M=M,dEdlnl=dE,P=P,Ppred=Pp,errP=P/Pp-1,H=Hb,Hpred=Hp,errH=Hb/Hp-1)
if __name__=='__main__':
    for KF in (1.0,0.0):
        for h in (0.5,0.4,0.32):
            r=run(h,KF=KF); print(f"KF={KF:g} h={h} N={r['N']}: M={r['M']:.3f} dE/dlnl={r['dEdlnl']:.3f} | P={r['P']:.5f} P_pred={r['Ppred']:.5f} err={r['errP']:+.2e} | H={r['H']:.5f} H_pred={r['Hpred']:.5f} err={r['errH']:+.2e}",flush=True)
    print('--- temoin : modele sans terme quartique (e0=1e3, cF~0), KF=0 : theorie covariante triviale')
    for h in (0.5,0.4):
        r=run(h,KF=0.0,e0=1e3); print(f"h={h}: P={r['P']:.5f} P_pred={r['Ppred']:.5f} err={r['errP']:+.2e} | H err={r['errH']:+.2e}",flush=True)
    print('--- axe y (z inchange) KF=1')
    for h in (0.5,0.4): r=run(h,KF=1.0,axis=1); print(f"h={h}: err P={r['errP']:+.2e} err H={r['errH']:+.2e}",flush=True)
