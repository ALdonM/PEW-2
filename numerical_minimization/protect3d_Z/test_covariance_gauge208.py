"""part208 : covariance de lagcov AVEC JAUGE (liens theta, champ E=theta_dot/h, terme de Faddeev temporel jauge) sur champ statique compact quelconque.
Jauge de depart A_x = 0 -> le boost de A' = Lambda A reste en jauge temporelle (A0'=0) : theta'_x=0, theta'_y(x)=h A_y(xi), theta'_z(x)=h A_z(xi), w=theta_dot=-v d_x theta, phi_dot=-v D_x phi.
Identite (theorie covariante, hors couche : pas besoin de Gauss car A_x=0 annule le terme de Gauss dans P) :
   P = gam v (M_rel + Sxx),  H = gam (M_rel + v^2 Sxx) + E_vide,  Sxx = -dE/dln(lambda_x)  (etirement du champ compact : phi(x/lam), A_y(x/lam), A_z(x/lam)).
P = sum p.(-D_x phi) + sum pi.(delta theta), delta theta_y = -h F_xy (moyenne des 2 plaquettes), delta theta_z = -h F_xz."""
import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,'.'); jax.config.update("jax_enable_x64", True)
import lagcov
SX,SY,SZ=2.5,3.5,3.0
def fields(x,y,z,hh,s=(0,0,0)):
    """phi (4,N,N,N) aux noeuds ; theta (3,...) aux liens : theta_y en (x,y+h/2,z), theta_z en (x,y,z+h/2), theta_x=0."""
    g=lambda x,y,z:np.exp(-(x**2/(2*SX**2)+y**2/(2*SY**2)+z**2/(2*SZ**2)))
    G=g(x,y,z); phi=np.stack([1-0.3*G,0.6*G,0.5*G*x/SX,0.4*G*y/SY])
    Ay=0.35*g(x,y+hh/2,z)*(1+x/SX); Az=0.25*g(x,y,z+hh/2)*(y/SY)
    return phi,np.stack([0*G,hh*Ay,hh*Az])
def run(h,v=0.1,L=32.0,nu=0.12,g2=0.02,KF=1.0):
    N=int(round(L/h)); c=(N-1)/2; x=(np.arange(N)-c)*h; X,Y,Z=np.meshgrid(x,x,x,indexing='ij'); gam=1/np.sqrt(1-v*v)
    S=lagcov.build(N,h=h,nu=nu,g2=g2,KF=KF); 
    def conf(lam=1.0,gb=1.0): return fields(X*gb/lam,Y,Z,h)
    E=lambda ph,th: float(S['energy'](jnp.array(ph),jnp.array(th)))
    ph0,th0=conf(); M=E(ph0,th0); one=np.stack([np.full((N,N,N),np.sqrt(1-1e-3)),*[np.zeros((N,N,N))]*3]); Evac=E(one,np.zeros((3,N,N,N))); Mr=M-Evac
    e=0.005; dE=(E(*conf(1+e))-E(*conf(1-e)))/(2*e)
    phb,thb=conf(1.0,gam); dxf=lambda a:(np.roll(a,-1,axis=-3)-np.roll(a,1,axis=-3))/(2*h)
    vel=-v*dxf(phb); w=-v*dxf(thb)                      # theta_dot = -v d_x theta ; phi_dot = -v d_x phi (theta_x=0 -> D_x=d_x)
    q=(jnp.array(phb),jnp.array(thb)); vv=(jnp.array(vel),jnp.array(w))
    Hb=float(S['hamil'](q,vv)); pp,pi=S['pfun'](q,vv)
    # translation covariante : dphi = -D_x phi (central), dtheta_y = -h F_xy, dtheta_z=-h F_xz (plaquettes moyennees)
    Fxy=(thb[0]+np.roll(thb[1],-1,axis=0)-np.roll(thb[0],-1,axis=1)-thb[1])/h**2        # plaq(th,0,1) (th[0]=0)
    Fxz=(thb[0]+np.roll(thb[2],-1,axis=0)-np.roll(thb[0],-1,axis=2)-thb[2])/h**2
    Fxy=0.5*(Fxy+np.roll(Fxy,1,axis=0)); Fxz=0.5*(Fxz+np.roll(Fxz,1,axis=0))
    dphi=-dxf(phb); dth=np.stack([0*Fxy,-h*Fxy,-h*Fxz])
    P=float(jnp.sum(pp*jnp.array(dphi))+jnp.sum(pi*jnp.array(dth)))
    Pp=gam*v*(Mr-dE); Hp=gam*(Mr-v**2*dE)+Evac; gs=S['gauss'](q,vv)
    return dict(h=h,KF=KF,Mrel=Mr,dE=dE,P=P,Ppred=Pp,errP=P/Pp-1,H=Hb,Hpred=Hp,errH=Hb/Hp-1,gauss_max=float(jnp.abs(gs).max()))
if __name__=='__main__':
    for KF in (1.0,0.0):
        for h in (0.5,0.4,0.32):
            r=run(h,KF=KF); print(f"KF={KF:g} h={h}: Mrel={r['Mrel']:.3f} dE/dlnl={r['dE']:.3f} | P={r['P']:.5f} pred={r['Ppred']:.5f} err={r['errP']:+.2e} | H err={r['errH']:+.2e} | |Gauss|max={r['gauss_max']:.2e}",flush=True)
