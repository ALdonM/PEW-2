import numpy as np, sys
fo=np.load(sys.argv[1]); do=np.load(sys.argv[2]); N2=int(sys.argv[3]); ETA=float(sys.argv[4])
No=fo.shape[0]
def cent(N,L): return (np.arange(N)+0.5)*L/N
eo,xo=cent(No,ETA),cent(No,2*np.pi); en,xn=cent(N2,ETA),cent(N2,2*np.pi)
def interp_eta(A):  # A[No,No] -> [N2,No], lineaire, bords bloques
    out=np.empty((N2,A.shape[1]))
    for j in range(A.shape[1]): out[:,j]=np.interp(en,eo,A[:,j])
    return out
def interp_xi(A,periodic):
    out=np.empty((A.shape[0],N2))
    for i in range(A.shape[0]):
        if periodic:
            xe=np.concatenate([xo[-1:]-2*np.pi,xo,xo[:1]+2*np.pi]); ae=np.concatenate([A[i,-1:],A[i],A[i,:1]])
            out[i]=np.interp(xn,xe,ae)
        else: out[i]=np.interp(xn,xo,A[i])   # bords bloques : conserve le saut au joint
    return out
L=np.log(np.maximum(np.pi-fo,1e-300))
fn=np.pi-np.exp(interp_xi(interp_eta(L),True))
dn=interp_xi(interp_eta(do),False)
np.save(sys.argv[5],fn); np.save(sys.argv[6],dn); print(fn.shape,dn.shape,np.pi-fn.max() if False else fn.max(),dn.min(),dn.max())
