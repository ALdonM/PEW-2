"""Degré exact (entier) de ζ : T³ → S³ sur réseau, par comptage signé des préimages d'un point ζ0 (application linéaire
par morceaux sur une triangulation de Kuhn, 6 tétraèdres par cube, puis projection radiale). Entier par construction."""
import numpy as np, itertools
PERMS = list(itertools.permutations(range(3)))
def parity(p):
    p=list(p); s=1
    for i in range(3):
        for j in range(i+1,3):
            if p[i]>p[j]: s=-s
    return s
def degree(phi, zeta0=None):
    z = phi / np.sqrt((phi**2).sum(0))
    if zeta0 is None:
        zeta0 = np.array([0.31, -0.52, 0.64, 0.47]); zeta0 /= np.linalg.norm(zeta0)
    total = 0
    for p in PERMS:
        # sommets : v0 = x, v1 = x+e_p0, v2 = v1+e_p1, v3 = v2+e_p2
        shifts=[(0,0,0)]; cur=[0,0,0]
        for ax in p:
            cur=cur.copy(); cur[ax]+=1; shifts.append(tuple(cur))
        V = np.stack([np.roll(z, shift=tuple(-s for s in sh), axis=(1,2,3)) for sh in shifts])  # (4 sommets, 4 comp, N,N,N)
        V = V.reshape(4,4,-1).transpose(2,1,0)          # (M, 4 comp, 4 sommets)
        M = V.shape[0]
        A = np.zeros((M,5,5)); A[:,:4,:4]=V; A[:,:4,4]=-zeta0; A[:,4,:4]=1.0
        b = np.zeros((M,5)); b[:,4]=1.0
        ok = np.abs(np.linalg.det(A))>1e-14
        sol = np.zeros((M,5)); sol[ok]=np.linalg.solve(A[ok], b[ok][...,None])[...,0]
        hit = ok & (sol[:,:4]>=0).all(1) & (sol[:,4]>0)
        sgn = np.sign(np.linalg.det(V[hit]))
        total += parity(p)*sgn.sum()
    return int(round(total))
if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        phi=np.load(f).astype(np.float64)
        print(f.split('/')[-2] if '/' in f else f, "degré =", degree(phi), "| autre ζ0 :", degree(phi, np.array([-0.2,0.1,0.9,-0.38])/np.linalg.norm([-0.2,0.1,0.9,-0.38])))
