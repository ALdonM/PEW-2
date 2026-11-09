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
def degree(phi, zeta0=None, carte=False):
    z = phi / np.sqrt((phi**2).sum(0))
    if zeta0 is None:
        zeta0 = np.array([0.31, -0.52, 0.64, 0.47]); zeta0 /= np.linalg.norm(zeta0)
    total = 0; cmap = np.zeros(z.shape[1:], dtype=np.int32)
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
        if carte:
            c = np.zeros(M, dtype=np.int32); c[hit] = (parity(p)*sgn).astype(np.int32); cmap += c.reshape(z.shape[1:])
    if carte: return int(round(total)), cmap
    return int(round(total))

def compte_noeuds(phi):
    """Compte nœuds et anti-nœuds : préimages signées de l'antipode du vide moyen, sommées par région « retournée » (u·m < 0, dilatée d'une maille).
    N'a de sens que si le fond est ordonné (order proche de 1)."""
    from scipy import ndimage
    z = phi / np.sqrt((phi**2).sum(0)); m = z.reshape(4, -1).mean(1); m = m / (np.linalg.norm(m) + 1e-30)
    zt = -m + np.array([0.031, -0.017, 0.023, 0.011]); zt = zt / np.linalg.norm(zt)   # perturbation : évite une valeur dégénérée
    tot, cmap = degree(phi, zt, carte=True)
    inv = ndimage.binary_dilation(np.einsum('a...,a->...', z, m) < 0, iterations=1, structure=np.ones((3,3,3)))
    lab, n = ndimage.label(inv, structure=np.ones((3,3,3)))
    if n == 0: return dict(n_plus=0, n_moins=0, hors=int(cmap.sum()), charges=[])
    s = ndimage.sum(cmap, lab, np.arange(1, n+1)).round().astype(int)
    nz = s[s != 0]
    return dict(n_plus=int((nz > 0).sum()), n_moins=int((nz < 0).sum()), hors=int(cmap[lab == 0].sum()), charges=sorted(nz.tolist())[:20])
if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        phi=np.load(f).astype(np.float64)
        print(f.split('/')[-2] if '/' in f else f, "degré =", degree(phi), "| autre ζ0 :", degree(phi, np.array([-0.2,0.1,0.9,-0.38])/np.linalg.norm([-0.2,0.1,0.9,-0.38])))
