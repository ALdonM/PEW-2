"""part181 -- ansatz produit de deux noeuds du modele Z (Z = Z1 + Z2 j, produit quaternionique), orientation relative par conjugaison.
Charge Q = degre de zeta : S^3 -> S^3 (additive pour un produit de deux champs -> 1 au vide)."""
import numpy as np
def to_q(phi):  # (4,...) -> Z1, Z2 complexes
    return phi[0] + 1j*phi[1], phi[2] + 1j*phi[3]
def from_q(Z1, Z2): return np.stack([Z1.real, Z1.imag, Z2.real, Z2.imag])
def qmul(A, B):
    (Z1, Z2), (W1, W2) = to_q(A), to_q(B)
    return from_q(Z1*W1 - Z2*np.conj(W2), Z1*W2 + Z2*np.conj(W1))
def qconj(A):
    Z1, Z2 = to_q(A); return from_q(np.conj(Z1), -Z2)
def unit_q(axis, beta):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    s = np.sin(beta/2); return np.array([np.cos(beta/2), s*axis[0], s*axis[1], s*axis[2]])
def conj_by(phi, g):  # g phi g^-1, g quaternion constant (4,)
    G = g.reshape(4, *([1]*(phi.ndim-1))) * np.ones_like(phi)
    return qmul(qmul(G, phi), qconj(G))
def embed(phi, N, centre):
    """place un noeud (n^3, centre au milieu) dans un reseau N^3 rempli du vide (1,0,0,0), centre = (cx,cy,cz) en mailles"""
    n = phi.shape[1]; out = np.zeros((4, N, N, N)); out[0] = 1.0
    c0 = (n-1)/2
    sh = [int(round(centre[i] - c0)) for i in range(3)]
    blk = phi.astype(float)
    # on remplace le vide par la valeur du noeud sur le bloc (le bord du noeud vaut le vide a 1e-3 pres)
    idx = [(np.arange(n) + sh[i]) % N for i in range(3)]
    out[:, idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]] = blk
    return out
def pair(knot, N, sep_cells, g=None, axis_sep=0, orient_B=None):
    c = (N-1)/2; ca = [c]*3; cb = [c]*3
    ca[axis_sep] -= sep_cells/2; cb[axis_sep] += sep_cells/2
    A = embed(knot, N, ca); B = embed(knot, N, cb)
    if g is not None: B = conj_by(B, g)
    P = qmul(A, B); return P
