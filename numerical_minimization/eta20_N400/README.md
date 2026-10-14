# PEW numerical minimization: eta_max=20, N=400

This directory is isolated from the theory papers and the HTML demos. It contains the native-staggered energy functional and a resumable L-BFGS-B optimization.

## Reproducibility

- Grid: `N_eta=N_xi=400`, `eta_max=20`.
- Boundary/core parameter: `f_core=pi`; `f0=e0=1`.
- Scale minimization: `E(a)=E2*a+E4/a`, `astar=sqrt(E4/E2)`, `G=2*sqrt(E2*E4)`.
- The starting fields are interpolated from the verified `eta_max=16, N=400` checkpoint, then smoothly extended to eta=20. The original eta=16 checkpoint is preserved as `f_eta16.npy` and `dchi_eta16.npy`.
- Iteration counts are written only from actual SciPy iterations. Checkpoints and JSONL history are updated every 250 iterations.

## Files

- `free_ar.py`: resumable optimizer.
- `energy_native_staggered.py`: energy functional and JAX gradient.
- `f_eta20_init_ckpt.npy`, `dchi_eta20_init_ckpt.npy`: interpolated starting fields.
- `f_free_eta20_N400_ckpt.npy`, `dchi_free_eta20_N400_ckpt.npy`, `free_eta20_N400_state.json`, `free_eta20_N400_history.jsonl`: current checkpoint and history, when present.
- `.github/workflows/pew-eta20-n400.yml`: manually triggerable GitHub Actions job; it uploads the checkpoint/history as an artifact.

The numerical values are optimizer outputs, not by themselves a derivation of physical constants.
