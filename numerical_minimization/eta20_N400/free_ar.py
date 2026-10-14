"""Minimisation PEW sur le rayon d'anneau libre.

Usage: python -u free_ar.py ETA_MAX N INIT_TAG OUT_TAG
MAXITER = nombre maximal d'iterations L-BFGS-B pour cette execution.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from energy_native_staggered import build_grid, energy_and_grad_native_staggered

ETA_MAX = float(sys.argv[1])
N = int(sys.argv[2])
INIT = sys.argv[3]
TAG = sys.argv[4]
MAXITER = int(os.environ.get("MAXITER", "10000"))
CK = int(os.environ.get("CHECKPOINT_EVERY", "250"))
if MAXITER < 1 or CK < 1:
    raise ValueError("MAXITER et CHECKPOINT_EVERY doivent etre >= 1")

f0 = e0 = 1.0
f_core = np.pi
eta, xi, deta, dxi, ETA, XI = build_grid(N, N, ETA_MAX)
n = N * N
ST = f"free_{TAG}_state.json"
H = f"free_{TAG}_history.jsonl"
FCK = f"f_free_{TAG}_ckpt.npy"
DCK = f"dchi_free_{TAG}_ckpt.npy"

have_f, have_d = os.path.exists(FCK), os.path.exists(DCK)
if have_f != have_d:
    raise RuntimeError(f"Checkpoint incomplet : verifier {FCK} et {DCK}.")
if have_f:
    if not os.path.exists(ST):
        raise RuntimeError(f"{ST} absent alors que le checkpoint existe.")
    f, d = np.load(FCK), np.load(DCK)
    with open(ST, encoding="utf-8") as fh:
        total_nit = int(json.load(fh)["nit_ckpt"])
else:
    f_init, d_init = f"f_{INIT}_ckpt.npy", f"dchi_{INIT}_ckpt.npy"
    if not (os.path.exists(f_init) and os.path.exists(d_init)):
        raise FileNotFoundError(f"Fichiers initiaux manquants : {f_init}, {d_init}")
    f, d = np.load(f_init), np.load(d_init)
    total_nit = 0
if f.shape != (N, N) or d.shape != (N, N):
    raise ValueError(f"Dimensions inattendues : f={f.shape}, dchi={d.shape}, attendu {(N,N)}")

xcur = np.concatenate([f.ravel(), d.ravel()])
info = {}
run_nit = 0  # iterations L-BFGS-B dans cette execution
last_saved_nit = None
last_history_it = -1

if os.path.exists(H):
    with open(H, encoding="utf-8") as fh:
        for line in fh:
            try:
                last_history_it = int(json.loads(line)["it"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass


def fg(x):
    E1, g1 = energy_and_grad_native_staggered(
        x, ETA, XI, deta, dxi, 1.0, f0, e0, f_core, N, N
    )
    Ea, ga = energy_and_grad_native_staggered(
        x, ETA, XI, deta, dxi, 2.0, f0, e0, f_core, N, N
    )
    E1, Ea = float(E1), float(Ea)
    g1, ga = np.asarray(g1, float), np.asarray(ga, float)
    E4 = (2.0 * E1 - Ea) / 1.5
    E2 = E1 - E4
    dE4 = (2.0 * g1 - ga) / 1.5
    dE2 = g1 - dE4
    if not (np.isfinite(E2) and np.isfinite(E4)) or E2 <= 1e-12 or E4 <= 1e-12:
        return 1e6, np.zeros_like(g1)
    G = 2.0 * np.sqrt(E2 * E4)
    grad = np.sqrt(E4 / E2) * dE2 + np.sqrt(E2 / E4) * dE4
    info.update(E2=E2, E4=E4, astar=np.sqrt(E4 / E2), E1=E1)
    return float(G), grad


def save_checkpoint(x, write_history=True):
    global last_saved_nit, last_history_it
    G, grad = fg(x)
    np.save(FCK, x[:n].reshape(N, N))
    np.save(DCK, x[n:].reshape(N, N))
    with open(ST, "w", encoding="utf-8") as fh:
        json.dump({"nit_ckpt": int(total_nit)}, fh)
        fh.write("\n")
    # Ne pas ajouter deux lignes d'historique pour le meme compteur.
    if write_history and total_nit > last_history_it:
        record = {
            "it": int(total_nit), "G": float(G),
            "E2": float(info["E2"]), "E4": float(info["E4"]),
            "astar": float(info["astar"]), "E_ar1": float(info["E1"]),
            "rms": float(np.sqrt(np.mean(np.asarray(grad) ** 2))),
            "t": time.time(),
        }
        with open(H, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        last_history_it = total_nit
        print(record, flush=True)
    last_saved_nit = total_nit


def callback(xk):
    global run_nit, total_nit
    run_nit += 1
    total_nit += 1
    if total_nit % CK == 0:
        save_checkpoint(xk)


short = 0
start_time = time.time()
while run_nit < MAXITER:
    remaining = MAXITER - run_nit
    res = minimize(
        fg, xcur, jac=True, method="L-BFGS-B", callback=callback,
        options=dict(maxiter=remaining, maxfun=max(2 * remaining, 100),
                     gtol=1e-14, ftol=1e-17, maxcor=20, maxls=100),
    )
    print("L-BFGS-B stop:", res.message,
          "nit_this_call", res.nit, "nit_this_run", run_nit,
          "nit_cumulative", total_nit, "nfev", res.nfev,
          "fun", res.fun, flush=True)
    xcur = res.x
    if run_nit >= MAXITER:
        break

    if res.nit < 5:
        fx, gx = fg(xcur)
        step = 1e-3 / max(np.linalg.norm(gx), 1e-30)
        accepted = 0
        for _ in range(60):
            xn = xcur - step * gx
            fn, gn = fg(xn)
            if np.isfinite(fn) and fn < fx:
                xcur, fx, gx = xn, fn, gn
                step *= 1.5
                accepted += 1
            else:
                step *= 0.5
        print("kick GD ->", float(fx), "accepted_steps", accepted, flush=True)
        # Sauvegarde aussi le champ après un kick, même si nit L-BFGS-B
        # n'a pas progressé. Les kicks ne sont pas comptés comme iterations L-BFGS-B.
        save_checkpoint(xcur, write_history=False)
        short = short + 1 if res.nit < 5 else 0
        if short >= 3:
            print("Arret : trois appels L-BFGS-B successifs ont avance de moins de 5 iterations.", flush=True)
            break
    else:
        short = 0

# Sauvegarde finale : le checkpoint periodique peut etre en retard de CK-1 pas.
if last_saved_nit != total_nit or not os.path.exists(ST):
    save_checkpoint(xcur)
Gfinal, gfinal = fg(xcur)
print("FINAL", {
    "nit_this_run": int(run_nit), "nit_cumulative": int(total_nit),
    "G": float(Gfinal), "E2": float(info["E2"]), "E4": float(info["E4"]),
    "astar": float(info["astar"]), "E_ar1": float(info["E1"]),
    "rms": float(np.sqrt(np.mean(np.asarray(gfinal) ** 2))),
    "elapsed_s": float(time.time() - start_time),
}, flush=True)
