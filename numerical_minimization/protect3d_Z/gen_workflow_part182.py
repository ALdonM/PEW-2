"""Génère .github/workflows/pew-part182-paires.yml : contrôle de part181 : h/2 (N=256, NB=128) et instantanés 3D à h=0,5."""
import sys
# (tag, orient, sep_mailles, opp)  -- sep en mailles (h = 0,5 : 24 -> 12 u, 40 -> 20 u)
# (tag, orient, sep_mailles_N128, opp, N, H, DT, NB, T_END, nch, snap)
runs = []
for o in ("pi_x", "pi_z", "pi2_x"):
    runs.append((f"h05snap_same_{o}_s24", o, 24, 0, 128, 0.5, 0.0025, 128, 40, 2, 2.5))
for o, opp, te in (("pi_x", 0, 28), ("pi_z", 0, 28), ("beta0", 0, 28), ("pi_x", 1, 20)):
    runs.append((f"h025_{'opp' if opp else 'same'}_{o}_s24", o, 24, opp, 256, 0.25, 0.00125, 128, te, 5, 1000))
ENV = {"MODE": "lang", "G0": "0.0", "G0R": "0.0", "MU": "0.375", "E0": "0.25", "N": "128", "H": "0.5", "T0": "10", "DT": "0.0025",
       "SNAP": "1000", "TIME_BUDGET": "19000", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "INIT": "file",
       "TON": "0", "T_END": "40", "MEAS": "0.5", "NOEUDS": "0",
       "TSTART": "0.0", "TQ": "0", "SEED": "1"}
def job(tag, o, sep, opp, N, H, DT, NB, TE, nch, snap, c):
    full = f"p182_{tag}"
    env = dict(ENV, EINJ='0', KC='0', INIT_FILE=f'pair_{tag}.npy', N=str(N), H=str(H), DT=str(DT), NB=str(NB), T_END=str(TE), SNAP=str(snap))
    big = N > 128
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    needs = f"    needs: {tag}_c{c-1}\n    if: always() && needs.{tag}_c{c-1}.outputs.finished == 'false'\n" if c > 1 else ""
    return f"""  {tag}_c{c}:
    name: part182 {tag} chunk {c}/{nch}
    runs-on: ubuntu-24.04
    timeout-minutes: 355
{needs}    outputs:
      finished: ${{{{ steps.check_done.outputs.finished }}}}
    defaults:
      run:
        working-directory: numerical_minimization/protect3d_Z
    steps:
      - uses: actions/checkout@v4
        with:
          ref: main
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: pip
          cache-dependency-path: numerical_minimization/protect3d_Z/requirements.txt
      - name: Restore checkpoint
        uses: actions/cache@v4
        with:
          path: |
            numerical_minimization/protect3d_Z/ck_{full}.npz
            numerical_minimization/protect3d_Z/{full}_history.jsonl
            numerical_minimization/protect3d_Z/{full}_params.json
            numerical_minimization/protect3d_Z/{full}_proj.npy
          key: p182-{tag}-c{c}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}
          restore-keys: |
            p182-{tag}-
      - name: Install dependencies
        run: python -m pip install -r requirements.txt
      - name: Fabrique la paire initiale
        run: python ../pairs_Z/make_pair.py pair_{tag}.npy {sep} {o} {opp}
      - name: Run or resume
        env:
{envs}
        run: python -u dyn.py {full} 2>&1 | tee run_{full}_c{c}.log
      - name: Check whether finished
        id: check_done
        shell: bash
        run: |
          python - <<'PYCODE'
          import json, os
          st = json.load(open('{full}_state.json')); print(st)
          open(os.environ['GITHUB_OUTPUT'], 'a').write('finished=' + ('true' if st['finished'] else 'false') + '\\n')
          PYCODE
      - name: Sauvegarde en artefact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: {full}-c{c}
          path: |
            numerical_minimization/protect3d_Z/{full}_history.jsonl
            numerical_minimization/protect3d_Z/{full}_params.json
            numerical_minimization/protect3d_Z/{full}_state.json
            numerical_minimization/protect3d_Z/{full}_champ_final.npy
            numerical_minimization/protect3d_Z/{full}_proj.npy
            numerical_minimization/protect3d_Z/snap_{full}_t*.npy
          if-no-files-found: ignore
      - name: Commit results to resultats/
        if: always()
        continue-on-error: true
        shell: bash
        run: |
          OUT=../../resultats/part182/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_proj.npy $OUT/; {'true' if big else 'cp ' + full + '_champ_final.npy $OUT/'} 2>/dev/null || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part182/{full}; git commit -q -m 'part182 {tag} chunk {c} : resultats' || exit 0
          ok=0; for i in $(seq 1 12); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part182 paires de noeuds - controle h/2 et instantanes 3D\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for r in runs:
    for c in range(1, r[9] + 1): out.append(job(*r, c))
open(".github/workflows/pew-part182-paires.yml", "w").write("".join(out))
print(len(runs), "runs,", "jobs prévus")
