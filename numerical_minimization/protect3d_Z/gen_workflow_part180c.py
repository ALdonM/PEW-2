"""Génère .github/workflows/pew-part180c-seuil-h2.yml : seuil de l'énergie des ondes porteuses injectées dans un nœud relaxé (système fermé, G0 = 0)."""
import sys
# (tag, EINJ, KC, N, H, DT, NB, T_END, nchunks, seed)
runs = [
 ("hk15_e8770_s1", 8770, 1.5, 128, 0.25, 0.00125, 64, 24, 6, 1),
 ("hk15_e26300_s1", 26300, 1.5, 128, 0.25, 0.00125, 64, 24, 6, 1),
 ("hk6_e26300_s1", 26300, 6.0, 128, 0.25, 0.00125, 64, 24, 6, 1),
 ("hk15_e2630_s2", 2630, 1.5, 128, 0.25, 0.00125, 64, 24, 6, 2),
 ("hk15_e2630_s3", 2630, 1.5, 128, 0.25, 0.00125, 64, 24, 6, 3),
 ("hk15_e8770_s2", 8770, 1.5, 128, 0.25, 0.00125, 64, 24, 6, 2),
]
ENV = {"MODE": "lang", "G0": "0.0", "G0R": "0.0", "MU": "0.375", "E0": "0.25", "N": "64", "H": "0.5", "T0": "10", "DT": "0.0025",
       "SNAP": "1000", "TIME_BUDGET": "19000", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "INIT": "file",
       "INIT_FILE": "../../resultats/part179/wk_T0/wk_T0_champ_final.npy", "TON": "2", "T_END": "62", "MEAS": "0.5", "NOEUDS": "0",
       "TSTART": "0.0", "TQ": "0", "SEED": "1"}
def job(tag, e, kc, N, H, DT, NB, TE, nch, sd, c):
    full = f"p180c_{tag}"
    env = dict(ENV, EINJ=str(e), KC=str(kc), N=str(N), H=str(H), DT=str(DT), NB=str(NB), T_END=str(TE), SEED=str(sd))
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    needs = f"    needs: {tag}_c{c-1}\n    if: always() && needs.{tag}_c{c-1}.outputs.finished == 'false'\n" if c > 1 else ""
    return f"""  {tag}_c{c}:
    name: part180c {tag} chunk {c}/{nch}
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
          key: p180c-{tag}-c{c}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}
          restore-keys: |
            p180c-{tag}-
      - name: Install dependencies
        run: python -m pip install -r requirements.txt
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
          if-no-files-found: ignore
      - name: Commit results to resultats/
        if: always()
        continue-on-error: true
        shell: bash
        run: |
          OUT=../../resultats/part180c/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ 2>/dev/null || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part180c/{full}; git commit -q -m 'part180c {tag} chunk {c} : resultats' || exit 0
          ok=0; for i in $(seq 1 12); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part180c seuil a h/2 energies elevees et graines\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for tag, e, kc, N, H, DT, NB, TE, nch, sd in runs:
    for c in range(1, nch + 1): out.append(job(tag, e, kc, N, H, DT, NB, TE, nch, sd, c))
open(".github/workflows/pew-part180c-seuil-h2.yml", "w").write("".join(out))
print(len(runs), "runs,", "jobs prévus")
