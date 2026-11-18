"""Génère .github/workflows/pew-part186b-cycle-lent.yml (à lancer depuis la racine du dépôt).
part186b : 1 nœud de Hopf relaxé (part179 wk_T0) dans la boîte périodique 3D du modèle Z ; relaxation (a=1) -> contraction a:1->AMIN
-> relâchement (a->1, sans friction) -> évolution libre. Mesures : Q, deg, rho_mean (houle : oscillation de la moyenne de rho),
P0 (k=0) et Pk (k!=0), spec (7 bandes de |k|) de delta rho (écume). dyn186.py = dyn.py + MODE=cycle, spectre ; relâchement lent (TBACK=30) vs rapide (3)."""
runs = [("slow_a05", 0.5, 30), ("fast_a05", 0.5, 3), ("slow_a07", 0.7, 30), ("ctl_a10", 1.0, 30)]   # (tag, AMIN, TBACK) ; départ = nœud relaxé de part179 (wk_T0)
NCH = 3
ENV = {"MODE": "cycle", "INIT": "file", "INIT_FILE": "../../resultats/part179/wk_T0/wk_T0_champ_final.npy", "MU": "0.375", "E0": "0.25", "N": "64", "H": "0.5", "DT": "0.0025",
       "T0": "0.0", "TREL": "5", "TCOMP": "40", "TBACK": "30", "G0R": "0.5", "G0C": "0.02", "T_END": "120", "MEAS": "1.0",
       "SNAP": "1000", "TIME_BUDGET": "18900", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "SEED": "1", "NOEUDS": "0"}
def job(tag, amin, tback, c):
    full = f"p186b_{tag}"; env = dict(ENV, AMIN=str(amin), TBACK=str(tback))
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    needs = f"    needs: {tag}_r{c-1}\n    if: always() && needs.{tag}_r{c-1}.outputs.finished == 'false'\n" if c > 1 else ""
    guard = f"""      - name: Garde-fou - le checkpoint doit exister
        run: test -f ck_{full}.npz || (echo "CHECKPOINT ABSENT"; exit 1)
""" if c > 1 else ""
    return f"""  {tag}_r{c}:
    name: part186b {tag} chunk {c}/{NCH}
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
          key: p186b-{tag}-r{c}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}
          restore-keys: |
            p186b-{tag}-
      - name: Install dependencies
        run: python -m pip install -r requirements.txt scipy
{guard}      - name: Run or resume
        env:
{envs}
        run: python -u dyn186.py {full} 2>&1 | tee run_{full}_r{c}.log
      - name: Check whether finished
        id: check_done
        shell: bash
        run: |
          python - <<'PYCODE'
          import json, os
          st = json.load(open('{full}_state.json')); print(st)
          open(os.environ['GITHUB_OUTPUT'], 'a').write('finished=' + ('true' if st['finished'] else 'false') + '\\n')
          PYCODE
      - name: Commit results to resultats/
        if: always()
        continue-on-error: true
        shell: bash
        run: |
          OUT=../../resultats/part186b/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ 2>/dev/null || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part186b/{full}; git commit -q -m 'part186b {tag} chunk {c} : resultats' || exit 0
          ok=0; for i in $(seq 1 12); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part186b cycle contraction-relachement (houle / ecume)\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for r in runs:
    for c in range(1, NCH + 1): out.append(job(*r, c))
open(".github/workflows/pew-part186b-cycle-lent.yml", "w").write("".join(out)); print(len(runs), "runs")
