"""Génère .github/workflows/pew-part186d-rho-aleatoire.yml (racine du dépôt).
part186d : comme part186c mais amplitude rho aléatoire au départ (NORM=0, RMS=1), boîte L=38,4 : N=96/h=0,4 (4 runs tau0=4,8 x 2 graines + témoin a=1) et N=128/h=0,3 (tau0=8, 1 graine, convergence en h)."""
runs = []   # (tag, dict d'environnement propre au run, nombre de tranches)
for tau0 in (4, 8):
    for sd in (1, 2):
        runs.append((f"A_t{tau0}_s{sd}", dict(N="96", H="0.4", TC0=str(tau0), XI0=str(tau0), T_END=str(8 * tau0), SEED=str(sd)), 3))   # boîte L=38,4 (h=0,4)
runs.append(("A_ctl_s1", dict(N="96", H="0.4", TC0="1000000", XI0="4", T_END="32", SEED="1"), 3))   # témoin a=1, sans friction
runs.append(("B_t8_s1", dict(N="128", H="0.3", TC0="8", XI0="8", T_END="64", SEED="1"), 8))   # même boîte L=38,4, maillage plus fin (h=0,3) : test de convergence
ENV = {"MODE": "hot", "NORM": "0", "RMS": "1.0", "INIT": "chaos", "MU": "0.375", "E0": "0.25", "DT": "0.0025", "T0": "0", "TH": "0", "TC": "0", "TSTART_H": "0", "G0": "0", "WARM": "0",
       "MEAS": "1.0", "SNAP": "1000", "TIME_BUDGET": "18900", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "1", "KC": "0"}
def job(tag, extra, c, NCH):
    full = f"p186d_{tag}"; env = dict(ENV, **extra)
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    needs = f"    needs: {tag}_r{c-1}\n    if: always() && needs.{tag}_r{c-1}.outputs.finished == 'false'\n" if c > 1 else ""
    guard = f"""      - name: Garde-fou - le checkpoint doit exister
        run: test -f ck_{full}.npz || (echo "CHECKPOINT ABSENT"; exit 1)
""" if c > 1 else ""
    return f"""  {tag}_r{c}:
    name: part186d {tag} chunk {c}/{NCH}
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
          key: p186d-{tag}-r{c}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}
          restore-keys: |
            p186d-{tag}-
      - name: Install dependencies
        run: python -m pip install -r requirements.txt scipy
{guard}      - name: Run or resume
        env:
{envs}
        run: python -u dyn186d.py {full} 2>&1 | tee run_{full}_r{c}.log
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
          OUT=../../resultats/part186d/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ 2>/dev/null || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part186d/{full}; git commit -q -m 'part186d {tag} chunk {c} : resultats' || exit 0
          ok=0; for i in $(seq 1 12); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part186d Kibble avec rho aleatoire, boite 38.4, h 0.4 / 0.3\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for r in runs:
    for c in range(1, r[2] + 1): out.append(job(r[0], r[1], c, r[2]))
open(".github/workflows/pew-part186d-rho-aleatoire.yml", "w").write("".join(out)); print(len(runs), "runs")
