"""Génère .github/workflows/pew-part186c-kibble.yml (à lancer depuis la racine du dépôt).
part186c : départ à la Kibble (direction n aléatoire de corrélation XI0 = tau0 ~ horizon, amplitude du vide partout), expansion radiative a = sqrt(tau/tau0), friction de Hubble 1.5/tau, sans bain.
Témoin : a = 1 sans friction. 64^3, h = 0,5, e0 = 0,25, mu = 0,375 (nœud R ~ 5). Comptage nœuds/anti-nœuds, charge nette, vs tau0."""
runs = []   # (tag, dict d'environnement propre au run)
for tau0 in (2, 4, 8):
    for sd in (1, 2):
        runs.append((f"exp_t{tau0}_s{sd}", dict(TC0=str(tau0), XI0=str(tau0), T_END=str(8 * tau0), SEED=str(sd))))
for sd in (1, 2):
    runs.append((f"ctl_s{sd}", dict(TC0="1000000", XI0="4", T_END="32", SEED=str(sd))))   # témoin : a = 1, pas de friction de Hubble
NCH = 2
ENV = {"MODE": "hot", "NORM": "1", "INIT": "chaos", "MU": "0.375", "E0": "0.25", "N": "64", "H": "0.5", "DT": "0.0025", "T0": "0", "TH": "0", "TC": "0", "TSTART_H": "0", "G0": "0", "WARM": "0",
       "MEAS": "1.0", "SNAP": "1000", "TIME_BUDGET": "18900", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "1", "KC": "0"}
def job(tag, extra, c):
    full = f"p186c_{tag}"; env = dict(ENV, **extra)
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    needs = f"    needs: {tag}_r{c-1}\n    if: always() && needs.{tag}_r{c-1}.outputs.finished == 'false'\n" if c > 1 else ""
    guard = f"""      - name: Garde-fou - le checkpoint doit exister
        run: test -f ck_{full}.npz || (echo "CHECKPOINT ABSENT"; exit 1)
""" if c > 1 else ""
    return f"""  {tag}_r{c}:
    name: part186c {tag} chunk {c}/{NCH}
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
          key: p186c-{tag}-r{c}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}
          restore-keys: |
            p186c-{tag}-
      - name: Install dependencies
        run: python -m pip install -r requirements.txt scipy
{guard}      - name: Run or resume
        env:
{envs}
        run: python -u dyn186c.py {full} 2>&1 | tee run_{full}_r{c}.log
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
          OUT=../../resultats/part186c/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ 2>/dev/null || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part186c/{full}; git commit -q -m 'part186c {tag} chunk {c} : resultats' || exit 0
          ok=0; for i in $(seq 1 12); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part186c trempe a la Kibble (champ aleatoire a l echelle de l horizon, expansion)\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for r in runs:
    for c in range(1, NCH + 1): out.append(job(r[0], r[1], c))
open(".github/workflows/pew-part186c-kibble.yml", "w").write("".join(out)); print(len(runs), "runs")
