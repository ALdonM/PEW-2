"""Génère .github/workflows/pew-part189-couplage-c-rho.yml. part189 : couplage vitesse de phase locale c = rho/rho_vac (E2 -> rho^(2CP)|grad Z|^2, CP=1).
Étapes : relax (nœud part179 relaxé avec friction sous la nouvelle énergie) -> K0 (nœud seul), Kp/Vp (houle de phase k=3), Ka/Va (houle d'amplitude k=3)."""
KF0 = "../../resultats/part179/wk_T0/wk_T0_champ_final.npy"; KF1 = "../../resultats/part189/p189_relax/p189_relax_champ_final.npy"
COMMON = {"N": "64", "H": "0.5", "MU": "0.375", "E0": "0.25", "DT": "0.0025", "T0": "0", "MEAS": "2.0", "SNAP": "1000", "TIME_BUDGET": "18900", "JAX_PLATFORM_NAME": "cpu",
          "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "0", "KC": "0", "SEED": "1", "FINAL32": "1", "CP": "1"}
HOT = {"MODE": "hot", "TC0": "1000000", "TH": "0", "TC": "0", "TSTART_H": "0", "G0": "0", "WARM": "0", "T_END": "22", "WK": "3.0", "WX0": "-10", "WSIG": "4", "INIT": "file", "INIT_FILE": KF1}
RELAX = {"MODE": "none", "G0": "0.5", "T_END": "20", "INIT": "file", "INIT_FILE": KF0, "NB": "64"}
jobs = [("relax", RELAX, None), ("K0", HOT, None), ("Kp", dict(HOT, WAVE="phase", WEPS="0.05"), 1), ("Vp", dict(HOT, WAVE="phase", WEPS="0.05", INIT="vac"), 1),
        ("Ka", dict(HOT, WAVE="amp", WEPS="0.05"), 1), ("Va", dict(HOT, WAVE="amp", WEPS="0.05", INIT="vac"), 1)]
def job(tag, extra):
    full = f"p189_{tag}"; env = dict(COMMON, **extra)
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    needs = "" if tag == "relax" else "    needs: relax\n"
    return f"""  {tag}:
    name: part189 {tag}
    runs-on: ubuntu-24.04
    timeout-minutes: 355
{needs}    defaults:
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
      - name: Install dependencies
        run: python -m pip install -r requirements.txt scipy
      - name: Run
        env:
{envs}
        run: |
          python -u dyn186c.py {full} 2>&1 | tee run_{full}.log
          python - <<'PYCODE'
          import json; st = json.load(open('{full}_state.json')); print(st); assert st['finished']
          PYCODE
      - name: Commit results to resultats/
        shell: bash
        run: |
          OUT=../../resultats/part189/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part189/{full}; git commit -q -m 'part189 {tag} : resultats'
          ok=0; for i in $(seq 1 20); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part189 couplage vitesse de phase c = rho/rho_vac (iceberg)\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for t, e, _ in jobs: out.append(job(t, e))
open(".github/workflows/pew-part189-couplage-c-rho.yml", "w").write("".join(out)); print(len(jobs), "jobs")
