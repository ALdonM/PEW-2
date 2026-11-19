"""part190 : amplitude de la porteuse sans masse (ou presque). Un noeud relaxé sous couplage c=rho (CP=1) dans une boîte 96^3, L=48, pour MU = masse de l'amplitude (mu_a) = 0,375 (référence) / 0,1 / 0,05 / 0,02.
Mesure : profil rho(r)-1 (pente log-log, ajustement rho_inf + A/r). V = mu^2(rho^2+1/(rho^2+d2)-2) : mu fixe la masse de l'amplitude m = 2 sqrt(2) mu."""
runs = [("m010b", "0.1"), ("m005b", "0.05")]
COMMON = {"N": "96", "NB": "96", "H": "0.5", "E0": "0.25", "DT": "0.005", "T0": "0", "MODE": "none", "G0": "0.5", "T_END": "24", "MEAS": "4.0", "SNAP": "1000", "TIME_BUDGET": "18900",
          "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "0", "KC": "0", "SEED": "1", "FINAL32": "1", "CP": "1", "INIT": "file", "INIT_FILE": "init96.npy"}
def job(tag, mu):
    full = f"p190_{tag}"; env = dict(COMMON, MU=mu); envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    return f"""  {tag}:
    name: part190 {tag}
    runs-on: ubuntu-24.04
    timeout-minutes: 355
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
      - name: Install dependencies
        run: python -m pip install -r requirements.txt scipy
      - name: Preparer le noeud dans la boite 96
        run: python prep_embed96.py ../../resultats/part179/wk_T0/wk_T0_champ_final.npy init96.npy 96
      - name: Run
        env:
{envs}
        run: |
          python -u dyn186c.py {full} 2>&1 | tee run_{full}.log
          python - <<'PYCODE'
          import json; st = json.load(open('{full}_state.json')); print(st)
          PYCODE
      - name: Commit results to resultats/
        if: always()
        shell: bash
        run: |
          OUT=../../resultats/part190/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part190/{full}; git commit -q -m 'part190b {tag} : resultats'
          ok=0; for i in $(seq 1 20); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part190b secours (T_END=24) amplitude de la porteuse sans masse (profil rho-1 autour du noeud)\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for t, m in runs: out.append(job(t, m))
open(".github/workflows/pew-part190b-secours.yml", "w").write("".join(out)); print(len(runs), "jobs")
