"""part191 : amplitude globale de la porteuse imposée (KW0 = rho_vac), amplitude moyenne rappelée (MEANC=1), couplage c = rho/rho_vac (CP=1), masse de l'amplitude locale mu_a = 0,375 (massive) ou 0,05 (presque sans masse).
Mesure : (porteuse) profil de delta rho / rho_vac autour du noeud, G_eff = delta(r_d) r_d / M en fonction de rho_vac ; (noeuds) masse M, taille, charge, stabilité en fonction de rho_vac."""
runs = [(f"m{mn}_r{rn}", mu, r) for mn, mu in (("0375", "0.375"), ("005", "0.05")) for rn, r in (("100", "1.0"), ("070", "0.7"), ("050", "0.5"))]
COMMON = {"N": "96", "NB": "96", "H": "0.5", "E0": "0.25", "DT": "0.005", "T0": "0", "MODE": "none", "G0": "0.5", "T_END": "40", "MEAS": "4.0", "SNAP": "1000", "TIME_BUDGET": "18900",
          "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "0", "KC": "0", "SEED": "1", "FINAL32": "1", "CP": "1", "MEANC": "1", "INIT": "file", "INIT_FILE": "init96.npy"}
def job(tag, mu, r):
    full = f"p191_{tag}"; env = dict(COMMON, MU=mu, KW0=r); envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    return f"""  {tag}:
    name: part191 {tag}
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
        run: python -u dyn186c.py {full} 2>&1 | tee run_{full}.log
      - name: Commit results to resultats/
        if: always()
        shell: bash
        run: |
          OUT=../../resultats/part191/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part191/{full}; git commit -q -m 'part191 {tag} : resultats' || exit 0
          ok=0; for i in $(seq 1 20); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part191 amplitude globale imposee, rigidite du milieu, G effectif et effets sur le noeud\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for t, m, r in runs: out.append(job(t, m, r))
open(".github/workflows/pew-part191-amplitude-imposee.yml", "w").write("".join(out)); print(len(runs), "jobs")
