"""part192 : amplitude globale de la porteuse imposée (KW0 = rho_vac), amplitude moyenne rappelée (MEANC=1), couplage c = rho/rho_vac (CP=1), masse de l'amplitude locale mu_a = 0,375 (massive) ou 0,05 (presque sans masse).
Mesure : (porteuse) profil de delta rho / rho_vac autour du noeud, G_eff = delta(r_d) r_d / M en fonction de rho_vac ; (noeuds) masse M, taille, charge, stabilité en fonction de rho_vac."""
# part192 : raideur du noeud (angulaire) fixe en absolu : WA = rv^-2 ; raideur radiale de la porteuse WR = rv^-s (s=0 porteuse faible=molle, s=1, s=2 tout fixe)
runs = [(f"s{sx}_r{rn}", "0.05", r, sx) for sx in (0, 1, 2) for rn, r in (("070", "0.7"), ("050", "0.5"))]
COMMON = {"N": "96", "NB": "96", "H": "0.5", "E0": "0.25", "DT": "0.005", "T0": "0", "MODE": "none", "G0": "0.5", "T_END": "40", "MEAS": "4.0", "SNAP": "1000", "TIME_BUDGET": "18900",
          "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "0", "KC": "0", "SEED": "1", "FINAL32": "1", "CP": "1", "MEANC": "1", "INIT": "file", "INIT_FILE": "init96.npy"}
def job(tag, mu, r, sx):
    full = f"p192_{tag}"; env = dict(COMMON, MU=mu, KW0=r, WA=str(round(float(r)**-2, 6)), WR=str(round(float(r)**-sx, 6))); envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    return f"""  {tag}:
    name: part192 {tag}
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
          OUT=../../resultats/part192/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part192/{full}; git commit -q -m 'part192 {tag} : resultats' || exit 0
          ok=0; for i in $(seq 1 20); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part192 amplitude globale imposee, rigidite du milieu, G effectif et effets sur le noeud\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for t, m, r, sx in runs: out.append(job(t, m, r, sx))
open(".github/workflows/pew-part192-rigidite-milieu.yml", "w").write("".join(out)); print(len(runs), "jobs")
