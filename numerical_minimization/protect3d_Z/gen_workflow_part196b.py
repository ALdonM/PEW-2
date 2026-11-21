"""part196b : retard de phase d'une onde libre de la porteuse (paquet de phase k=3) traversant le creux d'un noeud, milieu fixe (WA=WR=WV=rv^-2, KT=rv^-4), eponge spherique, mu_a=0,05.
Pour rv in {1 ; 0,5} : K0 (noeud seul), Kw (noeud + onde), Vw (porteuse seule + onde) ; rv=0,5 : en plus Kb, Vb (eps 0,025, linearite). Depart du noeud : champ final de part194 / rv."""
RUNS = []   # (tag, rv, mode, eps, T_END)
for rv, T in ((1.0, 20), (0.5, 36)):
    rn = f"{int(round(rv*100)):03d}"
    RUNS += [(f"r{rn}_K0", rv, "K0", None, T), (f"r{rn}_Kw", rv, "Kw", 0.05, T), (f"r{rn}_Vw", rv, "Vw", 0.05, T)]
    if rv == 0.5: RUNS += [(f"r{rn}_Kb", rv, "Kw", 0.025, T), (f"r{rn}_Vb", rv, "Vw", 0.025, T)]
COMMON = {"N": "96", "NB": "96", "H": "0.5", "E0": "0.25", "DT": "0.005", "T0": "0", "MODE": "hot", "TC0": "1000000", "TH": "0", "TC": "0", "TSTART_H": "0", "G0": "0", "WARM": "0", "MU": "0.375", "MEAS": "4.0",
          "SNAP": "1000", "TIME_BUDGET": "18900", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "0", "KC": "0", "SEED": "1", "FINAL32": "1", "CP": "1", "MEANC": "0",
          "SPR0": "16", "SPRB": "24", "SPSIG": "0.3", "WK": "3.0", "WX0": "-8", "WSIG": "3"}
def job(tag, rv, mode, eps, T):
    full = f"p196b_{tag}"; rn = f"{int(round(rv*100)):03d}"
    env = dict(COMMON, KW0=str(rv), WA=str(round(rv**-2, 6)), WR=str(round(rv**-2, 6)), WV=str(round(rv**-2, 6)), KT=str(round(rv**-4, 6)), WC=str(rv), T_END=str(T))
    if mode == "Vw": env.update(INIT="vac", WAVE="phase", WEPS=str(eps))
    else:
        env.update(INIT="file", INIT_FILE="init96.npy")
        if mode == "Kw": env.update(WAVE="phase", WEPS=str(eps))
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    src = "../../resultats/part193/p193_s2_r100/p193_s2_r100_champ_final.npy" if rv == 1.0 else "../../resultats/part194/p194_m0375_r050/p194_m0375_r050_champ_final.npy"
    prep = "" if mode == "Vw" else f"""      - name: Depart du noeud a l'equilibre (part194)
        run: python -c "import numpy as np; np.save('init96.npy', (np.load('{src}')/{rv}).astype(np.float32))"
"""
    return f"""  {tag}:
    name: part196b {tag}
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
{prep}      - name: Run
        env:
{envs}
        run: python -u dyn186c.py {full} 2>&1 | tee run_{full}.log
      - name: Commit results to resultats/
        if: always()
        shell: bash
        run: |
          OUT=../../resultats/part196b/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part196b/{full}; git commit -q -m 'part196b {tag} : resultats' || exit 0
          ok=0; for i in $(seq 1 20); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part196b retard de phase d'une onde libre de la porteuse, milieu fixe\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for r in RUNS: out.append(job(*r))
open(".github/workflows/pew-part196b-onde-libre-noeud-faible.yml", "w").write("".join(out)); print(len(RUNS), "jobs")
