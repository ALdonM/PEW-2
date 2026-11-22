"""part200 : retard de phase d'une onde libre de la porteuse (paquet de phase k=3) traversant le creux d'un noeud, milieu fixe (WA=WR=WV=rv^-2, KT=rv^-4), eponge spherique, mu_a=0,05.
Pour rv in {1 ; 0,5} : K0 (noeud seul), Kw (noeud + onde), Vw (porteuse seule + onde) ; rv=0,5 : en plus Kb, Vb (eps 0,025, linearite). Depart du noeud : champ final de part194 / rv."""
# (tag, genre, parametre) ; genre "fall" : GRAD=param, T=18 ; genre "wave" : eps=param, k=3, T=20 (depart : noeud seul de part196b)
RUNS = [("f_g0", "fall", 0.0), ("f_gp10", "fall", 0.01), ("f_gp05", "fall", 0.005), ("w_Kw", "wave", 0.05), ("w_Kb", "wave", 0.025)]
COMMON = {"N": "96", "NB": "96", "H": "0.5", "E0": "0.25", "DT": "0.005", "T0": "0", "MODE": "hot", "TC0": "1000000", "TH": "0", "TC": "0", "TSTART_H": "0", "G0": "0", "WARM": "0", "MU": "0.375", "MEAS": "2.0",
          "SNAP": "1000", "TIME_BUDGET": "18900", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "0", "KC": "0", "SEED": "1", "FINAL32": "1", "CP": "1", "MEANC": "0",
          "SPR0": "16", "SPRB": "24", "SPSIG": "0.3", "KW0": "1", "WA": "1", "WR": "1", "WV": "1", "KT": "1", "KF": "1", "NOEUDS": "0"}
def job(tag, kind, p):
    full = f"p200_{tag}"
    if kind == "fall":
        env = dict(COMMON, GRAD=str(p), INIT="file", INIT_FILE="init96.npy", T_END="18")
        src = "../../resultats/part193/p193_s2_r100/p193_s2_r100_champ_final.npy"
        prep = f"python -c \"import numpy as np; f=np.load('{src}').astype(np.float32); x=(np.arange(96)-48+.5)*.5; np.save('init96.npy', (f*(1+{p}*x)[None,:,None,None]).astype(np.float32))\""
    else:
        env = dict(COMMON, INIT="file", INIT_FILE="init96.npy", T_END="20", WAVE="phase", WEPS=str(p), WK="3.0", WX0="-8", WSIG="3", WC="1")
        src = "../../resultats/part196b/p196b_r100_K0/p196b_r100_K0_champ_final.npy"
        prep = f"python -c \"import numpy as np; np.save('init96.npy', np.load('{src}').astype(np.float32))\""
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    return f"""  {tag}:
    name: part200 {tag}
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
      - name: Depart
        run: {prep}
      - name: Run
        env:
{envs}
        run: python -u dyn186c.py {full} 2>&1 | tee run_{full}.log
      - name: Commit results to resultats/
        if: always()
        shell: bash
        run: |
          OUT=../../resultats/part200/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part200/{full}; git commit -q -m 'part200 {tag} : resultats' || exit 0
          ok=0; for i in $(seq 1 20); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part200 Faddeev temporel - chute libre (gradient impose) et retard d'onde libre (k=3)\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for r in RUNS: out.append(job(*r))
open(".github/workflows/pew-part200-faddeev-temporel.yml", "w").write("".join(out)); print(len(RUNS), "jobs")
