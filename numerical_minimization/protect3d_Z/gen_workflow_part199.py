"""part199 : retard de phase d'une onde libre de la porteuse (paquet de phase k=3) traversant le creux d'un noeud, milieu fixe (WA=WR=WV=rv^-2, KT=rv^-4), eponge spherique, mu_a=0,05.
Pour rv in {1 ; 0,5} : K0 (noeud seul), Kw (noeud + onde), Vw (porteuse seule + onde) ; rv=0,5 : en plus Kb, Vb (eps 0,025, linearite). Depart du noeud : champ final de part194 / rv."""
RUNS = [("g0", 0.0), ("gp10", 0.01), ("gp05", 0.005), ("gm10", -0.01)]   # (tag, GRAD) ; T=32
COMMON = {"N": "96", "NB": "96", "H": "0.5", "E0": "0.25", "DT": "0.005", "T0": "0", "MODE": "hot", "TC0": "1000000", "TH": "0", "TC": "0", "TSTART_H": "0", "G0": "0", "WARM": "0", "MU": "0.375", "MEAS": "2.0",
          "SNAP": "1000", "TIME_BUDGET": "18900", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "0", "KC": "0", "SEED": "1", "FINAL32": "1", "CP": "1", "MEANC": "0",
          "SPR0": "16", "SPRB": "24", "SPSIG": "0.3", "KW0": "1", "WA": "1", "WR": "1", "WV": "1", "KT": "1", "T_END": "32"}
def job(tag, g):
    full = f"p199_{tag}"
    env = dict(COMMON, GRAD=str(g), INIT="file", INIT_FILE="init96.npy")
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    src = "../../resultats/part193/p193_s2_r100/p193_s2_r100_champ_final.npy"
    return f"""  {tag}:
    name: part199 {tag}
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
      - name: Depart noeud equilibre fois pente de porteuse
        run: python -c "import numpy as np; f=np.load('{src}').astype(np.float32); x=(np.arange(96)-48+.5)*.5; np.save('init96.npy', (f*(1+{g}*x)[None,:,None,None]).astype(np.float32))"
      - name: Run
        env:
{envs}
        run: python -u dyn186c.py {full} 2>&1 | tee run_{full}.log
      - name: Commit results to resultats/
        if: always()
        shell: bash
        run: |
          OUT=../../resultats/part199/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part199/{full}; git commit -q -m 'part199 {tag} : resultats' || exit 0
          ok=0; for i in $(seq 1 20); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part199 chute libre d'un noeud dans une pente de porteuse imposee\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for r in RUNS: out.append(job(*r))
open(".github/workflows/pew-part199-chute-gradient.yml", "w").write("".join(out)); print(len(RUNS), "jobs")
