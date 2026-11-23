"""part201 : effet du terme de Faddeev temporel (KF) sur la dynamique d'une paire de noeuds wk_T0 (L0, NU=0, sans jauge, CP=0, sans ponderation de milieu).
Memes departs (produit quaternionique, N=96, sep 24 mailles = 12 unites, orientation beta0), KF=0 contre KF=1. Eponge spherique (SPR0=18, SPRB=24), T=24, MEAS=0,5.
Lancer depuis la racine du depot : python numerical_minimization/protect3d_Z/gen_workflow_part201.py"""
# (tag, opp, KF, dt)
RUNS = [("opp_kf0", 1, 0, 0.0025), ("opp_kf1", 1, 1, 0.0025), ("opp_kf1_dt5", 1, 1, 0.005), ("same_kf0", 0, 0, 0.0025), ("same_kf1", 0, 1, 0.0025)]
COMMON = {"N": "96", "NB": "96", "H": "0.5", "E0": "0.25", "T0": "0", "MODE": "hot", "TC0": "1000000", "TH": "0", "TC": "0", "TSTART_H": "0", "G0": "0", "WARM": "0", "MU": "0.375",
          "MEAS": "0.5", "SNAP": "1000", "TIME_BUDGET": "18900", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "0", "KC": "0", "SEED": "1", "FINAL32": "1",
          "CP": "0", "MEANC": "0", "SPR0": "18", "SPRB": "24", "SPSIG": "0.3", "KW0": "1", "WA": "1", "WR": "1", "WV": "1", "KT": "1", "NU": "0", "T_END": "24", "INIT": "file", "INIT_FILE": "init96.npy"}
def job(tag, opp, kf, dt):
    full = f"p201_{tag}"
    env = dict(COMMON, KF=str(kf), DT=str(dt))
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    return f"""  {tag}:
    name: part201 {tag}
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
        run: python make_pair201.py init96.npy 24 {opp}
      - name: Run
        env:
{envs}
        run: python -u dyn186c.py {full} 2>&1 | tee run_{full}.log
      - name: Commit results to resultats/
        if: always()
        shell: bash
        run: |
          OUT=../../resultats/part201/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_proj.npy {full}_champ_final.npy $OUT/ || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part201/{full}; git commit -q -m 'part201 {tag} : resultats' || exit 0
          ok=0; for i in $(seq 1 20); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part201 Faddeev temporel - paires de noeuds, KF=0 contre KF=1\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for r in RUNS: out.append(job(*r))
open(".github/workflows/pew-part201-faddeev-temporel-paires.yml", "w").write("".join(out)); print(len(RUNS), "jobs")
