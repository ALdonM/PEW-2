"""Génère .github/workflows/pew-part188c-kp3b.yml (racine du dépôt). part188c : paquet d'ondes (phase sans masse / amplitude massive) traversant un nœud relaxé ; témoins vide ; 64^3, h=0,5, t=22."""
KF = "../../resultats/part179/wk_T0/wk_T0_champ_final.npy"   # nœud relaxé de part179 (deg -1), au centre de la boîte
runs0 = [("Kp3", dict(INIT="file", INIT_FILE=KF, WAVE="phase", WEPS="0.05"), 1), ("Vp3", dict(INIT="vac", WAVE="phase", WEPS="0.05"), 1),
        ("Kp3b", dict(INIT="file", INIT_FILE=KF, WAVE="phase", WEPS="0.025"), 1), ("Vp3b", dict(INIT="vac", WAVE="phase", WEPS="0.025"), 1),
        ("Ka3", dict(INIT="file", INIT_FILE=KF, WAVE="amp", WEPS="0.05"), 1), ("Va3", dict(INIT="vac", WAVE="amp", WEPS="0.05"), 1)]   
runs=[r for r in runs0 if r[0]=="Kp3b"]  # K0 : réutiliser resultats/part188/p188_K0 (même code, même durée)
ENV = {"MODE": "hot", "TC0": "1000000", "TH": "0", "TC": "0", "TSTART_H": "0", "G0": "0", "WARM": "0", "N": "64", "H": "0.5", "MU": "0.375", "E0": "0.25", "DT": "0.0025", "T0": "0",
       "T_END": "22", "MEAS": "2.0", "SNAP": "1000", "TIME_BUDGET": "18900", "JAX_PLATFORM_NAME": "cpu", "OMP_NUM_THREADS": "4", "BARRIER": "0", "NOEUDS": "0", "KC": "0", "SEED": "1",
       "WK": "3.0", "WX0": "-10", "WSIG": "4", "FINAL32": "1"}
def job(tag, extra, c, NCH):
    full = f"p188b_{tag}"; env = dict(ENV, **extra)
    envs = "\n".join(f"          {k}: '{v}'" for k, v in env.items())
    needs = f"    needs: {tag}_r{c-1}\n    if: always() && needs.{tag}_r{c-1}.outputs.finished == 'false'\n" if c > 1 else ""
    guard = f"""      - name: Garde-fou - le checkpoint doit exister
        run: test -f ck_{full}.npz || (echo "CHECKPOINT ABSENT"; exit 1)
""" if c > 1 else ""
    return f"""  {tag}_r{c}:
    name: part188b {tag} chunk (relance) {c}/{NCH}
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
          key: p188b-{tag}-r{c}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}
          restore-keys: |
            p188b-{tag}-
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
        shell: bash
        run: |
          OUT=../../resultats/part188b/{full}; mkdir -p $OUT
          cp {full}_history.jsonl {full}_params.json {full}_state.json {full}_champ_final.npy $OUT/ 2>/dev/null || true
          cd ../..; git config user.name 'github-actions[bot]'; git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
          git add resultats/part188b/{full}; git commit -q -m 'part188b {tag} chunk (relance) {c} : resultats' || exit 0
          ok=0; for i in $(seq 1 12); do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && ok=1 && break; git rebase --abort 2>/dev/null; sleep $((RANDOM % 30 + 5)); done
          test $ok = 1
"""
out = ["name: PEW part188c part188c iceberg dans la houle (paquet de phase / amplitude traversant un noeud)\n\non:\n  workflow_dispatch:\n\npermissions:\n  contents: write\n  actions: read\n\njobs:\n"]
for r in runs:
    for c in range(1, r[2] + 1): out.append(job(r[0], r[1], c, r[2]))
open(".github/workflows/pew-part188c-kp3b.yml", "w").write("".join(out)); print(len(runs), "runs")
