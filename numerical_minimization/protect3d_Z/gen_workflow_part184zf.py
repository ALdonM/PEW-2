"""part184zf : balayage de nu a omega fixe pour trouver, a charge egale, des etats (omega, nu) comparables (test n1 revu).
Relaxation avec jauge N=128, 4 etapes de 25 unites, depuis relaxed128n (noeud sans jauge, run 44638873798)."""
import sys
CH = [(0.14, 0.14), (0.14, 0.16), (0.14, 0.18), (0.19, 0.20), (0.19, 0.23), (0.19, 0.26)]
SES = 'https://claude.ai/code/session_012qTVdCRHJ2LxJqKJUux8qy'
out = ["name: PEW part184zf balayage de nu a omega fixe relaxation avec jauge N128", "", "on:", "  workflow_dispatch:", "", "permissions:", "  contents: write", "  actions: read", "", "jobs:"]
for om, nu in CH:
    tag = 'o%02d_n%02d' % (round(om * 100), round(nu * 100))
    for s in range(1, 5):
        j = 'r_%s_s%d' % (tag, s)
        first = s == 1
        out += [f"  {j}:", f"    name: relaxation omega {om} nu {nu} etape {s}/4", "    runs-on: ubuntu-24.04", "    timeout-minutes: 240"]
        if not first: out += [f"    needs: r_{tag}_s{s-1}"]
        out += ["    defaults:", "      run:", "        working-directory: numerical_minimization/protect3d_Z", "    steps:",
                "      - uses: actions/checkout@v4", "        with:", "          ref: main",
                "      - uses: actions/setup-python@v5", "        with:", "          python-version: '3.12'", "          cache: pip", "          cache-dependency-path: numerical_minimization/protect3d_Z/requirements.txt",
                "      - run: python -m pip install -r requirements.txt scipy",
                "      - uses: actions/download-artifact@v4", "        with:", "          name: " + ("relaxed128n" if first else f"relaxg_{tag}_s{s-1}"), "          path: numerical_minimization/protect3d_Z"]
        if first: out += ["          run-id: '44638873798'", "          github-token: ${{ secrets.GITHUB_TOKEN }}"]
        env = ["N: '128'", "H: '0.5'", "RELT: '25'", f"OMREL: '{om}'", "RG2: '0.02'"] + (["RELIN: relaxed.npy"] if first else ["RELIN: relaxedg.npy", "RELST: relaxedg_state.npz"])
        out += ["      - name: relaxation avec jauge", "        env:"] + ["          " + e for e in env]
        out += [f"        run: python jauge.py relax {nu} relaxedg.npy | tee relax_{tag}_s{s}.log",
                "      - uses: actions/upload-artifact@v4", "        with:", f"          name: relaxg_{tag}_s{s}", "          path: |", "            numerical_minimization/protect3d_Z/relaxedg.npy", "            numerical_minimization/protect3d_Z/relaxedg_state.npz", "          if-no-files-found: error",
                "      - name: Archive dans resultats/part184", "        if: always()", "        working-directory: .", "        run: |",
                "          mkdir -p resultats/part184/gh_part184zf", f"          cp numerical_minimization/protect3d_Z/relax_{tag}_s{s}.log resultats/part184/gh_part184zf/ || true",
                '          git config user.name "github-actions"', '          git config user.email "actions@users.noreply.github.com"', "          git add resultats/part184/gh_part184zf",
                f'          git commit -m "part184zf relaxation omega {om} nu {nu} etape {s}/4 : journal', "", "          Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>", f'          Claude-Session: {SES}" || exit 0',
                "          for i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 20 + 5)); done"]
open('../../.github/workflows/pew-part184zf-balayage-nu.yml', 'w').write('\n'.join(out) + '\n')
