"""part184zg : q0 de paire (t=0,2) pour les etats (omega, nu) du balayage part184zf (run 45068587944)."""
CH = [(0.14, 0.14), (0.14, 0.16), (0.14, 0.18), (0.19, 0.20), (0.19, 0.23)]
SES = 'https://claude.ai/code/session_012qTVdCRHJ2LxJqKJUux8qy'
o = ["name: PEW part184zg q0 de paire pour le balayage en nu", "", "on:", "  workflow_dispatch:", "", "permissions:", "  contents: write", "  actions: read", "", "jobs:"]
for om, nu in CH:
    tag = 'o%02d_n%02d' % (round(om * 100), round(nu * 100)); T = 'ZG_' + tag
    o += [f"  q_{tag}:", f"    name: q0 omega {om} nu {nu}", "    runs-on: ubuntu-24.04", "    timeout-minutes: 120", "    defaults:", "      run:", "        working-directory: numerical_minimization/protect3d_Z", "    steps:",
          "      - uses: actions/checkout@v4", "        with:", "          ref: main", "      - uses: actions/setup-python@v5", "        with:", "          python-version: '3.12'", "          cache: pip", "          cache-dependency-path: numerical_minimization/protect3d_Z/requirements.txt",
          "      - run: python -m pip install -r requirements.txt scipy", "      - uses: actions/download-artifact@v4", "        with:", f"          name: relaxg_{tag}_s4", "          path: numerical_minimization/protect3d_Z", "          run-id: '45068587944'", "          github-token: ${{ secrets.GITHUB_TOKEN }}",
          "      - name: Paire courte", "        env:"] + ["          " + e for e in ["N: '128'", "H: '0.5'", "NB: '128'", "SEPC: '20'", "REXT: '22'", "RA1: '12'", "RA2: '22'", "D2: '0.001'", "GUARD: '1e-6'", "DT: '0.02'", "ORIENT: z", "VPROD: '1'", "QPROF: '1'", "QCUR: '1'", "STATE: relaxedg_state.npz", "RELAXED: relaxedg.npy"]] + [
          f"        run: python jauge.py pair {nu} 0.02 {om} 0.2 {T} opp opp | tee {T}.log", "      - name: Archive dans resultats/part184", "        if: always()", "        working-directory: .", "        run: |", "          mkdir -p resultats/part184/gh_part184zg", f"          cp numerical_minimization/protect3d_Z/{T}.log resultats/part184/gh_part184zg/ || true",
          '          git config user.name "github-actions"', '          git config user.email "actions@users.noreply.github.com"', "          git add resultats/part184/gh_part184zg", f'          git commit -m "part184zg q0 de paire omega {om} nu {nu} : journal', "", "          Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>", f'          Claude-Session: {SES}" || exit 0',
          "          for i in 1 2 3 4 5; do git pull --rebase origin main && git push origin HEAD:main && break || sleep $((RANDOM % 20 + 5)); done"]
open('../../.github/workflows/pew-part184zg-q0-paire.yml', 'w').write('\n'.join(o) + '\n')
