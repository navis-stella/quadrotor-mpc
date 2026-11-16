#!/usr/bin/env bash
# run.sh — run Python inside the WSL acados venv from Windows (Git Bash / Claude Code).
#
# The acados C libraries and the acados_template package live in WSL (Ubuntu),
# not on Windows. The real work is done by ~/acados_run.sh *inside* WSL, which
# exports the acados env vars, activates ~/acados_env, and cd's to the project.
# This wrapper just forwards your arguments to it.
#
# Usage (from Git Bash on Windows):
#   ./run.sh stage2_terminal/simulate_dare.py      same as: ./run.sh -m stage2_terminal.simulate_dare
#   ./run.sh -m pytest -q
#
# Scripts live in packages (core/, stageN_*/) and import each other by package
# name, so they must run as modules from the repo root. A first argument of the
# form <dir>/<file>.py is rewritten to -m <dir>.<file> for that reason.
#
# For python -c one-liners with quotes, run inside a WSL shell instead:
#   wsl
#   ~/acados_run.sh -c "import acados_template; print('ok')"
#
# Plot windows (see core/plot_utils.py): by default all figures open together
# at the end of the run. Save-only, no windows:
#   QUAD_PLOTS=none ./run.sh stage5_tracking/simulate_tracking.py
set -euo pipefail
export MSYS_NO_PATHCONV=1
export WSLENV="${WSLENV:+$WSLENV:}QUAD_PLOTS/u"   # forward QUAD_PLOTS into WSL
if [[ $# -gt 0 && "$1" == */*.py && "$1" != /* ]]; then   # relative paths only
    mod="${1%.py}"; mod="${mod#./}"; shift
    set -- -m "${mod//\//.}" "$@"
fi
# $HOME is expanded inside WSL, so the launcher path holds no user name
exec wsl.exe -e bash -c 'exec "$HOME/acados_run.sh" "$@"' run.sh "$@"
