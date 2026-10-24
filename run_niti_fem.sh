#!/usr/bin/env bash
# Launch a terminal, activate the FEniCSx conda environment and run niti_wire_fem.py.
#
# Usage:
#   ./run_niti_fem.sh                 # auto-detect the FEniCSx env and run the script
#   FENICS_ENV=myenv ./run_niti_fem.sh  # use a specific conda env name
#   ./run_niti_fem.sh --shell         # only open a shell inside the env, don't run
#   NP=4 ./run_niti_fem.sh            # run in parallel with 4 MPI processes
#
# After the run you are left in an interactive shell with the env still active,
# so you can re-run with:  python niti_wire_fem.py

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="niti_wire_fem.py"
SHELL_ONLY=0
[ "${1:-}" = "--shell" ] && SHELL_ONLY=1

# ── 1. If not attached to a terminal (e.g. double-clicked), reopen in one ─────
if [ ! -t 0 ] && [ -z "${NITI_IN_TERMINAL:-}" ]; then
    export NITI_IN_TERMINAL=1
    for term in gnome-terminal konsole xfce4-terminal x-terminal-emulator xterm; do
        if command -v "$term" >/dev/null 2>&1; then
            case "$term" in
                gnome-terminal) exec gnome-terminal -- bash "$0" "$@" ;;
                *)              exec "$term" -e bash "$0" "$@" ;;
            esac
        fi
    done
    echo "No terminal emulator found; continuing in the current session." >&2
fi

cd "$SCRIPT_DIR" || exit 1

# ── 2. Locate conda / mamba ───────────────────────────────────────────────────
find_conda_base() {
    if command -v conda >/dev/null 2>&1; then
        conda info --base 2>/dev/null && return 0
    fi
    for d in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" \
             "$HOME/anaconda3" "/opt/conda" "/opt/miniforge3" "/opt/miniconda3"; do
        [ -f "$d/etc/profile.d/conda.sh" ] && { echo "$d"; return 0; }
    done
    return 1
}

CONDA_BASE="$(find_conda_base)" || {
    echo "ERROR: conda was not found. Install Miniforge and create the env with:"
    echo "  conda create -n fenicsx -c conda-forge fenics-dolfinx mpich pyvista matplotlib"
    read -rp "Press Enter to close..." _
    exit 1
}
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ── 3. Pick the FEniCSx environment ───────────────────────────────────────────
env_has_dolfinx() {
    ls "$1"/lib/python3*/site-packages/dolfinx >/dev/null 2>&1
}

ENV_NAME="${FENICS_ENV:-}"
if [ -z "$ENV_NAME" ]; then
    # Common names first, then scan every env for an installed dolfinx package.
    for name in fenicsx fenicsx-env fenics fenics-env dolfinx; do
        if [ -d "$CONDA_BASE/envs/$name" ] && env_has_dolfinx "$CONDA_BASE/envs/$name"; then
            ENV_NAME="$name"; break
        fi
    done
fi
if [ -z "$ENV_NAME" ]; then
    for path in "$CONDA_BASE"/envs/*; do
        if [ -d "$path" ] && env_has_dolfinx "$path"; then
            ENV_NAME="$(basename "$path")"; break
        fi
    done
fi
if [ -z "$ENV_NAME" ]; then
    echo "ERROR: no conda environment containing dolfinx was found."
    echo "Available environments:"
    conda env list
    echo "Re-run with:  FENICS_ENV=<name> $0"
    read -rp "Press Enter to close..." _
    exit 1
fi

conda activate "$ENV_NAME" || { read -rp "Press Enter to close..." _; exit 1; }
echo "Activated FEniCSx environment: $ENV_NAME ($(python -c 'import dolfinx; print("dolfinx", dolfinx.__version__)' 2>/dev/null))"
echo "Working directory: $SCRIPT_DIR"

# ── 4. Run the simulation ─────────────────────────────────────────────────────
if [ "$SHELL_ONLY" -eq 0 ]; then
    echo "Running $PY_SCRIPT ..."
    echo "────────────────────────────────────────────────────────────"
    if [ -n "${NP:-}" ] && [ "${NP}" -gt 1 ]; then
        mpirun -n "$NP" python "$PY_SCRIPT"
    else
        python "$PY_SCRIPT"
    fi
    STATUS=$?
    echo "────────────────────────────────────────────────────────────"
    echo "Finished with exit code $STATUS"
fi

# ── 5. Stay in an interactive shell with the env still active ────────────────
RCFILE="$(mktemp)"
cat > "$RCFILE" <<EOF
[ -f ~/.bashrc ] && source ~/.bashrc
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$SCRIPT_DIR"
echo "You are in the '$ENV_NAME' environment. Re-run with: python $PY_SCRIPT   (type 'exit' to close)"
rm -f "$RCFILE"
EOF
exec bash --rcfile "$RCFILE" -i
