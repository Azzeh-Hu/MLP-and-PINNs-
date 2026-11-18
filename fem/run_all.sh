#!/usr/bin/env bash
# Run the full FEM reference: L0, L1, L2 for all cases, then the level comparison.
#   ./run_all.sh                  all levels + comparison
#   ./run_all.sh --verify         also run mms.py (code verification) and verify.py
#                                 (mesh / dt / tau_xi / symmetry checks)
#   ./run_all.sh --verify --sens  also run sensitivity.py on D0305_C3 (heating phase, L1)
# Full run with --verify takes roughly 2-3 h on 2 cores (L2 ~5 min per case).
#   ONLY=D0305_C3 ./run_all.sh    restrict to cases whose name contains ONLY
#   FENICS_ENV=myenv ./run_all.sh use a specific conda env (default: first env with dolfinx)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if ! python -c "import dolfinx" 2>/dev/null; then
    for d in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -f "$d/etc/profile.d/conda.sh" ] && { source "$d/etc/profile.d/conda.sh"; break; }
    done
    if [ -z "${FENICS_ENV:-}" ]; then
        FENICS_ENV=$(conda env list | awk '!/^#/ && NF {print $NF}' | while read -r p; do
            ls "$p"/lib/python3*/site-packages/dolfinx >/dev/null 2>&1 && { basename "$p"; break; }
        done)
    fi
    conda activate "${FENICS_ENV:?no conda env with dolfinx found; set FENICS_ENV}"
fi

LOG="run_all_$(date +%Y%m%d_%H%M).log"
{
    for script in L0_linear.py L1_apparent_heat_capacity.py L2_coupled_kinetics.py; do
        python -u "$script" --only "${ONLY:-}"
    done
    python -u compare_levels.py
    case " $* " in *" --verify "*) python -u mms.py && python -u verify.py ;; esac
    case " $* " in *" --sens "*) python -u sensitivity.py ;; esac
} 2>&1 | tee "$LOG"
echo "Results in ../results/fem/   log: fem/$LOG"
