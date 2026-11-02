# Sourced by the experiment scripts: paths, the device, and helpers.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC=$ROOT/experiments/src
export HETEROSV_DATA=${HETEROSV_DATA:-$ROOT/data}
export HETEROSV_RESULTS=${HETEROSV_RESULTS:-$ROOT/results}
export PYTHONPATH=$ROOT:$SRC:$ROOT/third_party/rhea${PYTHONPATH:+:$PYTHONPATH}
DATA=$HETEROSV_DATA
RESULTS=$HETEROSV_RESULTS
PAPER=$ROOT/paper_results
THREADS=${THREADS:-16}
if [ -z "${DEVICE:-}" ]; then
  DEVICE=$(python -c 'import torch; print("cuda:0" if torch.cuda.is_available() else "cpu")')
fi

step() { echo "[$(date '+%F %T')] $*"; }

# USE_PAPER_CALLS=1: take the calls of the paper's own run from paper_results/ instead of retraining.
retrain() { [ "${USE_PAPER_CALLS:-0}" != 1 ]; }

from_paper() {  # from_paper EXPERIMENT PATH...: copy results of the paper's run into results/
  local exp=$1; shift
  for f in "$@"; do
    mkdir -p "$(dirname "$RESULTS/$exp/$f")"
    cp -r "$PAPER/$exp/$f" "$RESULTS/$exp/$f"
  done
}

need() {  # need PATH MESSAGE
  [ -e "$1" ] || { echo "Missing $1: $2" >&2; exit 1; }
}

with_env() {  # with_env ENV COMMAND...: run a command with the tools of a conda environment of envs/
  local env=$1 prefix; shift
  prefix=$("${CONDA_EXE:-conda}" info --envs | awk -v e="$env" '$1 == e {print $NF}')
  [ -n "$prefix" ] || { echo "No conda environment $env: create it from envs/" >&2; exit 1; }
  PATH="$prefix/bin:$PATH" "$@"
}
