#!/usr/bin/env bash
# Build one detached mistsim worktree and locked venv per old
# environment in registry.ENVS. Idempotent: existing worktrees and
# venvs are kept. The dev4 environment is this repo's own .venv.
#
# Usage (from anywhere): bash setup_envs.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
ROOT="/home/christian/Documents/research/MIST/mistsim-versions"
mkdir -p "$ROOT"

# env key, mistsim commit, packages to install over the lock ("-" for
# none). Keep in sync with registry.ENVS.
# - cro500: ff99a9d's uv.lock is stale against its own pyproject
#   (jax<0.6.0, but the lock installs jax 0.9). Install the jax that
#   cro514 locks, so cro500 -> cro514 differs only in croissant and
#   mistsim.
# - cro521: mistsim never locked croissant 5.2.x; 5.1.4 and 5.2.1
#   declare the same dependencies.
ENVS=(
  "cro500 ff99a9d jax==0.5.3,jaxlib==0.5.3"
  "cro514 daa5e7b -"
  "cro521 daa5e7b croissant-sim==5.2.1"
  "dev2 ee3aca5 -"
  "dev3 03bde4c -"
)

for spec in "${ENVS[@]}"; do
  read -r key commit override <<<"$spec"
  dir="$ROOT/$key"
  if [ ! -d "$dir" ]; then
    git -C "$REPO" worktree add --detach "$dir" "$commit"
  fi
  # The lock decides croissant, s2fft and jax, as they were then.
  (cd "$dir" && uv sync --frozen --no-dev)
  if [ "$override" != "-" ]; then
    uv pip install --python "$dir/.venv/bin/python" --no-deps \
      ${override//,/ }
  fi
  "$dir/.venv/bin/python" -c \
    "import importlib.metadata as md; print('$key', \
md.version('croissant-sim'), md.version('jax'), md.version('s2fft'))"
done
