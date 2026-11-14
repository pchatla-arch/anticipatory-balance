#!/usr/bin/env bash
# Revive a RunPod pod (fresh or stopped) for Foresight in one command. Idempotent: every
# step is skipped when its result already exists, so it is safe to rerun.
#
# Fresh pod, from the pod's shell:
#   git clone https://github.com/pchatla-arch/anticipatory-balance.git /workspace/anticipatory-balance \
#     && BRANCH=session-d bash /workspace/anticipatory-balance/scripts/bootstrap_pod.sh
# Stopped-then-started pod (everything on disk): the same command, finishes in seconds.
#
# Afterwards:  source /workspace/mujoco_playground/rss2025/.venv/bin/activate
#              cd /workspace/anticipatory-balance && tmux new -s session
set -euo pipefail
WS=${WS:-/workspace}
BRANCH=${BRANCH:-main}
GIT_NAME=${GIT_NAME:-"Prasuna Chatla"}
GIT_EMAIL=${GIT_EMAIL:-"jiljilvadi@gmail.com"}
MENAGERIE_SHA=1b86ece576591213e2b666ebf59508454200ca97
say() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "No GPU on this pod; stop it and deploy a GPU pod."; exit 1; }

say "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

say "MuJoCo Playground (rss2025)"
PG="$WS/mujoco_playground"
[ -d "$PG/.git" ] || git clone -q https://github.com/google-deepmind/mujoco_playground.git "$PG"
cd "$PG/rss2025"
[ -x .venv/bin/python ] || uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
if ! python -c "import jax, mujoco_playground" 2>/dev/null; then
  uv --no-config sync --all-extras
fi
python -c "import tensorboardX, matplotlib" 2>/dev/null || uv pip install -q tensorboardX matplotlib

say "Menagerie (unitree_g1 at Playground's pinned commit)"
if [ ! -f mujoco_menagerie/unitree_g1/g1.xml ]; then
  rm -rf mujoco_menagerie
  git clone -q --filter=blob:none --no-checkout https://github.com/deepmind/mujoco_menagerie.git mujoco_menagerie
  (cd mujoco_menagerie && git sparse-checkout set unitree_g1 && git checkout -q "$MENAGERIE_SHA")
fi

say "Repository"
REPO="$WS/anticipatory-balance"
[ -d "$REPO/.git" ] || git clone -q https://github.com/pchatla-arch/anticipatory-balance.git "$REPO"
cd "$REPO"
git config user.name "$GIT_NAME"; git config user.email "$GIT_EMAIL"; git config credential.helper store
git fetch -q origin
git switch -q "$BRANCH" 2>/dev/null || git switch -q -c "$BRANCH" "origin/$BRANCH"
git pull -q --ff-only origin "$BRANCH" || echo "(branch not fast-forwardable; left as is)"
mkdir -p logs results

say "Check"
python -c "import jax; d=jax.devices(); print('jax devices', d); assert 'cuda' in str(d).lower(), 'no CUDA device'"
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux >/dev/null)
echo
echo "Ready on branch $BRANCH. Next:"
echo "  source $PG/rss2025/.venv/bin/activate && cd $REPO && tmux new -s session"
echo "  (first push will ask for your GitHub username and token once; the store keeps it)"
