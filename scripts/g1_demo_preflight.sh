#!/usr/bin/env bash
# Read-only G1 preflight: never launches scientific physics or creates a pod.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "=== Git source ==="
git rev-parse HEAD
test -z "$(git status --porcelain --untracked-files=no)" || { echo 'ERROR: tracked working tree dirty'; exit 2; }
echo "=== Python geometry/depth/language tests ==="
python -m unittest discover -s tests -p 'test_obstacle*py' -v
python -m unittest discover -s tests -p 'test_language_nav.py' -v
python -m unittest discover -s tests -p 'test_task_stack.py' -v
python -m unittest discover -s tests -p 'test_g1_pod_ownership.py' -v
python -m unittest discover -s tests -p 'test_g1_github_auth.py' -v
python -m unittest discover -s tests -p 'test_g1_detached_run.py' -v
echo "=== Runtime versions / checkpoint ==="
python - <<'PY'
import hashlib, json, os, pathlib, sys
import jax, mujoco
print(json.dumps({'python':sys.version.split()[0], 'jax':jax.__version__,
                  'mujoco':mujoco.__version__,'devices':[str(x) for x in jax.devices()]},indent=2))
root=pathlib.Path(os.environ.get('G1_CHECKPOINT_DIR','checkpoints/nominal_s1/000202342400'))
if not root.is_dir():
    raise SystemExit('FAIL CLOSED: set G1_CHECKPOINT_DIR to pinned nominal s1 checkpoint')
tree=hashlib.sha256()
files=0
for p in sorted(root.rglob('*')):
    if p.is_file():
        tree.update(p.relative_to(root).as_posix().encode()+b'\0'
                    +hashlib.sha256(p.read_bytes()).hexdigest().encode()+b'\n')
        files+=1
expected='14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de'
got=tree.hexdigest()
print(json.dumps({'checkpoint':str(root),'files':files,'tree_sha256':got},indent=2))
if got != expected:
    raise SystemExit('FAIL CLOSED: unexpected checkpoint digest')
PY
echo "PREFLIGHT SUCCEEDED. Contact-enabled camera/obstacle integration, physics and video are NOT tested here."
