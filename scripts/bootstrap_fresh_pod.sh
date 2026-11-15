#!/usr/bin/env bash
# Reconstruct the reviewed rss2025 environment on fresh pod storage.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_fresh_pod.sh

Installs the pinned MuJoCo Playground rss2025 environment on a fresh GPU pod.
Requires x86_64 Linux, glibc >= 2.28,
an NVIDIA GPU with driver >= 580, and git, curl, python3, and findmnt.

Optional environment override:
  FORESIGHT_SETUP_ROOT  Existing setup directory; defaults to the parent
                        of this repository (normally /workspace).

The venv is always <root>/mujoco_playground/rss2025/.venv. Playground and
Menagerie revisions and the upstream lock are pinned in this script. Installs
Python 3.12 and all locked extras. If uv is absent, installs official uv 0.13.8
locally without editing shell profiles. No global Git settings are changed.

Interrupted installations owned by this script can be resumed by rerunning the
same command. A matching completed setup is verified and reused without sync.
An unknown existing venv, changed checkout, or different setup manifest stops
the script for review. No training is started.

Provenance and logs: <root>/.foresight-bootstrap/
EOF
}

if (( $# )); then
  if [[ $# == 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    usage
    exit 0
  fi
  printf 'Unsupported arguments. Use --help.\n' >&2
  exit 2
fi

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

for FORESIGHT_TOOL in git curl python3 findmnt; do
  type -P "$FORESIGHT_TOOL" >/dev/null || die "Required command is missing: $FORESIGHT_TOOL. Use a pod image containing it; no system packages were installed."
done
FORESIGHT_SYSTEM_PYTHON="$(type -P python3)"
FORESIGHT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
FORESIGHT_SETUP_ROOT="${FORESIGHT_SETUP_ROOT:-$(dirname -- "$FORESIGHT_REPO_ROOT")}"
[[ -d "$FORESIGHT_SETUP_ROOT" ]] || die "Setup root does not exist: $FORESIGHT_SETUP_ROOT. Create the setup directory first."
FORESIGHT_SETUP_ROOT="$(cd -- "$FORESIGHT_SETUP_ROOT" && pwd -P)"

# Empty RunPod containers may expose /workspace on overlay storage. Record that
# fact without blocking this explicitly requested fresh-pod installation.
FORESIGHT_MOUNT_JSON="$(findmnt --json --target "$FORESIGHT_SETUP_ROOT" --output TARGET,FSTYPE,MAJ:MIN)"
"$FORESIGHT_SYSTEM_PYTHON" - "$FORESIGHT_SETUP_ROOT" "$FORESIGHT_MOUNT_JSON" <<'PY_STORAGE'
import json
import sys
root, raw = sys.argv[1:]
mount = json.loads(raw)["filesystems"][0]
print(f"Storage: {root} -> {mount.get('target', '?')} ({mount.get('fstype', '?')})")
if mount.get("fstype", "") in {"overlay", "overlayfs", "fuse.overlayfs", "tmpfs", "ramfs"} or mount.get("target") == "/":
    print("This setup is on container storage. Keep uploaded results or download the review ZIP before replacing the pod.")
PY_STORAGE
[[ -w "$FORESIGHT_SETUP_ROOT" ]] || die "Setup root is not writable: $FORESIGHT_SETUP_ROOT"

"$FORESIGHT_SYSTEM_PYTHON" - <<'PY'
import os
import platform
import re

if platform.system() != "Linux" or platform.machine() != "x86_64":
    raise SystemExit("This pinned bootstrap requires x86_64 Linux.")
try:
    libc = os.confstr("CS_GNU_LIBC_VERSION") or ""
except (ValueError, OSError):
    libc = ""
match = re.fullmatch(r"glibc (\d+)\.(\d+)(?:\.\d+)?", libc)
if not match or tuple(map(int, match.groups())) < (2, 28):
    raise SystemExit(f"Requires glibc >= 2.28; detected {libc or 'unknown libc'}.")
print(f"Platform: {platform.machine()}, {libc}")
PY
type -P nvidia-smi >/dev/null || die 'nvidia-smi is missing. Start a GPU pod with the NVIDIA driver exposed; no installation was attempted.'
FORESIGHT_GPU_CSV="$(nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader,nounits)" \
  || die 'nvidia-smi failed. Check the GPU pod/driver before installing.'
"$FORESIGHT_SYSTEM_PYTHON" - "$FORESIGHT_GPU_CSV" <<'PY'
import csv
import io
import re
import sys

rows = list(csv.reader(io.StringIO(sys.argv[1]), skipinitialspace=True))
if not rows:
    raise SystemExit("No NVIDIA GPU was reported; no installation was attempted.")
for row in rows:
    if len(row) != 4 or not re.fullmatch(r"\d+(?:\.\d+)*", row[2].strip()):
        raise SystemExit("Could not parse NVIDIA GPU/driver information.")
    if int(row[2].split(".")[0]) < 580:
        raise SystemExit("Warp 1.18 requires NVIDIA driver >= 580. Select a compatible pod; no installation was attempted.")
    print(f"GPU {row[0]}: {row[1]}, driver {row[2]}, {row[3]} MiB")
PY

readonly FORESIGHT_PLAYGROUND_REV=9de993826d1a486f18aec4e8d3179e93ff64ed41
readonly FORESIGHT_LOCK_BLOB=01c0761262ca13a0ba0abc8b863f3e494189e05e
readonly FORESIGHT_PYPROJECT_BLOB=c33a9c6e7968bc585421764a36a3c2f3b74132de
readonly FORESIGHT_MENAGERIE_REV=1b86ece576591213e2b666ebf59508454200ca97
readonly FORESIGHT_UV_INSTALLER_VERSION=0.13.8
FORESIGHT_PLAYGROUND="$FORESIGHT_SETUP_ROOT/mujoco_playground"
FORESIGHT_RSS2025="$FORESIGHT_PLAYGROUND/rss2025"
FORESIGHT_VENV="$FORESIGHT_RSS2025/.venv"
FORESIGHT_MENAGERIE="$FORESIGHT_RSS2025/mujoco_playground/external_deps/mujoco_menagerie"
FORESIGHT_STATE_ROOT="$FORESIGHT_SETUP_ROOT/.foresight-bootstrap"
FORESIGHT_STATE="$FORESIGHT_STATE_ROOT/state.json"
FORESIGHT_ATTEMPT_DIR=''
FORESIGHT_STAGING=''

for FORESIGHT_PATH in "$FORESIGHT_STATE_ROOT" "$FORESIGHT_STATE" "$FORESIGHT_PLAYGROUND" "$FORESIGHT_VENV" "$FORESIGHT_MENAGERIE"; do
  [[ ! -L "$FORESIGHT_PATH" ]] || die "Refusing a symlink at a managed setup path: $FORESIGHT_PATH"
done

# The marker owns only this exact recipe/path. Never adopt an unrelated venv.
manifest() {
  "$FORESIGHT_SYSTEM_PYTHON" - "$1" "$FORESIGHT_STATE" "$FORESIGHT_SETUP_ROOT" \
    "$FORESIGHT_VENV" "$FORESIGHT_PLAYGROUND_REV" "$FORESIGHT_LOCK_BLOB" \
    "$FORESIGHT_PYPROJECT_BLOB" "$FORESIGHT_MENAGERIE_REV" "$FORESIGHT_ATTEMPT_DIR" <<'PY'
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

action, state_file, root, venv, playground, lock, pyproject, menagerie, attempt = sys.argv[1:]
path = Path(state_file)
recipe = {
    "setup_root": root, "venv": venv, "playground_revision": playground,
    "uv_lock_git_blob": lock, "pyproject_git_blob": pyproject,
    "menagerie_revision": menagerie, "python_request": "3.12",
    "managed_python_root": str(path.parent / "python"), "extras": "all",
}
if path.exists():
    try:
        state = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Unreadable bootstrap manifest: {path}: {exc}")
    if (state.get("kind") != "foresight-fresh-pod-bootstrap"
            or state.get("schema_version") != 1 or state.get("recipe") != recipe
            or state.get("status") not in {"in_progress", "complete"}):
        raise SystemExit(f"Bootstrap manifest differs from this pinned setup: {path}. Nothing will be adopted; review it first.")
else:
    state = {"kind": "foresight-fresh-pod-bootstrap", "schema_version": 1,
             "recipe": recipe, "status": "absent"}
if action == "inspect":
    print(state["status"])
    raise SystemExit(0)
now = datetime.now(timezone.utc).isoformat()
state.setdefault("created_at", now)
state["last_attempt_dir"] = attempt
state["status"] = action
if action == "complete":
    state.setdefault("completed_at", now)
    state["last_verified_at"] = now
else:
    state["last_started_at"] = now
with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".state-", delete=False) as stream:
    json.dump(state, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
    temporary = stream.name
os.replace(temporary, path)
PY
}
FORESIGHT_PREVIOUS_STATE="$(manifest inspect)"
if [[ -e "$FORESIGHT_VENV" && "$FORESIGHT_PREVIOUS_STATE" == absent ]]; then
  die "Existing venv has no matching bootstrap ownership manifest: $FORESIGHT_VENV. It was left untouched; review/recover that environment first."
fi
if [[ "$FORESIGHT_PREVIOUS_STATE" == complete && ! -x "$FORESIGHT_VENV/bin/python" ]]; then
  die "Completed setup has no working Python at $FORESIGHT_VENV/bin/python. Preserve $FORESIGHT_STATE and review the damaged setup before rebuilding."
fi

verify_checkout() {
  local directory=$1 revision=$2 expected_url=$3 origin top
  [[ -d "$directory" && ! -L "$directory" ]] || die "Expected a real checkout directory: $directory"
  top="$(git -C "$directory" rev-parse --show-toplevel)" || die "Not a Git checkout: $directory"
  [[ "$top" == "$directory" ]] || die "Unexpected repository root for $directory"
  [[ "$(git -C "$directory" rev-parse HEAD)" == "$revision" ]] || die "Checkout is not at the reviewed revision: $directory. Existing files were left untouched."
  origin="$(git -C "$directory" remote get-url origin)" || die "Checkout has no origin: $directory"
  [[ "$origin" == "$expected_url" || "$origin" == "${expected_url%.git}" ]] || die "Checkout origin differs from the expected public upstream: $directory"
  [[ -z "$(git -C "$directory" status --porcelain --untracked-files=all)" ]] || die "Checkout has local changes: $directory. Existing files were left untouched."
}
if [[ -e "$FORESIGHT_PLAYGROUND" ]]; then
  verify_checkout "$FORESIGHT_PLAYGROUND" "$FORESIGHT_PLAYGROUND_REV" https://github.com/google-deepmind/mujoco_playground.git
fi
if [[ -e "$FORESIGHT_MENAGERIE" ]]; then
  verify_checkout "$FORESIGHT_MENAGERIE" "$FORESIGHT_MENAGERIE_REV" https://github.com/deepmind/mujoco_menagerie.git
fi

mkdir -p "$FORESIGHT_STATE_ROOT/attempts"
FORESIGHT_ATTEMPT_DIR="$(mktemp -d "$FORESIGHT_STATE_ROOT/attempts/$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
printf '%s\n' "$FORESIGHT_MOUNT_JSON" > "$FORESIGHT_ATTEMPT_DIR/storage.json"
printf '%s\n' "$FORESIGHT_GPU_CSV" > "$FORESIGHT_ATTEMPT_DIR/gpus.csv"
nvidia-smi > "$FORESIGHT_ATTEMPT_DIR/nvidia-smi.txt"
exec 3>&1 4>&2
exec > >(tee -a "$FORESIGHT_ATTEMPT_DIR/bootstrap.log") 2>&1
FORESIGHT_TEE_PID=$!
finish() {
  local status=$? log_status
  trap - EXIT
  set +e
  if [[ -n "$FORESIGHT_STAGING" && "$FORESIGHT_STAGING" == "$FORESIGHT_STATE_ROOT"/clone.* ]]; then
    rm -rf -- "$FORESIGHT_STAGING"
  fi
  exec 1>&3 2>&4
  wait "$FORESIGHT_TEE_PID"
  log_status=$?
  (( status != 0 || log_status == 0 )) || status=$log_status
  printf '\nBootstrap exit status: %s\nProvenance: %s\n' "$status" "$FORESIGHT_ATTEMPT_DIR"
  if (( status != 0 )); then
    printf 'Keep this provenance. If installation was interrupted, rerun this same command to resume its owned setup.\n' >&2
  fi
  exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$FORESIGHT_PREVIOUS_STATE" != complete ]]; then
  manifest in_progress
fi

ensure_checkout() {
  local directory=$1 revision=$2 url=$3 sparse=${4:-}
  if [[ ! -e "$directory" ]]; then
    FORESIGHT_STAGING="$(mktemp -d "$FORESIGHT_STATE_ROOT/clone.XXXXXX")"
    GIT_TERMINAL_PROMPT=0 git clone --filter=blob:none --no-checkout "$url" "$FORESIGHT_STAGING/repo"
    if [[ -n "$sparse" ]]; then
      git -C "$FORESIGHT_STAGING/repo" sparse-checkout set "$sparse"
    fi
    GIT_TERMINAL_PROMPT=0 git -C "$FORESIGHT_STAGING/repo" checkout --detach "$revision"
    verify_checkout "$FORESIGHT_STAGING/repo" "$revision" "$url"
    mkdir -p "$(dirname -- "$directory")"
    mv -- "$FORESIGHT_STAGING/repo" "$directory"
    rmdir -- "$FORESIGHT_STAGING"
    FORESIGHT_STAGING=''
  fi
  verify_checkout "$directory" "$revision" "$url"
}
ensure_checkout "$FORESIGHT_PLAYGROUND" "$FORESIGHT_PLAYGROUND_REV" https://github.com/google-deepmind/mujoco_playground.git
[[ "$(git hash-object "$FORESIGHT_RSS2025/uv.lock")" == "$FORESIGHT_LOCK_BLOB" ]] || die 'Pinned upstream uv.lock does not match the reviewed blob.'
[[ "$(git hash-object "$FORESIGHT_RSS2025/pyproject.toml")" == "$FORESIGHT_PYPROJECT_BLOB" ]] || die 'Pinned upstream pyproject.toml does not match the reviewed blob.'
ensure_checkout "$FORESIGHT_MENAGERIE" "$FORESIGHT_MENAGERIE_REV" https://github.com/deepmind/mujoco_menagerie.git unitree_g1
[[ -d "$FORESIGHT_MENAGERIE/unitree_g1" ]] || die 'The pinned Menagerie checkout lacks unitree_g1 assets.'

# Use this recipe's local uv, never an unrelated or much older binary from PATH.
FORESIGHT_UV=''
if [[ -x "$FORESIGHT_STATE_ROOT/bin/uv" ]]; then
  FORESIGHT_UV="$FORESIGHT_STATE_ROOT/bin/uv"
fi
if [[ "$FORESIGHT_PREVIOUS_STATE" != complete && -z "$FORESIGHT_UV" ]]; then
  # Official documented installer options: https://docs.astral.sh/uv/reference/installer/
  FORESIGHT_INSTALLER="$FORESIGHT_ATTEMPT_DIR/uv-installer.sh"
  curl --connect-timeout 30 --max-time 180 --retry 3 --fail --show-error --silent --location --proto '=https' --tlsv1.2 \
    "https://astral.sh/uv/$FORESIGHT_UV_INSTALLER_VERSION/install.sh" -o "$FORESIGHT_INSTALLER"
  UV_UNMANAGED_INSTALL="$FORESIGHT_STATE_ROOT/bin" sh "$FORESIGHT_INSTALLER"
  FORESIGHT_UV="$FORESIGHT_STATE_ROOT/bin/uv"
  [[ -x "$FORESIGHT_UV" ]] || die 'The official uv installer did not produce the expected local binary.'
fi
if [[ -n "$FORESIGHT_UV" ]]; then
  "$FORESIGHT_UV" --version | tee "$FORESIGHT_ATTEMPT_DIR/uv-version.txt"
else
  printf 'uv absent; completed environment verification does not require reinstalling it.\n' > "$FORESIGHT_ATTEMPT_DIR/uv-version.txt"
fi

if [[ "$FORESIGHT_PREVIOUS_STATE" == complete ]]; then
  printf '\nReusing the completed pinned environment; no dependency sync will run.\n'
else
  if [[ "$FORESIGHT_PREVIOUS_STATE" == in_progress ]]; then
    printf '\nResuming the matching bootstrap-owned installation.\n'
  else
    printf '\nInstalling the pinned fresh environment.\n'
  fi
  # Keep the venv's interpreter/cache together under the setup root. Avoid an
  # inherited active environment or LD_LIBRARY_PATH changing the target/runtime.
  (
    cd "$FORESIGHT_RSS2025"
    unset VIRTUAL_ENV PYTHONHOME PYTHONPATH LD_LIBRARY_PATH MUJOCO_GL
    export UV_PROJECT_ENVIRONMENT="$FORESIGHT_VENV"
    export UV_PYTHON_INSTALL_DIR="$FORESIGHT_STATE_ROOT/python"
    export UV_MANAGED_PYTHON=1
    export UV_CACHE_DIR="$FORESIGHT_STATE_ROOT/cache"
    "$FORESIGHT_UV" --no-config sync --frozen --all-extras --python 3.12
  )
fi

printf '\nVerifying pinned package versions and JAX GPU execution.\n'
(
  unset LD_LIBRARY_PATH MUJOCO_GL PYTHONHOME PYTHONPATH
  export JAX_DEFAULT_MATMUL_PRECISION=highest
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  "$FORESIGHT_VENV/bin/python" - "$FORESIGHT_VENV" "$FORESIGHT_RSS2025" "$FORESIGHT_ATTEMPT_DIR" <<'PY'
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

venv, source, output = map(Path, sys.argv[1:])
if sys.version_info[:2] != (3, 12) or Path(sys.prefix).resolve() != venv.resolve():
    raise SystemExit("Python is not 3.12 from the expected venv.")
expected = {
    "jax": "0.12.0.1", "jaxlib": "0.12.0.1",
    "jax-cuda12-plugin": "0.12.0.1", "jax-cuda12-pjrt": "0.12.0.1",
    "brax": "0.14.2", "mujoco": "3.16.0", "mujoco-mjx": "3.16.0",
    "warp-lang": "1.18.0", "tensorboardx": "2.6.5", "matplotlib": "3.11.2",
}
versions = {name: metadata.version(name) for name in expected}
for name, wanted in expected.items():
    print(f"{name}=={versions[name]}", flush=True)
    if versions[name] != wanted:
        raise SystemExit(f"Pinned version mismatch: {name}; wanted {wanted}")
packages = sorted((dist.metadata["Name"], dist.version) for dist in metadata.distributions()
                  if dist.metadata.get("Name"))
(output / "installed-versions.txt").write_text("".join(f"{name}=={version}\n" for name, version in packages))
import jax
import jax.numpy as jnp
devices = jax.devices()
gpus = [device for device in devices if device.platform == "gpu"]
if not gpus:
    raise SystemExit("JAX has no GPU device. The owned setup is retained for diagnosis.")
result = jax.jit(lambda x: x + 1)(jax.device_put(jnp.ones(4), gpus[0]))
result.block_until_ready()
if result.devices() != {gpus[0]} or not bool(jnp.all(result == 2)):
    raise SystemExit("JAX GPU execution check failed.")
import mujoco_playground
if Path(mujoco_playground.__file__).resolve() != (source / "mujoco_playground/__init__.py").resolve():
    raise SystemExit("Imported Playground does not match the pinned source checkout.")
verification = {
    "python": sys.version, "python_executable": sys.executable,
    "platform": platform.platform(), "versions": versions,
    "jax_devices": [str(device) for device in devices],
    "jax_gpu_execution": "passed", "playground_file": mujoco_playground.__file__,
}
(output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
print("JAX devices:", devices, flush=True)
print("JAX GPU execution: passed", flush=True)
PY
)
verify_checkout "$FORESIGHT_PLAYGROUND" "$FORESIGHT_PLAYGROUND_REV" https://github.com/google-deepmind/mujoco_playground.git
verify_checkout "$FORESIGHT_MENAGERIE" "$FORESIGHT_MENAGERIE_REV" https://github.com/deepmind/mujoco_menagerie.git
manifest complete
cp -- "$FORESIGHT_STATE" "$FORESIGHT_ATTEMPT_DIR/setup-manifest.json"
if [[ "$FORESIGHT_PREVIOUS_STATE" == complete ]]; then
  printf '\nCompleted environment verified and reused; no packages were installed.\n'
elif [[ "$FORESIGHT_PREVIOUS_STATE" == in_progress ]]; then
  printf '\nInterrupted installation resumed; the pinned environment is now complete.\n'
else
  printf '\nFresh pinned environment installed and verified.\n'
fi
printf 'Activate it with:\n  source %q\nNo training was started.\n' "$FORESIGHT_VENV/bin/activate"
