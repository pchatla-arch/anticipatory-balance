#!/usr/bin/env bash
# Cosmetic only: register names for already-existing Linux group IDs on empty pods.
set -euo pipefail
command -v getent >/dev/null || exit 0
if [[ "$(id -u)" != 0 || ! -w /etc/group ]]; then
  echo "Skipping group-name fix (not privileged)."
  exit 0
fi
for gid in $(id -G); do
  [[ $gid =~ ^[0-9]+$ ]] || continue
  if ! getent group "$gid" >/dev/null; then
    name="runpod_extra_gid_${gid}"
    if ! getent group "$name" >/dev/null; then
      printf '%s:x:%s:\n' "$name" "$gid" >> /etc/group
      printf 'Registered missing local group name for GID %s\n' "$gid"
    fi
  fi
done
