#!/usr/bin/env bash
set -euo pipefail

FSX_DNS="${FSX_DNS:-}"
FSX_MOUNT_NAME="${FSX_MOUNT_NAME:-}"
MOUNT_POINT="${MOUNT_POINT:-/mnt/fsx}"
LUSTRE_OPTS="${LUSTRE_OPTS:-noatime,flock}"

if [[ -z "$FSX_DNS" || -z "$FSX_MOUNT_NAME" ]]; then
  echo "Set FSX_DNS and FSX_MOUNT_NAME before running."
  exit 1
fi

sudo mkdir -p "$MOUNT_POINT"

if mount | grep -q "on ${MOUNT_POINT} "; then
  echo "Already mounted: $MOUNT_POINT"
  exit 0
fi

sudo mount -t lustre -o "$LUSTRE_OPTS" "${FSX_DNS}@tcp:/${FSX_MOUNT_NAME}" "$MOUNT_POINT"
echo "Mounted FSx at $MOUNT_POINT"
