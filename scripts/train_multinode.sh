#!/bin/bash
# LEGACY WRAPPER
#
# This script is no longer maintained as an official entrypoint.
# Official launchers:
#   1) scripts/train_modal.py  (canonical production launcher)
#   2) scripts/train_local.py  (official local fallback; single-node)
#
# For distributed production runs, use the Modal launcher path.

set -euo pipefail

echo "[legacy] scripts/train_multinode.sh is deprecated and intentionally disabled."
echo "[legacy] Use scripts/train_modal.py as the canonical distributed launcher."
echo "[legacy] Example:"
echo "  modal run scripts/train_modal.py --config configs/base_grpo.yaml --run-name my_run --n-gpus-per-node 8"
exit 2
