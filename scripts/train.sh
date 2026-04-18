#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# grpo-composer single-file launcher (local, single-node)
# ──────────────────────────────────────────────────────────────────
#
# Thin shell wrapper around scripts/train_local.py with sane defaults
# and full pass-through to its CLI. Use this when you want a one-liner
# without remembering every Python flag.
#
# Usage
# -----
#   bash scripts/train.sh                              # base GRPO on GSM8K, 1 GPU
#   bash scripts/train.sh -c examples/kalman_dapo.yaml -d
#   bash scripts/train.sh -c configs/papers/dapo.yaml -m Qwen/Qwen2.5-1.5B-Instruct -g 4
#   bash scripts/train.sh -c configs/papers/info_grpo.yaml -n my_run -e 3 -- ++data.train_batch_size=512
#
# Anything after `--` is forwarded verbatim as `--extra-overrides`
# (Hydra-style overrides), e.g. ++trainer.save_freq=10.
#
# Flags
# -----
#   -c, --config         composer YAML, relative to repo root
#                        (default: configs/base_grpo.yaml)
#   -m, --model          HF model id or local path
#                        (default: Qwen/Qwen2.5-0.5B-Instruct)
#   -p, --preset         dataset preset    (default: gsm8k)
#   -n, --name           run name          (default: derived from --config)
#   -e, --epochs         total epochs      (default: 1)
#   -g, --gpus           GPUs per node     (default: 1)
#   -d, --debug          turn on composer + DAPO debug
#   -h, --help           show this help
#
# Environment
# -----------
# All GRPO_COMPOSER_* env vars set in your shell are inherited.

set -euo pipefail

# Resolve repo root (works regardless of where the script is invoked from).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Defaults
CONFIG="configs/base_grpo.yaml"
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PRESET="gsm8k"
RUN_NAME=""
EPOCHS=1
GPUS=1
DEBUG=0
EXTRA=""

usage() {
    sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -euo/d'
    exit 0
}

# Parse short + long options. `--` ends option parsing; everything after it
# becomes the --extra-overrides string.
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)   CONFIG="$2";   shift 2 ;;
        -m|--model)    MODEL="$2";    shift 2 ;;
        -p|--preset)   PRESET="$2";   shift 2 ;;
        -n|--name)     RUN_NAME="$2"; shift 2 ;;
        -e|--epochs)   EPOCHS="$2";   shift 2 ;;
        -g|--gpus)     GPUS="$2";     shift 2 ;;
        -d|--debug)    DEBUG=1;       shift   ;;
        -h|--help)     usage          ;;
        --)            shift; EXTRA="$*"; break ;;
        *)
            echo "unknown option: $1" >&2
            echo "see: bash scripts/train.sh --help" >&2
            exit 2
            ;;
    esac
done

# Default run name = basename of the config (without .yaml) + _local
if [[ -z "${RUN_NAME}" ]]; then
    RUN_NAME="$(basename "${CONFIG}" .yaml)_local"
fi

CMD=(python "${REPO_ROOT}/scripts/train_local.py"
     --config "${CONFIG}"
     --model "${MODEL}"
     --dataset-preset "${PRESET}"
     --run-name "${RUN_NAME}"
     --total-epochs "${EPOCHS}"
     --n-gpus-per-node "${GPUS}")

if [[ -n "${EXTRA}" ]]; then
    CMD+=(--extra-overrides "${EXTRA}")
fi

if [[ "${DEBUG}" -eq 1 ]]; then
    CMD+=(--debug)
fi

echo "[train.sh] repo: ${REPO_ROOT}"
echo "[train.sh] cmd : ${CMD[*]}"
exec "${CMD[@]}"
#!/bin/bash
# LEGACY WRAPPER
#
# Official launchers:
#   1) scripts/train_modal.py  (canonical production launcher)
#   2) scripts/train_local.py  (official local fallback)
#
# This script is retained only for backward compatibility.

set -euo pipefail

CONFIG=${1:-configs/base_grpo.yaml}
MODEL=${2:-Qwen/Qwen2.5-0.5B-Instruct}
DATASET_PRESET=${3:-gsm8k}
NUM_GPUS=${4:-1}

if [[ "$DATASET_PRESET" == "openai/gsm8k" ]]; then
    DATASET_PRESET="gsm8k"
fi

echo "[legacy] scripts/train.sh is deprecated."
echo "[legacy] Redirecting to scripts/train_local.py."

exec python scripts/train_local.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    --dataset-preset "$DATASET_PRESET" \
    --n-gpus-per-node "$NUM_GPUS" \
    --run-name "legacy_$(basename "$CONFIG" .yaml)"
