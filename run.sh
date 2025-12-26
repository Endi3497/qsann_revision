#!/usr/bin/env sh

# Lightweight launcher for QSANN_revision main.py
# Adjust variables below; all flags map to argparse options in main.py.

# Dataset settings
DATASET_CHOICE=${DATASET_CHOICE:-"mnist"}          # mnist|fmnist|cifar10|medmnist|pcam
DATASET_LABELS=${DATASET_LABELS:-"0 1 2 3 4 5 6 7 8 9"}            # space-separated labels (required for mnist/fmnist)
SAMPLES_PER_LABEL=${SAMPLES_PER_LABEL:-40}        # per-label cap; unset/empty to keep all
MEDMNIST_SUBSET=${MEDMNIST_SUBSET:-"organamnist"}    # pathmnist|dermamnist|retinamnist|bloodmnist|organamnist (only for medmnist)
PCAM_ROOT=${PCAM_ROOT:-"/home/junyeollee/QSANN/codes/QSANN_revision/data/camelyon/RGB"} # class0/class1 .npy patches for PCam/Camelyon16
MODEL_MODULE=${MODEL_MODULE:-"model"}                  # python module to import (e.g., model or model_revision)

# Data/patch geometry
IMAGE_SIZE=${IMAGE_SIZE:-32}
PATCH_SIZE=${PATCH_SIZE:-4}

# Split counts (required)
TRAIN_COUNT=${TRAIN_COUNT:-"320"}
VAL_COUNT=${VAL_COUNT:-"0"}
TEST_COUNT=${TEST_COUNT:-"80"}

# Quantum ansatz
NUM_QUBITS=${NUM_QUBITS:-8}
VQC_LAYERS=${VQC_LAYERS:-1}
REUPLOADING=${REUPLOADING:-2}
MEASUREMENT=${MEASUREMENT:-"z_zz_ring"}         # statevector|correlations|xyz|z_zz_ring
BACKEND_DEVICE=${BACKEND_DEVICE:-"cpu"}            # cpu|gpu
USE_TORCH_AUTOGRAD=${USE_TORCH_AUTOGRAD:-"--use-torch-autograd"}  # set to empty to disable
NUM_CLASSES=${NUM_CLASSES:-"10"}                     # set to override inferred class count (e.g., 5)

# Attention / aggregation / classifier
QKV_MODE=${QKV_MODE:-"separate"}                     # separate only
QKV_DIM=${QKV_DIM:-64}
ATTN_TYPE=${ATTN_TYPE:-"rbf"}                      # dot|rbf
ATTN_LAYERS=${ATTN_LAYERS:-1}
RBF_GAMMA=${RBF_GAMMA:-1.0}
AGG_MODE=${AGG_MODE:-"concat"}                  # concat|gap_gmp|attn_pool
HIDDEN_DIMS=${HIDDEN_DIMS:-"__none__"}                   # space-separated ints
DROPOUT=${DROPOUT:-0.3}

# Training
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.01}
NUM_WORKERS=${NUM_WORKERS:-2}
DEVICE=${DEVICE:-"cuda:0"}                         # cpu|cuda:0|cuda:1|...
EARLY_STOP=${EARLY_STOP:-""}                       # set to "--early-stop" to enable
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-10}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0.0}
NO_POS_WEIGHT=${NO_POS_WEIGHT:-""}                 # set to "--no-pos-weight" to disable pos_weight
NO_BALANCE_SAMPLER=${NO_BALANCE_SAMPLER:-"--no-balance-sampler"}       # set to "--no-balance-sampler" to disable sampler
SEED=${SEED:-42}
SAVE_STATEVECTOR=${SAVE_STATEVECTOR:-""}           # set to "--save-statevector" to enable
SAVE_STATEVECTOR_EPOCH=${SAVE_STATEVECTOR_EPOCH:-""}

# Logging / checkpoints
LOG_DIR=${LOG_DIR:-"results/logs"}
MODEL_DIR=${MODEL_DIR:-"results/models"}
RUN_NAME=${RUN_NAME:-"auto"}                       # "auto" embeds dataset/count/timestamp
MODEL_MODULE=${MODEL_MODULE:-"model"}                  # python module to import (e.g., model or model_revision)

PYTHON=${PYTHON:-python}

set -e
[ -n "$BASH_VERSION" ] && set -o pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

# Require split counts
if [ -z "$TRAIN_COUNT" ] || [ -z "$VAL_COUNT" ] || [ -z "$TEST_COUNT" ]; then
  echo "ERROR: Set TRAIN_COUNT, VAL_COUNT, and TEST_COUNT to split by counts." >&2
  exit 1
fi

# Handle hidden dims: "__none__" forces no hidden layers
HIDDEN_ARG=""
if [ "${HIDDEN_DIMS}" = "__none__" ]; then
  HIDDEN_ARG="--hidden-dims"
elif [ -n "${HIDDEN_DIMS}" ]; then
  HIDDEN_ARG="--hidden-dims ${HIDDEN_DIMS}"
fi

SAVE_STATEVECTOR_EPOCH_ARG=""
if [ -n "${SAVE_STATEVECTOR_EPOCH}" ]; then
  SAVE_STATEVECTOR_EPOCH_ARG="--save-statevector-epoch ${SAVE_STATEVECTOR_EPOCH}"
fi

mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
if [ "$RUN_NAME" = "auto" ]; then
  LOG_BASENAME="run-${DATASET_CHOICE}-${TS}"
else
  LOG_BASENAME="$RUN_NAME"
fi
LOG_FILE="${LOG_DIR}/${LOG_BASENAME}.log"

{
  echo "=== Experiment settings ($(date -Iseconds)) ==="
  echo "model_module=${MODEL_MODULE}"
  echo "dataset_choice=${DATASET_CHOICE}"
  echo "dataset_labels=${DATASET_LABELS}"
  echo "samples_per_label=${SAMPLES_PER_LABEL}"
  echo "medmnist_subset=${MEDMNIST_SUBSET}"
  echo "pcam_root=${PCAM_ROOT}"
  echo "image_size=${IMAGE_SIZE} patch_size=${PATCH_SIZE}"
  echo "train/val/test counts=${TRAIN_COUNT}/${VAL_COUNT}/${TEST_COUNT}"
  echo "num_qubits=${NUM_QUBITS} vqc_layers=${VQC_LAYERS} reuploading=${REUPLOADING} measurement=${MEASUREMENT}"
  echo "backend_device=${BACKEND_DEVICE} use_torch_autograd=${USE_TORCH_AUTOGRAD}"
  echo "qkv_mode=${QKV_MODE} qkv_dim=${QKV_DIM} attn_type=${ATTN_TYPE} attn_layers=${ATTN_LAYERS} rbf_gamma=${RBF_GAMMA}"
  echo "agg_mode=${AGG_MODE} hidden_dims=${HIDDEN_DIMS} dropout=${DROPOUT}"
  echo "epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LR} num_workers=${NUM_WORKERS} device=${DEVICE}"
  echo "early_stop=${EARLY_STOP} early_stop_patience=${EARLY_STOP_PATIENCE} early_stop_min_delta=${EARLY_STOP_MIN_DELTA}"
  echo "no_pos_weight_flag=${NO_POS_WEIGHT} balance_sampler_flag=${NO_BALANCE_SAMPLER}"
  echo "save_statevector=${SAVE_STATEVECTOR} save_statevector_epoch=${SAVE_STATEVECTOR_EPOCH}"
  echo "seed=${SEED} run_name_arg=${RUN_NAME}"
  echo

  $PYTHON main.py \
  --dataset-choice "$DATASET_CHOICE" \
  ${DATASET_LABELS:+--dataset-labels $DATASET_LABELS} \
  ${SAMPLES_PER_LABEL:+--samples-per-label "$SAMPLES_PER_LABEL"} \
  ${MEDMNIST_SUBSET:+--medmnist-subset "$MEDMNIST_SUBSET"} \
  --pcam-root "$PCAM_ROOT" \
  --model-module "$MODEL_MODULE" \
  --image-size "$IMAGE_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --train-count "$TRAIN_COUNT" \
  --val-count "$VAL_COUNT" \
  --test-count "$TEST_COUNT" \
  --num-qubits "$NUM_QUBITS" \
  --vqc-layers "$VQC_LAYERS" \
  --reuploading "$REUPLOADING" \
  --measurement "$MEASUREMENT" \
  --backend-device "$BACKEND_DEVICE" \
  $USE_TORCH_AUTOGRAD \
  ${NUM_CLASSES:+--num-classes "$NUM_CLASSES"} \
  --qkv-mode "$QKV_MODE" \
  --qkv-dim "$QKV_DIM" \
  --attn-type "$ATTN_TYPE" \
  --attn-layers "$ATTN_LAYERS" \
  --rbf-gamma "$RBF_GAMMA" \
  --agg-mode "$AGG_MODE" \
  ${HIDDEN_ARG} \
  --dropout "$DROPOUT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LR" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  $EARLY_STOP \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --early-stop-min-delta "$EARLY_STOP_MIN_DELTA" \
  $NO_POS_WEIGHT \
  $NO_BALANCE_SAMPLER \
  $SAVE_STATEVECTOR \
  $SAVE_STATEVECTOR_EPOCH_ARG \
  --seed "$SEED" \
  --log-dir "$LOG_DIR" \
  --model-dir "$MODEL_DIR" \
  --run-name "$RUN_NAME"
} 2>&1 | tee "$LOG_FILE"
