#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/common_medclip_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"

DATA_ROOT="${DATA_ROOT:-./data_root/ISIC2019LT/ISIC_2019_Training_Input}"
LT_SPLIT_ROOT="${LT_SPLIT_ROOT:-./split/ISIC2019LT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
LOG_ROOT="${LOG_ROOT:-./log/tpcsd}"

FACTOR="${FACTOR:-100}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-2048}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-100}"
TRAIN_NOISE_STD="${TRAIN_NOISE_STD:-0.01}"
PROJECTOR_LR="${PROJECTOR_LR:-1e-4}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.05}"
VIRTUAL_LOSS_WEIGHT="${VIRTUAL_LOSS_WEIGHT:-1.0}"
HARDEST_K="${HARDEST_K:-3}"
HARDEST_FRACTION="${HARDEST_FRACTION:-0.5}"
VIRTUAL_CONF_THRESH="${VIRTUAL_CONF_THRESH:-0.6}"
VIRTUAL_CENTER_COS_THRESH="${VIRTUAL_CENTER_COS_THRESH:-0.2}"
STAGE1_RUN_DIR="${STAGE1_RUN_DIR:-}"
CKPT_TAG="${CKPT_TAG:-best}"

TRAIN_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/training_if${FACTOR}.csv"
VAL_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/validation.csv"
TEST_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/testing.csv"

if [[ -z "${STAGE1_RUN_DIR}" ]]; then
  STAGE1_RUN_DIR="$(ls -td "${CHECKPOINT_ROOT}"/run_tpcsd_isic2019lt_if${FACTOR}_* 2>/dev/null | head -n 1 || true)"
fi

[[ -d "${DATA_ROOT}" ]] || { echo "Missing data dir: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${TRAIN_CSV}" ]] || { echo "Missing split file: ${TRAIN_CSV}" >&2; exit 1; }
[[ -f "${VAL_CSV}" ]] || { echo "Missing split file: ${VAL_CSV}" >&2; exit 1; }
[[ -f "${TEST_CSV}" ]] || { echo "Missing split file: ${TEST_CSV}" >&2; exit 1; }

[[ -n "${STAGE1_RUN_DIR}" ]] || { echo "Could not resolve latest Stage1 run dir." >&2; exit 1; }

ENCODER_CKPT="${STAGE1_RUN_DIR}/resnet_encoder_${CKPT_TAG}.pth"
PROTOTYPE_CKPT="${STAGE1_RUN_DIR}/prototype_memory_${CKPT_TAG}.pth"
PROJECTOR_CKPT="${STAGE1_RUN_DIR}/projector_${CKPT_TAG}.pth"

[[ -f "${ENCODER_CKPT}" ]] || { echo "Missing encoder_ckpt: ${ENCODER_CKPT}" >&2; exit 1; }
[[ -f "${PROJECTOR_CKPT}" ]] || { echo "Missing projector_ckpt: ${PROJECTOR_CKPT}" >&2; exit 1; }
[[ -f "${PROTOTYPE_CKPT}" ]] || { echo "Missing prototype_ckpt: ${PROTOTYPE_CKPT}" >&2; exit 1; }

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_NAME="run_tpcsd_stage2_isic2019lt_if${FACTOR}_${TIMESTAMP}"

python train_stage2.py \
  --dataset ISIC2019LT \
  --data_path "${DATA_ROOT}" \
  --csv_file_train "${TRAIN_CSV}" \
  --csv_file_val "${VAL_CSV}" \
  --csv_file_test "${TEST_CSV}" \
  --run_name "${RUN_NAME}" \
  --checkpoints "${CHECKPOINT_ROOT}" \
  --log_dir "${LOG_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --stage2_batch_size "${STAGE2_BATCH_SIZE}" \
  --workers "${NUM_WORKERS}" \
  --image_size 224 \
  --seed "${SEED}" \
  --backbone medclip_vit \
  --use_projector \
  --proj_dim 128 \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --projector_ckpt "${PROJECTOR_CKPT}" \
  --prototype_ckpt "${PROTOTYPE_CKPT}" \
  --merge_real \
  --virtual_ratio 1.0 \
  --aas_alpha 2.0 \
  --lambda_mu 0.5 \
  --cov_scale_factor 1.0 \
  --cosine_scale 16.0 \
  --lr 1e-3 \
  --projector_lr "${PROJECTOR_LR}" \
  --delta_noise 0.01 \
  --train_noise_std "${TRAIN_NOISE_STD}" \
  --hardest_k "${HARDEST_K}" \
  --hardest_fraction "${HARDEST_FRACTION}" \
  --virtual_conf_thresh "${VIRTUAL_CONF_THRESH}" \
  --virtual_center_cos_thresh "${VIRTUAL_CENTER_COS_THRESH}" \
  --anchor_weight "${ANCHOR_WEIGHT}" \
  --virtual_loss_weight "${VIRTUAL_LOSS_WEIGHT}"
