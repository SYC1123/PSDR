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
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
PRETRAINED="${PRETRAINED:-True}"
RECAL_TAIL_FACTOR="${RECAL_TAIL_FACTOR:-1.0}"
PROTO_SEP_MARGIN="${PROTO_SEP_MARGIN:-0.5}"
NUM_PROTOTYPES_PER_CLASS="${NUM_PROTOTYPES_PER_CLASS:-1}"

TRAIN_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/training_if${FACTOR}.csv"
VAL_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/validation.csv"
TEST_CSV="${LT_SPLIT_ROOT}/shared_eval_seed${SEED}/testing.csv"

[[ -d "${DATA_ROOT}" ]] || { echo "Missing data dir: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${TRAIN_CSV}" ]] || { echo "Missing split file: ${TRAIN_CSV}" >&2; exit 1; }
[[ -f "${VAL_CSV}" ]] || { echo "Missing split file: ${VAL_CSV}" >&2; exit 1; }
[[ -f "${TEST_CSV}" ]] || { echo "Missing split file: ${TEST_CSV}" >&2; exit 1; }

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_NAME="run_tpcsd_isic2019lt_if${FACTOR}_${TIMESTAMP}"

PRETRAIN_FLAG="--pretrained"
if [[ "${PRETRAINED}" == "False" || "${PRETRAINED}" == "false" || "${PRETRAINED}" == "0" ]]; then
  PRETRAIN_FLAG="--no-pretrained"
fi

python train_stage1.py \
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
  --workers "${NUM_WORKERS}" \
  --image_size "${IMAGE_SIZE}" \
  --seed "${SEED}" \
  --backbone medclip_vit \
  "${PRETRAIN_FLAG}" \
  --use_projector \
  --proj_dim 128 \
  --num_prototypes_per_class "${NUM_PROTOTYPES_PER_CLASS}" \
  --cls_loss ce \
  --pcd_weight 1.0 \
  --pcd_temp 0.05 \
  --pcd_margin 0.85 \
  --spkd_weight 10.0 \
  --punif_weight 1.0 \
  --proto_sep_margin "${PROTO_SEP_MARGIN}" \
  --punif_warmup_start_epoch 40 \
  --punif_warmup_end_epoch 60 \
  --proto_momentum 0.96 \
  --recal_interval 5 \
  --recal_alpha 0.15 \
  --recal_tail_factor "${RECAL_TAIL_FACTOR}" \
  --queue_size 1024
