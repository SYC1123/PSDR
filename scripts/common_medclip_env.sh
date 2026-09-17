#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HF_HOME="${HF_HOME:-${REPO_ROOT}/pretrained/archives/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

export BIO_CLINICAL_BERT_DIR="${BIO_CLINICAL_BERT_DIR:-${REPO_ROOT}/pretrained/Bio_ClinicalBERT_local}"
export MEDCLIP_TEXT_MODEL_DIR="${MEDCLIP_TEXT_MODEL_DIR:-${BIO_CLINICAL_BERT_DIR}}"
export MEDCLIP_WEIGHTS_PATH="${MEDCLIP_WEIGHTS_PATH:-${REPO_ROOT}/pretrained/medclip-vit/pytorch_model.bin}"
export MEDCLIP_VIT_WEIGHTS_PATH="${MEDCLIP_VIT_WEIGHTS_PATH:-${MEDCLIP_WEIGHTS_PATH}}"

if [[ -z "${MEDCLIP_VISION_MODEL_DIR:-}" && -f "${REPO_ROOT}/pretrained/medclip-vit/config.json" ]]; then
  export MEDCLIP_VISION_MODEL_DIR="${REPO_ROOT}/pretrained/medclip-vit"
fi

if [[ -z "${MEDCLIP_VIT_DIR:-}" && -n "${MEDCLIP_VISION_MODEL_DIR:-}" ]]; then
  export MEDCLIP_VIT_DIR="${MEDCLIP_VISION_MODEL_DIR}"
fi
