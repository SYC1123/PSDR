#!/bin/bash

set -euo pipefail

OUTDIR="${OUTDIR:-./data_root/ISIC2019LT}"
mkdir -p "${OUTDIR}"

wget "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
wget "https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"

unzip -jn ISIC_2019_Training_Input.zip -d "${OUTDIR}"
mv ISIC_2019_Training_GroundTruth.csv "${OUTDIR}"

rm ISIC_2019*.zip
