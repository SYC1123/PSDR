#!/bin/bash

set -euo pipefail

OUTDIR="${OUTDIR:-./data_root/ISIC_Archive}"
mkdir -p "${OUTDIR}"

download_class() {
    local search_filter="$1"
    local limit="$2"
    local out_csv="$3"

    echo "Downloading ${out_csv} ..."
    isic image download --search "${search_filter}" --limit "${limit}" "${OUTDIR}"

    if [ -f "${OUTDIR}/metadata.csv" ]; then
        mv "${OUTDIR}/metadata.csv" "${OUTDIR}/${out_csv}"
        echo "Saved metadata: ${out_csv}"
    else
        echo "Warning: metadata.csv was not generated; skipping ${out_csv}"
    fi
}

download_class 'diagnosis_2:"Benign melanocytic proliferations"' 12875 NV.csv
download_class 'diagnosis_2:"Malignant melanocytic proliferations (Melanoma)"' 4522 MEL.csv
download_class 'diagnosis_3:"Basal cell carcinoma"' 3393 BCC.csv
download_class 'diagnosis_3:"Seborrheic keratosis"' 1464 SK.csv
download_class 'diagnosis_3:"Solar or actinic keratosis"' 869 AK.csv
download_class 'diagnosis_3:"Squamous cell carcinoma in situ"' 656 SCC.csv
download_class 'diagnosis_3:"Pigmented benign keratosis"' 384 BKL.csv
download_class 'diagnosis_3:"Solar lentigo"' 270 SL.csv
download_class 'diagnosis_2:"Benign soft tissue proliferations - Vascular"' 253 VASC.csv
download_class 'diagnosis_3:"Dermatofibroma"' 246 DF.csv
download_class 'diagnosis_4:"Actinic keratosis, Lichenoid"' 16 LK.csv
download_class 'diagnosis_3:"Lentigo simplex"' 27 LS.csv
download_class 'diagnosis_3:"Hemangioma"' 15 AN.csv

if [ -f "./merge.py" ]; then
    python3 ./merge.py
else
    echo "merge.py not found; skipping CSV merge."
fi

echo "Download complete."
