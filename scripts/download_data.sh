#!/usr/bin/env bash
# Downloads the IEEE-CIS Fraud Detection dataset into data/raw/.
#
# Order of attempts:
#   1. Official Kaggle competition "ieee-fraud-detection" (requires your Kaggle
#      account to have accepted that competition's rules).
#   2. Public Kaggle *dataset* mirror of the same files (used only if attempt 1
#      returns 401). Integrity is checked afterwards by src/data/validate.py,
#      which verifies the exact schema and row counts.
#
# Credentials come from ~/.kaggle/kaggle.json (standard mechanism) or
# KAGGLE_USERNAME / KAGGLE_KEY env vars. They are never hardcoded or printed.
#
# Usage: bash scripts/download_data.sh
set -euo pipefail

DATA_DIR="data/raw"
COMPETITION="ieee-fraud-detection"
MIRROR_DATASET="niangmohamed/ieeecis-fraud-detection"
KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"

mkdir -p "${DATA_DIR}"

if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    AUTH=("${KAGGLE_USERNAME}:${KAGGLE_KEY}")
elif [[ -f "${KAGGLE_JSON}" ]]; then
    # Parse username/key out of kaggle.json without echoing them.
    CREDS="$(python3 -c 'import json,os; d=json.load(open(os.path.expanduser("~/.kaggle/kaggle.json"))); print(d["username"], d["key"])')"
    AUTH=("${CREDS%% *}:${CREDS##* }")
else
    echo "ERROR: no Kaggle credentials found." >&2
    echo "Either export KAGGLE_USERNAME / KAGGLE_KEY, or place your token at ~/.kaggle/kaggle.json" >&2
    exit 1
fi

if [[ -f "${DATA_DIR}/train_transaction.csv" && -f "${DATA_DIR}/test_transaction.csv" ]]; then
    echo "Dataset already present in ${DATA_DIR}/ — nothing to do."
    exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

ZIP_PATH="${DATA_DIR}/download.zip"
SOURCE=""

echo "Attempt 1/2: official competition endpoint ..."
HTTP_CODE="$(curl -s -u "${AUTH[0]}" -o "${ZIP_PATH}" -w '%{http_code}' \
    -L "https://www.kaggle.com/api/v1/competitions/data/download-all/${COMPETITION}")"
if [[ "${HTTP_CODE}" == "200" ]]; then
    SOURCE="official competition ${COMPETITION}"
else
    echo "  -> HTTP ${HTTP_CODE}; falling back to public dataset mirror."
    HTTP_CODE="$(curl -s -u "${AUTH[0]}" -o "${ZIP_PATH}" -w '%{http_code}' \
        -L "https://www.kaggle.com/api/v1/datasets/download/${MIRROR_DATASET}")"
    if [[ "${HTTP_CODE}" != "200" ]]; then
        echo "ERROR: both download attempts failed (last HTTP ${HTTP_CODE})." >&2
        echo "Manual fallback: download the CSVs yourself from" >&2
        echo "https://www.kaggle.com/competitions/${COMPETITION}/data and place them in ${DATA_DIR}/" >&2
        exit 1
    fi
    SOURCE="dataset mirror ${MIRROR_DATASET}"
fi

echo "Downloaded from ${SOURCE}. Extracting ..."
unzip -q -o "${ZIP_PATH}" -d "${TMP_DIR}"
# Archives may contain inner zips (train.zip / test.zip).
find "${TMP_DIR}" -name '*.zip' -exec unzip -q -o {} -d "${TMP_DIR}" \;
mv "${TMP_DIR}"/*.csv "${DATA_DIR}/"
rm -f "${ZIP_PATH}"

echo "Done (source: ${SOURCE}). Files in ${DATA_DIR}:"
ls -lh "${DATA_DIR}"

echo
echo "Next: run 'python -m src.data.validate --data-dir ${DATA_DIR}' to verify the schema."
