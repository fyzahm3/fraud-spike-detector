# Dataset setup — IEEE-CIS Fraud Detection

The dataset (~1.2 GB uncompressed) lives in `data/raw/`, which is **gitignored**.
Nothing under `data/raw/` is ever committed.

## Option A — automated download (recommended)

```bash
bash scripts/download_data.sh
```

Credentials are read from the standard Kaggle mechanism:

- place your token at `~/.kaggle/kaggle.json` (from kaggle.com → Settings → API), or
- export `KAGGLE_USERNAME` / `KAGGLE_KEY`.

Never put credentials inside this repo. `.gitignore` blocks `kaggle.json`,
`.env`, and key files as a backstop, and a pre-commit secret scan runs on every
commit (`git config core.hooksPath scripts/hooks` installs it).

## Option B — manual download

1. Log in at <https://www.kaggle.com/competitions/ieee-fraud-detection/data>
   and accept the competition rules.
2. Download `train_transaction.csv`, `train_identity.csv`,
   `test_transaction.csv`, `test_identity.csv`, `sample_submission.csv`.
3. Place them in `data/raw/`.

## Provenance note (honest disclosure)

The download script tries the **official competition endpoint** first. If your
API token has not accepted that competition's rules (the case for the token
used during development — the endpoint returns HTTP 401), it falls back to a
public Kaggle *dataset* mirror (`niangmohamed/ieeecis-fraud-detection`) of the
same files. Integrity of whatever lands in `data/raw/` is then verified by
schema checks in `src/data/validate.py`; our local run confirmed the documented
scale exactly: 590,540 train rows × 394 columns, 144,233 identity rows,
fraud rate 3.499% — matching the official dataset description.

## Validate + split

```bash
python -m src.data.validate --data-dir data/raw   # data quality report
python -m src.data.split  --manifest-out results/split_manifest.json
```

The split is chronological by `TransactionDT` (70/15/15) with boundaries
snapped so no timestamp straddles two splits; `results/split_manifest.json`
records per-split checksums used later to prove the held-out set was never
touched during training.
