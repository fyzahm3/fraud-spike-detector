# fraud-spike-detector

Fraud-spike detection for card payments: scores individual transactions,
aggregates risk over rolling time windows per entity (card / merchant /
device cluster) to detect *spikes*, and writes an auditable, human-readable
risk brief to a review queue. **Defense-only** — it flags and explains;
a human always decides.

> Work in progress, built phase by phase (see git history).
> Data setup instructions: [docs/DATA_SETUP.md](docs/DATA_SETUP.md)
> Full README with measured results arrives with the final checkpoint.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # fill in values when later phases need them
bash scripts/download_data.sh   # ~1.2 GB into gitignored data/raw/
pytest                          # full suite incl. leakage-safety tests
```
