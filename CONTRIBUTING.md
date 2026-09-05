# Contributing

Thanks for looking. This project has a small number of properties it will not trade away, and
knowing them up front makes a contribution much likelier to land.

## The invariants

These are enforced by tests, not by review. Breaking one fails CI.

1. **No fabricated results.** Every number displayed anywhere — site, README, brief — must trace to a
   committed file under `results/` or be computed by a model at request time. A figure typed into a
   template or into prose is a bug, and there are tests that scan for exactly that.
2. **Leakage discipline is structural.** No function in `src/models/train.py` accepts a test frame.
   Thresholds and hyperparameters come from validation only. Evaluation refuses to run if a split's
   checksum differs from `results/split_manifest.json`.
3. **Causality of streaming features.** Every graph or spike feature for row *i* uses strictly-past
   rows only. Windows are half-open `(t-W, t]`, ties break on `(TransactionDT, TransactionID)`, and
   cold-start entities default to 0.
4. **Defense-only.** No blocking, fund-holding, or payment-execution code, anywhere. `ReviewQueue`'s
   public API is exactly `{enqueue, list_pending, resolve, get_audit_log}` — extend a signature, never
   add a fifth method.
5. **The LLM explains, it does not decide.** Scores, confidence, costs and the recommended-action enum
   are computed in Python before any model call.
6. **Determinism.** Same seed ⇒ byte-identical models and scores.
7. **The dashboard escapes structurally.** Queue data reaches the page through `textContent`. Nothing
   is assigned to `innerHTML`.

If a change requires breaking one of these, that is a conversation to have in an issue first — not a
test to relax.

## Getting set up

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # xgboost needs `brew install libomp` on macOS
pytest -q                            # 141 tests, ~30s, no dataset required
```

Tests that need the ~650MB IEEE-CIS dataset skip themselves when `data/raw/` is absent, so a clean
checkout is fully testable. To work on the model itself, see [`docs/DATA_SETUP.md`](docs/DATA_SETUP.md).

Enable the secret-scan pre-commit hook once after cloning:

```bash
git config core.hooksPath scripts/hooks
```

## Making a change

- Run `pytest -q` before opening a PR. CI runs the same suite.
- If results change, regenerate the committed artifacts (`evaluate.py`, `scripts/compare_phases.py`)
  and update the README table — those numbers are stated as measured.
- Match the surrounding code. Comments here explain *why* a thing is the way it is, especially where
  the obvious approach was rejected; please keep that habit.
- Keep the web dependency set (`requirements-web.txt`) honest — it exists because the hosted instance
  has 512MB.

## Reporting something

Open an issue with what you expected, what happened, and the command you ran. For anything
security-related, see [SECURITY.md](SECURITY.md) instead.
