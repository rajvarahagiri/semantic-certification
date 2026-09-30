# Phase 0 Runbook

This is the execution order. Do not skip ahead.

## A. Freeze environment

1. Keep repo private.
2. Fill in Wikimedia User-Agent contact.
3. Create venv and install dependencies.
4. Run tests: `pytest`.
5. Run `semcert pin-models` and commit the generated SHA values privately.
6. Record Python, PyTorch, Transformers, CUDA/MPS versions in your experiment notes.

## B. 500-article probe

1. `semcert collect-probe --limit 500`
2. Inspect `data/raw/failures.csv`; collection failures should be explained, not silently dropped.
3. `semcert run-probe`
4. Record exactly these first three decision numbers:
   - `anchor_positive_base_rate`
   - `lead_change_rate`
   - `label_flip_rate_given_lead_change`
5. Also record `noise_floor_disagreement_rate`.

## C. Probe decision

Proceed only if:
- base rate is between 20% and 70%;
- label-flip rate after lead changes is at least 2%;
- noise floor is not above epsilon.

If base rate/flip rate fail, tighten the task exactly once before Phase 0A, as allowed by the preregistration, then freeze the new prompt and prompt version.

## D. Never-recompute baseline

The initial probe report includes an event-weighted indicator. The next implementation milestone, only after the probe passes, is a proper replay that measures current-state staleness at sampled query times.

## E. Paper budget check

Use measured lead-changing revisions/day, a fixed population count, and conservative overlap assumptions. If the 5% budget cannot plausibly support useful event and state auditing, stop.

## F. Phase 0A then Phase 0B

Only implement these after B–E pass. Keep tuning/evaluation split intact: Aug 3–16 vs Aug 17–30, 2026.
