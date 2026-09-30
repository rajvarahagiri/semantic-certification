# Semantic Certification Controller — Phase 0

Private research implementation of the frozen **Phase 0 Preregistration — Semantic Certification Controller v1.0** dated September 28, 2026.

## What is implemented now

This repository implements the low-cost probe and Test-0 groundwork, not the later adaptive controller:

1. Select a configurable English-Wikipedia category hierarchy.
2. Fetch historical revisions plus a pre-window anchor revision.
3. Normalize **lead-only** model input, preserving `[REF]` and `[CITATION_NEEDED]` markers.
4. Mark lead changes, bot-like edits, and probable reverts.
5. Fetch daily per-article pageviews for real read-demand weights.
6. Pin exact Hugging Face model revisions before inference.
7. Label `lead_reference_need` with Qwen3-8B in non-thinking mode.
8. Score each changed lead against its **last full-inference anchor** with BGE-small-en-v1.5.
9. Run the 500-article probe: base rate, lead-change rate, flip rate, noise floor, and never-recompute staleness indicator.
10. Run a paper audit-budget calculation.

## Important: keep this repository private

The frozen preregistration says the controller design, code, and results stay non-public until patent counsel advises otherwise.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[models,dev]'
```

Edit `config/phase0.yaml` and replace the Wikimedia User-Agent contact placeholder with your own contact information.

### Pin model revisions

Do this **before** the real probe:

```bash
semcert pin-models
cat artifacts/model_lock.json
```

The probe refuses real inference without this lock file.

## 1. Collect the 500-article probe

```bash
semcert collect-probe --limit 500
```

Outputs:

- `data/raw/articles.csv`
- `data/raw/revisions.jsonl`
- `data/raw/pageviews.csv`
- `data/raw/failures.csv`
- `data/raw/collection_summary.json`

The recompute baseline is **lead-changing revisions only**, not all revisions.

## 2. Run the probe

```bash
semcert run-probe
```

Outputs:

- `reports/probe_summary.json`
- `reports/probe_revision_labels.csv`

The probe reports:

- anchor positive-label base rate;
- overall evaluated positive rate;
- percentage of collected revision transitions that change normalized lead input;
- label-flip rate conditional on a lead change;
- duplicate-run disagreement rate on up to 200 unchanged inputs;
- an event-weighted never-recompute staleness indicator;
- preregistered go/no-go checks.

The never-recompute number in this first implementation is an **event-weighted diagnostic**, not yet the final state-audit estimate of `E_G(t)`.

## 3. Run Test 0 — never recompute

After the probe passes:

```bash
semcert test0
```

This writes `reports/test0_summary.json` and `reports/test0_final_states.csv`. The final-state stale fraction is a diagnostic kill test; formal state-audit certification belongs to Phase 0A.

## 4. Paper budget check

Example:

```bash
semcert budget-check \
  --lead-changes-per-day 1200 \
  --populations 8 \
  --overlap-factor 2
```

The tool computes the exact zero-failure audit count needed to put a one-sided 95% binomial upper bound below epsilon. At epsilon=0.005 this is approximately 598 audits, matching the preregistration discussion.

## Mock smoke test

`--mock` exists only to validate plumbing without downloading models. Never use mock numbers as experimental results.

```bash
semcert run-probe --mock
```

## Reproducibility notes

- The normalized lead and pipeline version are persisted.
- The extraction hash includes the pipeline version.
- The full and gate model repository SHAs are persisted in `artifacts/model_lock.json`.
- Qwen3 is invoked with `enable_thinking=False` and deterministic `do_sample=False` generation for this experiment.
- Bot exclusion uses a conservative heuristic (username ending in `bot` or a bot-like revision tag); this should be audited before the final experiment.
- Revert detection combines MediaWiki revert/undo tags with repeated historical SHA-1 content.

## Phase 0B is intentionally not implemented yet

Per the frozen protocol, do **not** build the five-arm adaptive controller until:

1. the never-recompute baseline is nontrivial;
2. the paper budget check is economically plausible;
3. Phase 0A demonstrates read-relevant state-error heterogeneity.

That is the project's kill-test discipline.
