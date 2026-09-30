# Phase 0 v2-A Results
## Consequence-Aware Semantic Change Control

Status: PRIMARY GATE FAILED
Date: 2026-09-30

## Primary Result

At identical 5% recomputation cost:

- Random: 7.280% median impact-weighted stale exposure
- Probability only: 6.318%
- Impact only: 4.188%
- Joint known impact: 3.783%
- Joint learned impact: 3.855%

Best simple baseline:

    Impact Only = 4.188%

Joint Known Impact:

    3.783%

Relative improvement:

    9.67%

Preregistered requirement:

    >= 15%

H1-v2 therefore FAILED.

Per the preregistration, the consequence-aware joint-controller branch is
not advanced on the basis of this experiment.

## Learned Impact Secondary Result

Known-impact benefit versus best simple:

    0.405 percentage points

Learned-impact benefit:

    0.333 percentage points

Recovery fraction:

    82.10%

The predefined learned-impact recovery threshold was 70%.

However, H2-v2 was contingent on the primary joint-control hypothesis.
Because H1-v2 failed, this secondary result does not convert Phase 0 v2-A
into a passing experiment.

It does indicate that historical downstream outcomes recovered much of the
impact information available from the oracle weights in this synthetic setup.

## Small Change / Large Impact Finding

The experiment strongly demonstrated that entity-count magnitude can differ
materially from downstream consequence magnitude.

Example:

- stale customers: 10 / 100 = 10%
- stale downstream order exposure: 38.5%

The two highest-impact stale customers alone represented approximately:

- rank 1: 23.90% of orders
- rank 2: 11.95% of orders

Thus a relatively small number of changed/stale entities can represent a
disproportionately large downstream business exposure.

## Interpretation

The experiment supports consequence-aware prioritization as an important
systems principle.

It does not establish that combining semantic-change probability with impact
provides enough additional value over impact-only prioritization to support
the preregistered joint-controller thesis.

Impact Only materially outperformed Probability Only, suggesting that
downstream consequence may be a more important allocation signal in this
synthetic workload than semantic-change probability.

## Next Research Question

Any further work should be separately preregistered.

A possible new research direction is:

Can a data/AI system learn dynamic downstream consequence weights from
observed dependencies and outcomes, and use those learned weights to
prioritize recomputation, validation, auditing, or other scarce processing?

This is a new hypothesis and must not be treated as a successful result of
Phase 0 v2-A.

