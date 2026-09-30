# Phase 0 v1 — Experimental Results

Status: COMPLETE / STOPPED PER FEASIBILITY CRITERIA

## Probe

- Evaluated articles: 431
- Lead-changing states: 105
- Positive-label base rate: 64.27%
- Label-flip rate conditional on lead change: 10.48%
- Model noise-floor disagreement: 0 / 200

The semantic task was sufficiently nontrivial to continue beyond the probe.

## Test 0 — Never Recompute

- Unweighted final current-state stale rate: 1.1601%
- Pageview-weighted final stale rate: 0.1773%
- Event-weighted stale rate: 8.5714%

Event-level error and current-state error were materially different.

## Gate / Oracle Diagnostics

Anchor-relative distance showed useful predictive signal.

A fixed threshold of 0.0005 on the full exploratory trace produced:

- 22.86% model-call savings
- 0 observed stale states on that trace

More permissive thresholds produced substantially greater savings but also stale states.

These were oracle/feasibility diagnostics, not statistical certification results.

## Time-Split Diagnostics

Tuning period:
- August 3–16, 2026

Evaluation period:
- August 17–30, 2026

A group-aware exploratory policy produced high model-call savings with low realized
pageview-weighted stale state, but further baseline testing showed that realized
pageview-weighted error was too easy an objective: even never-recompute remained
below the 0.5% weighted-error target on the evaluation trace.

Therefore realized weighted error alone does not demonstrate the proposed
certification controller.

## v0.1 label_flip issue

The stored `label_flip` field was found not to be reliably anchor-relative.

Four mismatches were identified.

All subsequent analysis recomputed anchor-relative flip truth as:

    fresh_label != anchor_label

The stored v0.1 `label_flip` field must not be used as ground truth.

## Certification Calibration Check

Tuning events: 54

Corrected anchor-relative flips: 5

Observed tuning flip rate: 9.259%

For a one-sided 95% confidence bound:

- Zero-miss audits required for upper bound <= 0.5%: 598
- Best possible bound with only 54 events even with zero misses: 5.397%

Low-distance region:

threshold < 0.0005
- audits: 14
- misses: 0
- 95% upper miss bound: 19.264%
- projected capacity for N=431 at epsilon=0.5%: 11 indebted objects
- projected capacity for N=13 at epsilon=0.5%: 0 indebted objects

Thus the current calibration dataset cannot support useful projected
certification at epsilon = 0.5%.

## Conclusion

Phase 0 v1 did not establish that the Semantic Certification Controller can
provide statistically certified semantic validity under the preregistered
0.5% target and 5% audit budget.

Useful findings remain:

- the semantic task is nontrivial;
- anchor-relative distance contains predictive signal;
- selective recomputation can substantially reduce inference;
- event error and current-state staleness are distinct quantities;
- workload/read-demand concentration materially affects observed state risk.

However:

- statistically useful event-level miss bounds were not established;
- H2 was not demonstrated;
- H3 was not demonstrated;
- certification feasibility under the 5% budget was not established.

No post-hoc change to the Phase 0 v1 target is made.

Any subsequent experiment should be preregistered separately as Phase 0 v2.
