# Phase 0 v2 Preregistration
## Consequence-Aware Semantic Change Control

Status: FROZEN BEFORE RESULTS  
Date: 2026-09-30

## 1. Relationship to Phase 0 v1

Phase 0 v1 remains unchanged.

Phase 0 v1 found that:
- semantic changes were nontrivial;
- anchor-relative distance contained useful signal;
- selective recomputation could reduce full-model calls;
- event error and current-state staleness were different quantities;
- the available calibration sample was insufficient to support useful
  statistical certification at epsilon = 0.5% under the original budget.

Phase 0 v2 tests a new hypothesis:

Record-count magnitude is not equivalent to consequence magnitude.

A small number of changed entities may represent a disproportionately large
share of downstream business activity.

Example:

100 customers may collectively represent 100 million orders.

If 10 customers leave, customer-count change is only 10%.

But if those customers represent 70 million orders, downstream business
exposure is 70%.

The controller should therefore reason about the consequence of a change,
not merely the number of changed records.

No Phase 0 v1 success criterion is retroactively modified.

---

## 2. Research Question

Can a semantic change-control system reduce downstream impact from stale
model-derived state by prioritizing recomputation using both:

1. estimated probability that semantic state became stale; and
2. estimated downstream consequence if that state is stale?

---

## 3. Core Principle

For object i at time t:

    q_i(t) = estimated probability that stored semantic state is stale

    w_i(t) = downstream consequence / impact weight

Define consequence-adjusted semantic risk:

    R_i(t) = q_i(t) * w_i(t)

For population G:

    ProjectedImpactDebt_G(t)
        = sum_i(q_i(t) * w_i(t))
          / sum_i(w_i(t))

For realized evaluation:

    RealizedImpactStale_G(t)
        = sum_i(1[stale_i(t)] * w_i(t))
          / sum_i(w_i(t))

Unweighted stale fraction is also reported:

    CountStale_G(t)
        = sum_i(1[stale_i(t)])
          / |G|

The experiment explicitly tests whether CountStale can appear small while
RealizedImpactStale is materially larger.

---

## 4. Sources of Impact Weight

Impact weight may come from one or more of:

### Direct observable impact
- order volume;
- GMV;
- revenue;
- transaction count;
- affected users;
- downstream records.

### Dependency-derived impact
- number of downstream objects;
- dependency fan-out;
- workflow criticality;
- graph centrality;
- number of dependent systems.

### Learned impact
Historical observations of:

    upstream change
        ->
    downstream outcome magnitude

may be used to estimate future impact.

Phase 0 v2-A tests both known and learned impact weights.

---

## 5. Phase 0 v2-A — Synthetic Consequence Test

Purpose:

Determine whether consequence-aware prioritization has algorithmic value
before spending money or model inference on a real dataset.

This is NOT a statistical-certification experiment.

No LLM is used.

---

## 6. Synthetic Population

Number of customers:

    100

Total downstream order exposure:

    100,000,000 orders

Impact concentration:

    Top 10 customers together represent exactly 70% of orders.

The top 10 are distributed using rank weight:

    weight_r proportional to 1/r
    for ranks r = 1 ... 10

and normalized so their combined share equals 70%.

The remaining 90 customers share the remaining 30% equally.

Thus:

    10% of entities -> 70% of downstream impact

This structure is fixed before evaluation.

---

## 7. Historical Period for Learned Impact

Before the evaluation period, generate:

    14 historical daily outcome observations per customer.

Observed daily impact is generated from the true impact weight with
multiplicative log-normal variation.

Noise parameter:

    sigma = 0.7

Learned impact weight:

    mean historical observed impact for the customer

followed by normalization across customers.

The learned controller does not receive the true impact weight.

---

## 8. Evaluation Event Stream

Evaluation duration:

    28 days

Change events per day:

    20

Total events per simulation:

    560

For each change:

1. select a customer uniformly from the 100 customers;
2. generate a gate/change score:

       score ~ Beta(1.5, 4.0)

3. define latent semantic-flip probability:

       q_true = min(
           0.35,
           0.005 + 0.35 * score^2
       )

4. generate a noisy probability estimate q_hat from q_true using
   multiplicative log-normal noise with sigma = 0.35;

5. generate the actual semantic flip using Bernoulli(q_true).

All controllers within the same random seed receive the identical:
- customer sequence;
- scores;
- probability estimates;
- semantic-flip outcomes.

---

## 9. Compute Constraint

The experiment models scarce full semantic-model capacity.

Each day:

    20 changed objects arrive.

The controller may perform:

    exactly 1 full recomputation per day.

Therefore:

    recompute budget = 5% of changed events.

This 5% recompute budget is specific to Phase 0 v2-A.

It is NOT the Phase 0 v1 state/event audit budget.

The controller selects which one of the twenty daily changes receives the
full recomputation.

---

## 10. Controllers / Baselines

All policies receive identical event streams.

### Arm A — Random

Choose one changed object randomly.

### Arm B — Probability Only

Priority:

    q_hat

Choose the event with highest q_hat.

### Arm C — Impact Only

Priority:

    w_i

Choose the event with highest impact weight.

### Arm D — Joint Known Impact

Priority:

    q_hat * w_i

Uses the true synthetic downstream impact weight.

This is the oracle consequence-aware controller.

### Arm E — Joint Learned Impact

Priority:

    q_hat * w_hat_i

where w_hat_i is learned from the 14-day historical outcome period.

### Reference — Recompute All

Recompute every change.

### Reference — Never Recompute

Recompute no changes.

---

## 11. State Transition

Every object begins semantically current.

On a recomputed event:

    stale_i = False

On a skipped event where no semantic flip occurs:

    stale_i remains unchanged

On a skipped event where a semantic flip occurs:

    stale_i toggles

This models a binary derived semantic state.

---

## 12. Evaluation Repetitions

Run:

    100 fixed random seeds

Seeds:

    0 through 99

Each policy within one seed receives the identical underlying event stream.

This creates paired comparisons between policies.

No seed may be dropped after results are observed.

---

## 13. Primary Measurement

Primary metric:

    mean realized impact-weighted stale exposure
    during the 28-day evaluation period

For each policy report:

- median across 100 seeds;
- mean across 100 seeds;
- interquartile range;
- maximum impact-weighted stale exposure;
- unweighted stale fraction;
- exact recomputation count;
- fraction of downstream order exposure stale.

---

## 14. Primary Hypothesis — Consequence Awareness

H1-v2:

At the same recomputation cost, jointly using semantic-change probability
and downstream impact reduces impact-weighted stale exposure relative to
using either signal independently.

Define:

    BestSimple
        = better of Probability Only and Impact Only

Primary Gate:

The median impact-weighted stale exposure of Joint Known Impact must be at
least 15% lower than BestSimple.

If this does not occur, stop pursuing the consequence-aware joint-controller
branch.

---

## 15. Learned-Impact Hypothesis

H2-v2:

Historical downstream outcomes can recover enough impact information to
retain most of the benefit of an oracle impact-aware controller.

Let:

    B = median exposure of BestSimple
    K = median exposure of Joint Known Impact
    L = median exposure of Joint Learned Impact

Known-impact benefit:

    B - K

Learned-impact benefit:

    B - L

Recovery fraction:

    (B - L) / (B - K)

Primary learned-impact gate:

    recovery fraction >= 70%

If Joint Known Impact passes but Joint Learned Impact fails this gate,
the impact-aware idea may remain useful, but the claim that impact can be
learned from the specified history is not supported by this experiment.

---

## 16. Small-Change / Large-Impact Diagnostic

Report examples where:

    number of stale customers is small

but:

    percentage of stale downstream order exposure is large.

Specifically report:

- number of stale customers;
- percentage of customers stale;
- percentage of downstream order exposure stale;
- identities/ranks of the highest-impact stale customers.

This diagnostic is descriptive and is not itself a pass criterion.

---

## 17. What Phase 0 v2-A Does NOT Establish

Passing v2-A does not establish:

- statistical certification;
- a 0.5% validity guarantee;
- patent novelty;
- real-world prevalence of the synthetic distribution;
- usefulness of any particular LLM;
- audit-budget feasibility.

It establishes only whether consequence-aware control has algorithmic value
under a deliberately consequence-skewed workload.

---

## 18. Phase 0 v2-B Gate

Phase 0 v2-B will be designed only if v2-A passes.

Before new LLM inference, v2-B must perform a paper calculation covering:

- calibration sample size;
- impact-weighted debt bounds;
- event-audit requirements;
- state-audit requirements;
- full-model-equivalent cost;
- expected impact concentration;
- overlap benefits;
- certification lifetime.

If optimistic arithmetic shows that impact-weighted certification still
cannot fit inside the available compute/audit budget, Phase 0 v2 stops
without a new expensive model run.

---

## 19. Real Dataset Gate

A real transactional dataset is considered only after:

1. v2-A passes;
2. v2-B paper economics pass.

Preferred structure:

    entity/customer
        ->
    transactions/orders
        ->
    monetary or operational impact

The dataset must allow downstream consequence to be measured independently
of the semantic model.

---

## 20. Kill Discipline

Do not:

- change the 70% impact concentration after results;
- change the 5% recomputation budget after results;
- change the 100 seeds after results;
- change the 15% primary improvement threshold after results;
- redefine BestSimple after results;
- remove unfavorable seeds;
- change the learned-impact 70% recovery gate after results.

A failed result remains a failed result.

A new hypothesis requires a separately versioned preregistration.

---

## 21. Confidentiality

Until patent counsel advises otherwise:

- repository remains private;
- experiment design remains non-public;
- results remain non-public;
- code/results are not publicly demonstrated or posted;
- dated commits and experiment outputs are retained.

