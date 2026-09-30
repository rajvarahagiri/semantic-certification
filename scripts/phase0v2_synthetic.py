from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# FROZEN PHASE 0 v2-A CONSTANTS
# ============================================================

N_CUSTOMERS = 100
TOP_N = 10
TOP_SHARE = 0.70
TOTAL_ORDERS = 100_000_000

HISTORY_DAYS = 14
EVAL_DAYS = 28
EVENTS_PER_DAY = 20
RECOMPUTES_PER_DAY = 1

N_SEEDS = 100

HISTORY_SIGMA = 0.7
QHAT_SIGMA = 0.35

SEEDS = list(range(100))

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)


# ============================================================
# TRUE IMPACT DISTRIBUTION
# ============================================================

def make_true_weights():
    """
    Top 10 customers = exactly 70% of impact.
    Within top 10: weight proportional to 1/rank.
    Remaining 90 split remaining 30% equally.
    """
    weights = np.zeros(N_CUSTOMERS, dtype=float)

    harmonic = np.array(
        [1.0 / r for r in range(1, TOP_N + 1)]
    )
    harmonic /= harmonic.sum()

    weights[:TOP_N] = TOP_SHARE * harmonic
    weights[TOP_N:] = (
        (1.0 - TOP_SHARE)
        / (N_CUSTOMERS - TOP_N)
    )

    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights[:TOP_N].sum(), TOP_SHARE)

    return weights


TRUE_W = make_true_weights()


# ============================================================
# HISTORICAL IMPACT LEARNING
# ============================================================

def learn_weights(rng):
    """
    Historical observed downstream impact with multiplicative
    lognormal noise centered at mean multiplier 1.
    """
    noise = rng.lognormal(
        mean=-(HISTORY_SIGMA ** 2) / 2,
        sigma=HISTORY_SIGMA,
        size=(HISTORY_DAYS, N_CUSTOMERS),
    )

    observed = TRUE_W[None, :] * noise

    learned = observed.mean(axis=0)
    learned /= learned.sum()

    return learned


# ============================================================
# EVENT STREAM
# ============================================================

def generate_events(rng):
    rows = []

    for day in range(EVAL_DAYS):
        for event_in_day in range(EVENTS_PER_DAY):

            customer = int(
                rng.integers(0, N_CUSTOMERS)
            )

            score = float(
                rng.beta(1.5, 4.0)
            )

            q_true = min(
                0.35,
                0.005 + 0.35 * (score ** 2),
            )

            q_noise = float(
                rng.lognormal(
                    mean=-(QHAT_SIGMA ** 2) / 2,
                    sigma=QHAT_SIGMA,
                )
            )

            q_hat = min(
                1.0,
                q_true * q_noise,
            )

            flip = bool(
                rng.random() < q_true
            )

            rows.append({
                "day": day,
                "event_in_day": event_in_day,
                "customer": customer,
                "score": score,
                "q_true": q_true,
                "q_hat": q_hat,
                "flip": flip,
            })

    return pd.DataFrame(rows)


# ============================================================
# POLICY DEFINITIONS
# ============================================================

POLICIES = [
    "random",
    "probability_only",
    "impact_only",
    "joint_known",
    "joint_learned",
    "recompute_all",
    "never_recompute",
]


def select_daily_recompute(
    policy,
    day_events,
    true_w,
    learned_w,
    random_rng,
):
    """
    Returns the DataFrame index of the one event to recompute.
    Special references may return ALL or NONE.
    """

    if policy == "recompute_all":
        return "ALL"

    if policy == "never_recompute":
        return "NONE"

    if policy == "random":
        choices = day_events.index.to_numpy()

        return int(
            random_rng.choice(choices)
        )

    if policy == "probability_only":
        priority = day_events["q_hat"].to_numpy()

    elif policy == "impact_only":
        priority = np.array([
            true_w[int(c)]
            for c in day_events["customer"]
        ])

    elif policy == "joint_known":
        priority = np.array([
            row.q_hat * true_w[int(row.customer)]
            for row in day_events.itertuples()
        ])

    elif policy == "joint_learned":
        priority = np.array([
            row.q_hat * learned_w[int(row.customer)]
            for row in day_events.itertuples()
        ])

    else:
        raise ValueError(policy)

    # Stable tie-break: earliest event in the day.
    best_pos = int(np.argmax(priority))

    return int(day_events.index[best_pos])


# ============================================================
# ONE POLICY REPLAY
# ============================================================

def run_policy(
    policy,
    events,
    true_w,
    learned_w,
    seed,
):
    stale = np.zeros(
        N_CUSTOMERS,
        dtype=bool,
    )

    weighted_trace = []
    unweighted_trace = []

    recomputes = 0

    # Strongest small-count / large-impact example.
    best_small_large = None

    random_rng = np.random.default_rng(
        1_000_000 + seed
    )

    checkpoint = 0

    for day in range(EVAL_DAYS):

        day_events = events[
            events["day"] == day
        ]

        selected = select_daily_recompute(
            policy,
            day_events,
            true_w,
            learned_w,
            random_rng,
        )

        for idx, row in day_events.iterrows():

            customer = int(row["customer"])

            if selected == "ALL":
                recompute = True

            elif selected == "NONE":
                recompute = False

            else:
                recompute = (
                    int(idx) == int(selected)
                )

            if recompute:
                # Full semantic inference makes current state correct.
                stale[customer] = False
                recomputes += 1

            else:
                # A semantic flip changes correctness relative to
                # the stored derived value.
                if bool(row["flip"]):
                    stale[customer] = ~stale[customer]

            weighted_stale = float(
                true_w[stale].sum()
            )

            unweighted_stale = float(
                stale.mean()
            )

            weighted_trace.append(
                weighted_stale
            )

            unweighted_trace.append(
                unweighted_stale
            )

            stale_count = int(stale.sum())

            if 0 < stale_count <= 10:

                stale_ids = np.where(stale)[0]

                top_stale = sorted(
                    stale_ids,
                    key=lambda c: true_w[c],
                    reverse=True,
                )[:5]

                candidate = {
                    "policy": policy,
                    "seed": seed,
                    "day": int(day),
                    "event_checkpoint": int(checkpoint),
                    "stale_customers": stale_count,
                    "customer_stale_pct": stale_count / N_CUSTOMERS,
                    "impact_stale_pct": weighted_stale,
                    "top_stale_customers": [
                        {
                            "customer_rank": int(c + 1),
                            "impact_share": float(true_w[c]),
                            "order_exposure": float(
                                true_w[c] * TOTAL_ORDERS
                            ),
                        }
                        for c in top_stale
                    ],
                }

                if (
                    best_small_large is None
                    or candidate["impact_stale_pct"]
                    > best_small_large["impact_stale_pct"]
                ):
                    best_small_large = candidate

            checkpoint += 1

    expected = {
        "recompute_all":
            EVAL_DAYS * EVENTS_PER_DAY,

        "never_recompute":
            0,
    }.get(
        policy,
        EVAL_DAYS * RECOMPUTES_PER_DAY,
    )

    assert recomputes == expected, (
        policy,
        recomputes,
        expected,
    )

    return {
        "policy": policy,
        "seed": seed,

        "mean_impact_stale":
            float(np.mean(weighted_trace)),

        "median_impact_stale":
            float(np.median(weighted_trace)),

        "max_impact_stale":
            float(np.max(weighted_trace)),

        "mean_unweighted_stale":
            float(np.mean(unweighted_trace)),

        "max_unweighted_stale":
            float(np.max(unweighted_trace)),

        "final_impact_stale":
            float(weighted_trace[-1]),

        "final_unweighted_stale":
            float(unweighted_trace[-1]),

        "recomputes": recomputes,

        "best_small_large":
            best_small_large,
    }


# ============================================================
# SUMMARY
# ============================================================

def summarize_policy(df):
    x = df["mean_impact_stale"]

    return {
        "mean": float(x.mean()),
        "median": float(x.median()),
        "q25": float(x.quantile(0.25)),
        "q75": float(x.quantile(0.75)),

        "mean_of_seed_max":
            float(df["max_impact_stale"].mean()),

        "median_seed_max":
            float(df["max_impact_stale"].median()),

        "overall_max":
            float(df["max_impact_stale"].max()),

        "mean_unweighted_stale":
            float(df["mean_unweighted_stale"].mean()),

        "median_unweighted_stale":
            float(df["mean_unweighted_stale"].median()),

        "recomputes_per_seed":
            int(df["recomputes"].iloc[0]),
    }


# ============================================================
# RUN
# ============================================================

print()
print("=== PHASE 0 v2-A SYNTHETIC CONSEQUENCE TEST ===")
print()
print("Customers:               ", N_CUSTOMERS)
print("Total order exposure:     ", f"{TOTAL_ORDERS:,}")
print("Top-10 impact share:      ", f"{100*TOP_SHARE:.1f}%")
print("Evaluation days:          ", EVAL_DAYS)
print("Changes/day:              ", EVENTS_PER_DAY)
print("Recomputes/day:           ", RECOMPUTES_PER_DAY)
print("Controller compute budget:", "5.0%")
print("Seeds:                    ", N_SEEDS)
print()

seed_rows = []
best_examples = {
    policy: None
    for policy in POLICIES
}

for seed in SEEDS:

    rng = np.random.default_rng(seed)

    learned_w = learn_weights(rng)

    events = generate_events(rng)

    for policy in POLICIES:

        result = run_policy(
            policy,
            events,
            TRUE_W,
            learned_w,
            seed,
        )

        example = result.pop(
            "best_small_large"
        )

        seed_rows.append(result)

        if example is not None:

            existing = best_examples[policy]

            if (
                existing is None
                or example["impact_stale_pct"]
                > existing["impact_stale_pct"]
            ):
                best_examples[policy] = example

    if (
        (seed + 1) % 10 == 0
        or seed == SEEDS[-1]
    ):
        print(
            f"Progress: {seed + 1:3d}/{N_SEEDS} seeds "
            f"({100*(seed+1)/N_SEEDS:5.1f}%)"
        )


results = pd.DataFrame(seed_rows)

results.to_csv(
    REPORTS / "phase0v2_synthetic_seed_results.csv",
    index=False,
)


summaries = {}

for policy in POLICIES:

    sub = results[
        results["policy"] == policy
    ]

    summaries[policy] = summarize_policy(sub)


# ============================================================
# FROZEN H1 GATE
# ============================================================

prob_med = summaries[
    "probability_only"
]["median"]

impact_med = summaries[
    "impact_only"
]["median"]

if prob_med <= impact_med:
    best_simple_name = "probability_only"
    best_simple = prob_med
else:
    best_simple_name = "impact_only"
    best_simple = impact_med


known = summaries[
    "joint_known"
]["median"]

learned = summaries[
    "joint_learned"
]["median"]


if best_simple > 0:
    known_improvement = (
        best_simple - known
    ) / best_simple
else:
    known_improvement = float("nan")


h1_pass = (
    math.isfinite(known_improvement)
    and known_improvement >= 0.15
)


# ============================================================
# FROZEN H2 GATE
# ============================================================

known_benefit = best_simple - known
learned_benefit = best_simple - learned

if known_benefit > 0:
    recovery_fraction = (
        learned_benefit / known_benefit
    )
else:
    recovery_fraction = float("nan")


h2_pass = (
    h1_pass
    and math.isfinite(recovery_fraction)
    and recovery_fraction >= 0.70
)


# ============================================================
# OUTPUT
# ============================================================

summary = {
    "preregistration": "Phase 0 v2-A",
    "seeds": N_SEEDS,
    "customers": N_CUSTOMERS,
    "total_orders": TOTAL_ORDERS,
    "top10_impact_share": TOP_SHARE,
    "changes_per_day": EVENTS_PER_DAY,
    "recomputes_per_day": RECOMPUTES_PER_DAY,
    "compute_fraction": (
        RECOMPUTES_PER_DAY
        / EVENTS_PER_DAY
    ),

    "policies": summaries,

    "best_simple": {
        "policy": best_simple_name,
        "median_impact_stale":
            best_simple,
    },

    "h1_v2": {
        "joint_known_median":
            known,

        "relative_improvement_vs_best_simple":
            known_improvement,

        "required_improvement":
            0.15,

        "pass":
            bool(h1_pass),
    },

    "h2_v2": {
        "joint_learned_median":
            learned,

        "known_impact_benefit":
            known_benefit,

        "learned_impact_benefit":
            learned_benefit,

        "recovery_fraction":
            recovery_fraction,

        "required_recovery":
            0.70,

        "pass":
            bool(h2_pass),
    },

    "small_change_large_impact_examples":
        best_examples,
}


with open(
    REPORTS / "phase0v2_synthetic_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


# Human-readable summary.

print()
print("=== PRIMARY RESULTS ===")
print()

table_rows = []

for policy in POLICIES:

    s = summaries[policy]

    table_rows.append({
        "policy": policy,
        "median impact stale %":
            100 * s["median"],

        "mean impact stale %":
            100 * s["mean"],

        "IQR low %":
            100 * s["q25"],

        "IQR high %":
            100 * s["q75"],

        "overall max %":
            100 * s["overall_max"],

        "mean count stale %":
            100 * s["mean_unweighted_stale"],

        "recomputes":
            s["recomputes_per_seed"],
    })


display = pd.DataFrame(table_rows)

print(
    display.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print()
print("=== H1-v2: CONSEQUENCE-AWARE JOINT CONTROL ===")
print()

print(
    "Best simple policy:",
    best_simple_name,
)

print(
    "Best simple median impact stale:",
    f"{100*best_simple:.3f}%"
)

print(
    "Joint-known median impact stale:",
    f"{100*known:.3f}%"
)

print(
    "Relative improvement:",
    f"{100*known_improvement:.2f}%"
)

print(
    "Required:",
    ">= 15.00%"
)

print(
    "H1-v2 PASS:",
    h1_pass,
)


print()
print("=== H2-v2: LEARNED IMPACT ===")
print()

print(
    "Joint-learned median impact stale:",
    f"{100*learned:.3f}%"
)

print(
    "Known-impact benefit:",
    f"{100*known_benefit:.3f} percentage points"
)

print(
    "Learned-impact benefit:",
    f"{100*learned_benefit:.3f} percentage points"
)

print(
    "Benefit recovery:",
    (
        f"{100*recovery_fraction:.2f}%"
        if math.isfinite(recovery_fraction)
        else "n/a"
    )
)

print(
    "Required:",
    ">= 70.00%"
)

print(
    "H2-v2 PASS:",
    h2_pass,
)


print()
print("=== SMALL CHANGE / LARGE IMPACT EXAMPLE ===")
print()

# Show strongest example from the never-recompute reference.
example = best_examples["never_recompute"]

if example:

    print("Policy: never_recompute")
    print("Seed:", example["seed"])
    print("Day:", example["day"] + 1)
    print(
        "Stale customers:",
        example["stale_customers"],
        f"({100*example['customer_stale_pct']:.1f}% of customers)"
    )
    print(
        "Stale order exposure:",
        f"{100*example['impact_stale_pct']:.1f}%"
    )

    print("Highest-impact stale customers:")

    for x in example["top_stale_customers"]:

        print(
            f"  rank {x['customer_rank']:2d} | "
            f"{100*x['impact_share']:6.2f}% of orders | "
            f"{x['order_exposure']:,.0f} orders"
        )


print()
print("Saved:")
print(
    "  reports/phase0v2_synthetic_seed_results.csv"
)
print(
    "  reports/phase0v2_synthetic_summary.json"
)
print()

if h1_pass:
    print(
        "V2-A CONSEQUENCE-AWARE GATE: PASS"
    )
else:
    print(
        "V2-A CONSEQUENCE-AWARE GATE: FAIL"
    )

if h1_pass and h2_pass:
    print(
        "V2-A LEARNED-IMPACT GATE: PASS"
    )
elif h1_pass:
    print(
        "V2-A LEARNED-IMPACT GATE: FAIL"
    )

