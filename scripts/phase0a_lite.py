from pathlib import Path
import json
import math
import pandas as pd

EPSILON = 0.005
MIN_DEMAND_SHARE = 0.15
HETEROGENEITY_MULTIPLE = 2.0
Z95 = 1.959963984540054

ROOT = Path(".")
REPORTS = ROOT / "reports"
RAW = ROOT / "data" / "raw"


def as_bool(s):
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def wilson_interval(k, n, z=Z95):
    if n == 0:
        return float("nan"), float("nan")

    p = k / n
    z2 = z * z
    denom = 1 + z2 / n

    center = (p + z2 / (2 * n)) / denom
    half = (
        z
        * math.sqrt(
            p * (1 - p) / n
            + z2 / (4 * n * n)
        )
        / denom
    )

    return max(0.0, center - half), min(1.0, center + half)


print("[1/7] Loading existing probe outputs...")

rev = pd.read_csv(REPORTS / "probe_revision_labels.csv")
final = pd.read_csv(REPORTS / "test0_final_states.csv")
articles = pd.read_csv(RAW / "articles.csv")
pv = pd.read_csv(RAW / "pageviews.csv")

rev["lead_changed"] = as_bool(rev["lead_changed"])
# v0.1 label_flip is not reliably anchor-relative.
# Recompute truth directly from anchor and fresh labels.
rev["label_flip"] = (
    rev["fresh_label"].astype(int)
    != rev["anchor_label"].astype(int)
)
rev["is_revert"] = as_bool(rev["is_revert"])

final["final_stale"] = pd.to_numeric(
    final["final_stale"], errors="coerce"
).fillna(0).astype(int)

final["pageviews"] = pd.to_numeric(
    final["pageviews"], errors="coerce"
)

print(
    f"      {len(final):,} evaluated articles, "
    f"{len(rev):,} evaluated states"
)

print("[2/7] Building churn statistics...")

lead_events = rev[rev["lead_changed"]].copy()

lead_counts = (
    lead_events.groupby("article_id")
    .size()
    .rename("lead_change_count")
)

final = final.merge(
    lead_counts,
    how="left",
    left_on="article_id",
    right_index=True,
)

final["lead_change_count"] = (
    final["lead_change_count"]
    .fillna(0)
    .astype(int)
)

final["churn_band"] = "churn_0"
final.loc[
    final["lead_change_count"] == 1,
    "churn_band"
] = "churn_1"

final.loc[
    final["lead_change_count"] >= 2,
    "churn_band"
] = "churn_2plus"

print("[3/7] Building real read-demand populations...")

# Important: don't interpret a missing pageview API result as zero demand.
titles_with_pageviews = set(pv["title"].dropna().unique())

final["has_pageview_data"] = final["title"].isin(
    titles_with_pageviews
)

demand_df = final[
    final["has_pageview_data"]
].copy()

demand_df["pageviews"] = demand_df["pageviews"].fillna(0)

total_demand = demand_df["pageviews"].sum()

if total_demand <= 0:
    raise RuntimeError("No usable pageview demand found.")

demand_df = demand_df.sort_values(
    ["pageviews", "article_id"],
    ascending=[False, True],
).copy()

demand_df["cum_before"] = (
    demand_df["pageviews"].cumsum()
    - demand_df["pageviews"]
) / total_demand


def demand_band(x):
    if x < 0.50:
        return "read_top_50pct_demand"
    if x < 0.80:
        return "read_next_30pct_demand"
    return "read_tail_20pct_demand"


demand_df["read_band"] = demand_df[
    "cum_before"
].map(demand_band)

final = final.merge(
    demand_df[["article_id", "read_band"]],
    on="article_id",
    how="left",
)

print("[4/7] Computing current-state population error...")

usable_demand = final[
    final["has_pageview_data"]
].copy()

all_demand = usable_demand["pageviews"].sum()


def summarize_population(name, sub):
    n = len(sub)
    stale = int(sub["final_stale"].sum())

    rate = stale / n if n else float("nan")
    lo, hi = wilson_interval(stale, n)

    with_pv = sub[
        sub["has_pageview_data"]
    ].copy()

    population_demand = with_pv["pageviews"].sum()

    demand_share = (
        population_demand / all_demand
        if all_demand > 0
        else float("nan")
    )

    weighted_stale = (
        (
            with_pv["pageviews"]
            * with_pv["final_stale"]
        ).sum()
        / population_demand
        if population_demand > 0
        else float("nan")
    )

    point_threshold_pass = (
        demand_share >= MIN_DEMAND_SHARE
        and rate >= HETEROGENEITY_MULTIPLE * EPSILON
    )

    # Stricter diagnostic:
    # lower 95% Wilson bound is also above epsilon.
    strict_pass = (
        point_threshold_pass
        and lo > EPSILON
    )

    return {
        "population": name,
        "articles": n,
        "stale_articles": stale,
        "state_stale_rate": rate,
        "ci95_low": lo,
        "ci95_high": hi,
        "read_demand_share": demand_share,
        "pageview_weighted_stale_rate": weighted_stale,
        "gate_a_point_candidate": point_threshold_pass,
        "gate_a_strict_candidate": strict_pass,
    }


rows = []

rows.append(
    summarize_population("ALL", final)
)

for band in [
    "read_top_50pct_demand",
    "read_next_30pct_demand",
    "read_tail_20pct_demand",
]:
    rows.append(
        summarize_population(
            band,
            final[final["read_band"] == band],
        )
    )

for band in [
    "churn_0",
    "churn_1",
    "churn_2plus",
]:
    rows.append(
        summarize_population(
            band,
            final[final["churn_band"] == band],
        )
    )

pop = pd.DataFrame(rows)

print("[5/7] Computing score-conditional event miss diagnostics...")

events = lead_events[
    lead_events["anchor_distance"].notna()
].copy()

events["anchor_distance"] = pd.to_numeric(
    events["anchor_distance"],
    errors="coerce",
)

events = events[
    events["anchor_distance"].notna()
].copy()

score_rows = []

if len(events) >= 5:
    # Equal-count bins are diagnostic only.
    # This avoids tiny cells in the 105-event probe.
    try:
        events["score_bin"] = pd.qcut(
            events["anchor_distance"],
            q=5,
            duplicates="drop",
        )

        for interval, sub in events.groupby(
            "score_bin",
            observed=True,
        ):
            n = len(sub)
            misses = int(sub["label_flip"].sum())
            rate = misses / n
            lo, hi = wilson_interval(misses, n)

            score_rows.append({
                "score_bin": str(interval),
                "events": n,
                "misses": misses,
                "miss_rate": rate,
                "ci95_low": lo,
                "ci95_high": hi,
                "score_min": sub["anchor_distance"].min(),
                "score_max": sub["anchor_distance"].max(),
            })
    except ValueError:
        pass

score_df = pd.DataFrame(score_rows)

print("[6/7] Accounting for exclusions...")

evaluated_ids = set(final["article_id"])

excluded = articles[
    ~articles["article_id"].isin(evaluated_ids)
].copy()

excluded["reason"] = (
    "not_present_in_test0_final_states"
)

missing_pv = final[
    ~final["has_pageview_data"]
][["article_id", "title"]].copy()

missing_pv["reason"] = "missing_pageview_data"

# Save outputs
pop.to_csv(
    REPORTS / "phase0a_lite_populations.csv",
    index=False,
)

score_df.to_csv(
    REPORTS / "phase0a_lite_score_bins.csv",
    index=False,
)

excluded.to_csv(
    REPORTS / "phase0a_lite_exclusions.csv",
    index=False,
)

missing_pv.to_csv(
    REPORTS / "phase0a_lite_missing_pageviews.csv",
    index=False,
)

gate_candidates = pop[
    (pop["population"] != "ALL")
    & pop["gate_a_point_candidate"]
]

strict_candidates = pop[
    (pop["population"] != "ALL")
    & pop["gate_a_strict_candidate"]
]

summary = {
    "phase": "Phase 0A-lite",
    "formal_phase0a": False,
    "note": (
        "Diagnostic reuse of the 500-article probe. "
        "No new model inference performed."
    ),
    "epsilon": EPSILON,
    "evaluated_articles": int(len(final)),
    "lead_change_states": int(len(lead_events)),
    "global_final_state_stale_rate": float(
        final["final_stale"].mean()
    ),
    "global_stale_articles": int(
        final["final_stale"].sum()
    ),
    "pageview_covered_articles": int(
        final["has_pageview_data"].sum()
    ),
    "excluded_articles": int(len(excluded)),
    "gate_a_point_pass": bool(len(gate_candidates)),
    "gate_a_strict_pass": bool(len(strict_candidates)),
    "gate_a_point_populations": gate_candidates[
        "population"
    ].tolist(),
    "gate_a_strict_populations": strict_candidates[
        "population"
    ].tolist(),
}

with open(
    REPORTS / "phase0a_lite_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

print("[7/7] Complete.")
print()

print("=== PHASE 0A-LITE SUMMARY ===")
print(json.dumps(summary, indent=2))

print()
print("=== POPULATIONS ===")

display_cols = [
    "population",
    "articles",
    "stale_articles",
    "state_stale_rate",
    "ci95_low",
    "ci95_high",
    "read_demand_share",
    "pageview_weighted_stale_rate",
    "gate_a_point_candidate",
    "gate_a_strict_candidate",
]

print(
    pop[display_cols]
    .to_string(index=False)
)

print()
print("Reports written:")
print("  reports/phase0a_lite_summary.json")
print("  reports/phase0a_lite_populations.csv")
print("  reports/phase0a_lite_score_bins.csv")
print("  reports/phase0a_lite_exclusions.csv")
print("  reports/phase0a_lite_missing_pageviews.csv")
