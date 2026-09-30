from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

EPSILON = 0.005

ROOT = Path(".")
REPORTS = ROOT / "reports"
RAW = ROOT / "data" / "raw"

LABELS_FILE = REPORTS / "probe_revision_labels.csv"
REVISIONS_FILE = RAW / "revisions.jsonl"

GATE_MODEL = "BAAI/bge-small-en-v1.5"
GATE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# Recompute when anchor-distance >= threshold.
# 0.0 = recompute every lead change.
# 2.0 = effectively never recompute.
THRESHOLDS = [
    0.0,
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.002,
    0.004,
    0.008,
    0.012,
    0.020,
    0.050,
    0.100,
    2.000,
]


def pct(x):
    return f"{100*x:.3f}%"


print("\n=== Phase 0A Oracle Controller Simulation ===")
print("NO NEW QWEN CALLS")
print()

print("[1/6] Loading existing Qwen labels...")

labels = pd.read_csv(LABELS_FILE)
labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
labels["revision_id"] = labels["revision_id"].astype(int)
labels["article_id"] = labels["article_id"].astype(int)
labels["fresh_label"] = labels["fresh_label"].astype(int)
labels["pageviews"] = pd.to_numeric(labels["pageviews"], errors="coerce").fillna(0)

needed_revision_ids = set(labels["revision_id"])

print(f"      labeled states: {len(labels):,}")
print(f"      articles:       {labels['article_id'].nunique():,}")

print("[2/6] Loading normalized lead text...")

revision_text = {}

with open(REVISIONS_FILE) as f:
    for line in f:
        r = json.loads(line)
        rid = int(r["revision_id"])
        if rid in needed_revision_ids:
            revision_text[rid] = r.get("normalized_lead", "")

missing = needed_revision_ids - set(revision_text)

if missing:
    raise RuntimeError(
        f"{len(missing)} labeled revisions are missing normalized lead text."
    )

labels["normalized_lead"] = labels["revision_id"].map(revision_text)

print(f"      matched texts:  {len(revision_text):,}")

print("[3/6] Loading BGE-small and embedding states...")

start = time.time()

model = SentenceTransformer(
    GATE_MODEL,
    revision=GATE_REVISION,
    device="cpu",       # predictable and fast enough for ~536 texts
)

texts = labels["normalized_lead"].fillna("").tolist()

emb = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

elapsed = time.time() - start

print(f"      embeddings complete in {elapsed:.1f}s")

# Map DataFrame row index -> normalized embedding.
embedding_by_index = {
    idx: emb[pos]
    for pos, idx in enumerate(labels.index)
}

print("[4/6] Building replay timeline...")

labels = labels.sort_values(
    ["timestamp", "article_id", "revision_id"]
).copy()

# Earliest labeled row per article is the full-inference anchor.
anchor_rows = (
    labels.sort_values(["timestamp", "revision_id"])
    .groupby("article_id", as_index=False)
    .first()
)

anchor_indices = set()

for article_id in labels["article_id"].unique():
    article_rows = labels[labels["article_id"] == article_id]
    anchor_indices.add(article_rows.index[0])

events = labels.loc[
    ~labels.index.isin(anchor_indices)
].copy()

# Only the 105 lead-changing labeled states should be here.
print(f"      initial anchors: {len(anchor_indices):,}")
print(f"      replay events:   {len(events):,}")

if len(events) == 0:
    raise RuntimeError("No replay events found.")

# Static article metadata.
article_meta = (
    labels.groupby("article_id")
    .agg(
        title=("title", "first"),
        pageviews=("pageviews", "max"),
    )
)

lead_change_counts = (
    events.groupby("article_id")
    .size()
    .rename("lead_change_count")
)

article_meta = article_meta.join(lead_change_counts).fillna(
    {"lead_change_count": 0}
)

article_meta["lead_change_count"] = (
    article_meta["lead_change_count"].astype(int)
)

# The Phase-0A-lite population that looked interesting.
churn2_ids = set(
    article_meta[
        article_meta["lead_change_count"] >= 2
    ].index
)

all_ids = list(article_meta.index)

total_pageviews = article_meta["pageviews"].sum()

print(
    f"      churn_2plus:     {len(churn2_ids):,} articles, "
    f"{article_meta.loc[list(churn2_ids), 'pageviews'].sum()/total_pageviews:.1%} "
    "of read demand"
)

print("[5/6] Replaying threshold policies...")

rows = []

for n, threshold in enumerate(THRESHOLDS, start=1):

    # State per article.
    state = {}

    for article_id in all_ids:
        article_rows = labels[labels["article_id"] == article_id]
        idx = article_rows.index[0]
        row = article_rows.loc[idx]

        state[article_id] = {
            "anchor_embedding": embedding_by_index[idx],
            "stored_label": int(row["fresh_label"]),
            "fresh_label": int(row["fresh_label"]),
            "stale": False,
        }

    recomputes = 0
    skips = 0

    stale_rates = []
    weighted_stale_rates = []
    churn2_stale_rates = []

    scores = []

    for event_num, (idx, row) in enumerate(events.iterrows(), start=1):

        aid = int(row["article_id"])
        current_emb = embedding_by_index[idx]

        anchor_emb = state[aid]["anchor_embedding"]

        # Embeddings are already normalized.
        score = float(
            max(
                0.0,
                1.0 - np.dot(anchor_emb, current_emb)
            )
        )

        scores.append(score)

        fresh = int(row["fresh_label"])

        if score >= threshold:
            # Full model recompute.
            recomputes += 1

            state[aid]["stored_label"] = fresh
            state[aid]["anchor_embedding"] = current_emb

        else:
            # Keep the previously stored derived label.
            skips += 1

        state[aid]["fresh_label"] = fresh
        state[aid]["stale"] = (
            state[aid]["stored_label"] != fresh
        )

        # Actual current-state stale fraction.
        stale_count = sum(
            int(v["stale"])
            for v in state.values()
        )

        global_stale = stale_count / len(state)

        # Pageview-weighted current-state stale fraction.
        stale_pv = 0.0

        for article_id, v in state.items():
            if v["stale"]:
                stale_pv += article_meta.loc[
                    article_id, "pageviews"
                ]

        weighted_stale = (
            stale_pv / total_pageviews
            if total_pageviews > 0
            else 0.0
        )

        if churn2_ids:
            churn_stale = sum(
                int(state[x]["stale"])
                for x in churn2_ids
            ) / len(churn2_ids)
        else:
            churn_stale = 0.0

        stale_rates.append(global_stale)
        weighted_stale_rates.append(weighted_stale)
        churn2_stale_rates.append(churn_stale)

    baseline_calls = len(events)

    savings = (
        1 - recomputes / baseline_calls
        if baseline_calls
        else 0
    )

    final_global = stale_rates[-1]
    final_weighted = weighted_stale_rates[-1]
    final_churn = churn2_stale_rates[-1]

    max_global = max(stale_rates)
    max_weighted = max(weighted_stale_rates)
    max_churn = max(churn2_stale_rates)

    fraction_global_valid = np.mean(
        np.array(stale_rates) <= EPSILON
    )

    fraction_weighted_valid = np.mean(
        np.array(weighted_stale_rates) <= EPSILON
    )

    fraction_churn_valid = np.mean(
        np.array(churn2_stale_rates) <= EPSILON
    )

    all_constraints_valid = (
        max_global <= EPSILON
        and max_weighted <= EPSILON
        and max_churn <= EPSILON
    )

    rows.append({
        "threshold": threshold,
        "lead_change_events": baseline_calls,
        "recomputes": recomputes,
        "skips": skips,
        "recompute_rate": recomputes / baseline_calls,
        "model_call_savings": savings,

        "final_global_stale": final_global,
        "max_global_stale": max_global,

        "final_pageview_weighted_stale": final_weighted,
        "max_pageview_weighted_stale": max_weighted,

        "final_churn2plus_stale": final_churn,
        "max_churn2plus_stale": max_churn,

        "fraction_events_global_at_epsilon": fraction_global_valid,
        "fraction_events_weighted_at_epsilon": fraction_weighted_valid,
        "fraction_events_churn2_at_epsilon": fraction_churn_valid,

        "all_trace_constraints_at_epsilon": all_constraints_valid,

        "mean_dynamic_anchor_distance": float(np.mean(scores)),
        "max_dynamic_anchor_distance": float(np.max(scores)),
    })

    print(
        f"      [{n:02d}/{len(THRESHOLDS)}] "
        f"τ={threshold:<7g} "
        f"recompute={recomputes:3d}/{baseline_calls} "
        f"savings={pct(savings):>8} "
        f"max-global={pct(max_global):>8} "
        f"max-churn2={pct(max_churn):>8}"
    )

results = pd.DataFrame(rows)

print("[6/6] Writing reports...")

out_csv = REPORTS / "phase0a_oracle_thresholds.csv"
results.to_csv(out_csv, index=False)

# Find the highest-savings policy that never violates epsilon
# for the observed trace.
valid = results[
    results["all_trace_constraints_at_epsilon"]
].sort_values("model_call_savings", ascending=False)

if len(valid):
    best = valid.iloc[0].to_dict()
else:
    best = None

# Also find best policies under less stringent diagnostics.
global_valid = results[
    results["max_global_stale"] <= EPSILON
].sort_values("model_call_savings", ascending=False)

weighted_valid = results[
    results["max_pageview_weighted_stale"] <= EPSILON
].sort_values("model_call_savings", ascending=False)

summary = {
    "phase": "Phase 0A oracle threshold simulation",
    "formal_guarantee": False,
    "new_qwen_calls": 0,
    "epsilon": EPSILON,
    "articles": len(all_ids),
    "lead_change_events": len(events),
    "churn2plus_articles": len(churn2_ids),
    "best_all_constraints_policy": best,
    "best_global_only_policy": (
        global_valid.iloc[0].to_dict()
        if len(global_valid)
        else None
    ),
    "best_pageview_weighted_policy": (
        weighted_valid.iloc[0].to_dict()
        if len(weighted_valid)
        else None
    ),
}

with open(
    REPORTS / "phase0a_oracle_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

print()
print("=== ORACLE THRESHOLD RESULTS ===")
print()

show = results[[
    "threshold",
    "recomputes",
    "skips",
    "model_call_savings",
    "max_global_stale",
    "max_pageview_weighted_stale",
    "max_churn2plus_stale",
    "all_trace_constraints_at_epsilon",
]]

print(show.to_string(index=False))

print()
print("=== BEST POLICY ===")

if best is None:
    print(
        "NO threshold kept global, pageview-weighted, and "
        "churn_2plus current-state error <= 0.5% "
        "throughout the observed trace."
    )
else:
    print(
        f"threshold:          {best['threshold']}"
    )
    print(
        f"model-call savings: {pct(best['model_call_savings'])}"
    )
    print(
        f"recomputes:         {int(best['recomputes'])}"
        f"/{int(best['lead_change_events'])}"
    )
    print(
        f"max global stale:   {pct(best['max_global_stale'])}"
    )
    print(
        f"max weighted stale: {pct(best['max_pageview_weighted_stale'])}"
    )
    print(
        f"max churn2 stale:   {pct(best['max_churn2plus_stale'])}"
    )

print()
print(f"Saved: {out_csv}")
print("Saved: reports/phase0a_oracle_summary.json")
print()
print(
    "IMPORTANT: This is an oracle replay using Qwen labels already "
    "computed for evaluation. It is an economics/feasibility diagnostic, "
    "not a statistical certification result."
)
