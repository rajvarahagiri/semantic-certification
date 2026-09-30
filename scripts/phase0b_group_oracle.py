from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(".")
REPORTS = ROOT / "reports"
RAW = ROOT / "data" / "raw"

EPSILON = 0.005

TUNE_START = pd.Timestamp("2026-08-03T00:00:00Z")
TUNE_END   = pd.Timestamp("2026-08-17T00:00:00Z")
EVAL_START = pd.Timestamp("2026-08-17T00:00:00Z")
EVAL_END   = pd.Timestamp("2026-08-31T00:00:00Z")

GATE_MODEL = "BAAI/bge-small-en-v1.5"
GATE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

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
    2.0,
]


def pct(x):
    return f"{100*x:.3f}%"


print("\n=== Phase 0B Group-Aware Oracle ===")
print("Tune:     Aug 3–16")
print("Evaluate: Aug 17–30")
print("NO NEW QWEN CALLS\n")

# ---------------------------------------------------------
# 1. Load labels
# ---------------------------------------------------------
print("[1/8] Loading existing labels...")

labels = pd.read_csv(REPORTS / "probe_revision_labels.csv")
labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
labels["article_id"] = labels["article_id"].astype(int)
labels["revision_id"] = labels["revision_id"].astype(int)
labels["fresh_label"] = labels["fresh_label"].astype(int)

print(f"      rows:     {len(labels)}")
print(f"      articles: {labels['article_id'].nunique()}")

# ---------------------------------------------------------
# 2. Join raw lead text
# ---------------------------------------------------------
print("[2/8] Joining normalized lead text...")

wanted = set(labels["revision_id"])
rev_text = {}

with open(RAW / "revisions.jsonl") as f:
    for line in f:
        r = json.loads(line)
        rid = int(r["revision_id"])
        if rid in wanted:
            rev_text[rid] = r.get("normalized_lead", "")

missing = wanted - set(rev_text)
if missing:
    raise RuntimeError(f"Missing lead text for {len(missing)} revisions")

labels["normalized_lead"] = labels["revision_id"].map(rev_text)

# ---------------------------------------------------------
# 3. Embeddings
# ---------------------------------------------------------
print("[3/8] Embedding 536 states with BGE...")

model = SentenceTransformer(
    GATE_MODEL,
    revision=GATE_REVISION,
    device="cpu",
)

emb = model.encode(
    labels["normalized_lead"].fillna("").tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

emb_by_idx = {
    idx: emb[pos]
    for pos, idx in enumerate(labels.index)
}

# ---------------------------------------------------------
# 4. Build timeline / anchors
# ---------------------------------------------------------
print("[4/8] Building timeline...")

labels = labels.sort_values(
    ["timestamp", "article_id", "revision_id"]
).copy()

initial_idx = {}

for aid, sub in labels.groupby("article_id"):
    initial_idx[aid] = sub.index[0]

anchor_indices = set(initial_idx.values())

events = labels.loc[
    ~labels.index.isin(anchor_indices)
].sort_values("timestamp").copy()

tune_events = events[
    (events["timestamp"] >= TUNE_START)
    & (events["timestamp"] < TUNE_END)
].copy()

eval_events = events[
    (events["timestamp"] >= EVAL_START)
    & (events["timestamp"] < EVAL_END)
].copy()

print(f"      tune events: {len(tune_events)}")
print(f"      eval events: {len(eval_events)}")

# ---------------------------------------------------------
# 5. Define protected population USING TUNE ONLY
# ---------------------------------------------------------
print("[5/8] Freezing protected population from tuning window...")

tune_churn = (
    tune_events.groupby("article_id")
    .size()
    .rename("tune_lead_changes")
)

all_ids = sorted(labels["article_id"].unique())

meta = pd.DataFrame(index=all_ids)
meta.index.name = "article_id"

meta = meta.join(tune_churn)
meta["tune_lead_changes"] = (
    meta["tune_lead_changes"]
    .fillna(0)
    .astype(int)
)

# FROZEN RULE:
# protected = article had >= 2 lead-changing events during tuning.
protected_ids = set(
    meta[
        meta["tune_lead_changes"] >= 2
    ].index
)

print(
    f"      protected articles (>=2 tune lead changes): "
    f"{len(protected_ids)}"
)

if len(protected_ids) == 0:
    raise RuntimeError(
        "No articles meet the preregistered >=2 tune-change protected rule. "
        "Do not redefine it after seeing evaluation outcomes."
    )

# ---------------------------------------------------------
# 6. Real pageviews split by time
# ---------------------------------------------------------
print("[6/8] Loading tuning/evaluation pageview demand...")

pv = pd.read_csv(RAW / "pageviews.csv")
pv["timestamp"] = pd.to_datetime(
    pv["timestamp"].astype(str),
    format="%Y%m%d%H",
    utc=True,
)
pv["views"] = pd.to_numeric(pv["views"], errors="coerce").fillna(0)

title_by_article = (
    labels.groupby("article_id")["title"].first().to_dict()
)

tune_pv_by_title = (
    pv[
        (pv["timestamp"] >= TUNE_START)
        & (pv["timestamp"] < TUNE_END)
    ]
    .groupby("title")["views"]
    .sum()
)

eval_pv_by_title = (
    pv[
        (pv["timestamp"] >= EVAL_START)
        & (pv["timestamp"] < EVAL_END)
    ]
    .groupby("title")["views"]
    .sum()
)

tune_demand = {
    aid: float(tune_pv_by_title.get(title_by_article[aid], 0))
    for aid in all_ids
}

eval_demand = {
    aid: float(eval_pv_by_title.get(title_by_article[aid], 0))
    for aid in all_ids
}

protected_tune_share = (
    sum(tune_demand[x] for x in protected_ids)
    / max(1, sum(tune_demand.values()))
)

protected_eval_share = (
    sum(eval_demand[x] for x in protected_ids)
    / max(1, sum(eval_demand.values()))
)

print(f"      protected tuning demand share: {pct(protected_tune_share)}")
print(f"      protected eval demand share:   {pct(protected_eval_share)}")

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def initial_state():
    s = {}

    for aid in all_ids:
        idx = initial_idx[aid]
        row = labels.loc[idx]

        s[aid] = {
            "anchor_embedding": emb_by_idx[idx],
            "stored_label": int(row["fresh_label"]),
            "fresh_label": int(row["fresh_label"]),
            "stale": False,
        }

    return s


def run_policy(
    protected_threshold,
    general_threshold,
    event_df,
    state,
    demand,
):
    recomputes = 0
    skips = 0

    global_trace = []
    weighted_trace = []
    protected_trace = []

    for idx, row in event_df.iterrows():
        aid = int(row["article_id"])
        current_emb = emb_by_idx[idx]

        score = max(
            0.0,
            float(
                1.0
                - np.dot(
                    state[aid]["anchor_embedding"],
                    current_emb,
                )
            ),
        )

        threshold = (
            protected_threshold
            if aid in protected_ids
            else general_threshold
        )

        fresh = int(row["fresh_label"])

        if score >= threshold:
            recomputes += 1
            state[aid]["stored_label"] = fresh
            state[aid]["anchor_embedding"] = current_emb
        else:
            skips += 1

        state[aid]["fresh_label"] = fresh
        state[aid]["stale"] = (
            state[aid]["stored_label"] != fresh
        )

        # Global state error
        global_stale = (
            sum(int(v["stale"]) for v in state.values())
            / len(state)
        )

        # Pageview-weighted state error
        total_demand = sum(demand.values())
        stale_demand = sum(
            demand[aid]
            for aid, v in state.items()
            if v["stale"]
        )

        weighted_stale = (
            stale_demand / total_demand
            if total_demand > 0
            else 0.0
        )

        # Protected group
        protected_stale = (
            sum(
                int(state[x]["stale"])
                for x in protected_ids
            )
            / len(protected_ids)
        )

        global_trace.append(global_stale)
        weighted_trace.append(weighted_stale)
        protected_trace.append(protected_stale)

    return {
        "state": state,
        "events": len(event_df),
        "recomputes": recomputes,
        "skips": skips,
        "savings": (
            skips / len(event_df)
            if len(event_df)
            else 0
        ),
        "max_global": max(global_trace, default=0),
        "max_weighted": max(weighted_trace, default=0),
        "max_protected": max(protected_trace, default=0),
        "final_global": global_trace[-1] if global_trace else 0,
        "final_weighted": weighted_trace[-1] if weighted_trace else 0,
        "final_protected": protected_trace[-1] if protected_trace else 0,
    }


def tune_candidate(pt, gt):
    state = initial_state()

    r = run_policy(
        pt,
        gt,
        tune_events,
        state,
        tune_demand,
    )

    valid = (
        r["max_global"] <= EPSILON
        and r["max_weighted"] <= EPSILON
        and r["max_protected"] <= EPSILON
    )

    return r, valid


# ---------------------------------------------------------
# 7. Tune global and group-aware policies
# ---------------------------------------------------------
print("[7/8] Tuning policies on Aug 3–16 ONLY...")

# Global baseline: same threshold everywhere.
global_candidates = []

for t in THRESHOLDS:
    r, valid = tune_candidate(t, t)

    if valid:
        global_candidates.append({
            "threshold": t,
            "savings": r["savings"],
        })

if not global_candidates:
    raise RuntimeError("No valid global threshold on tuning window.")

best_global = max(
    global_candidates,
    key=lambda x: (x["savings"], x["threshold"]),
)

# Group-aware controller:
# Protected population can only be as strict or stricter than general.
group_candidates = []

pairs = [
    (pt, gt)
    for pt, gt in itertools.product(THRESHOLDS, THRESHOLDS)
    if pt <= gt
]

for n, (pt, gt) in enumerate(pairs, start=1):
    r, valid = tune_candidate(pt, gt)

    if valid:
        group_candidates.append({
            "protected_threshold": pt,
            "general_threshold": gt,
            "savings": r["savings"],
        })

    if n % 20 == 0 or n == len(pairs):
        print(
            f"      tuning pairs: {n}/{len(pairs)} "
            f"({100*n/len(pairs):.1f}%)"
        )

if not group_candidates:
    raise RuntimeError("No valid group-aware policy on tuning window.")

best_group = max(
    group_candidates,
    key=lambda x: (
        x["savings"],
        x["general_threshold"],
        -x["protected_threshold"],
    ),
)

print()
print("Frozen after tuning:")
print(
    f"  GLOBAL threshold: "
    f"{best_global['threshold']}"
)
print(
    f"  GROUP protected τ: "
    f"{best_group['protected_threshold']}"
)
print(
    f"  GROUP general τ:   "
    f"{best_group['general_threshold']}"
)

# ---------------------------------------------------------
# 8. Untouched evaluation
# ---------------------------------------------------------
print("\n[8/8] Evaluating frozen policies on Aug 17–30...")

# Replay tuning first to construct each policy's real eval-start state.
global_state = initial_state()
run_policy(
    best_global["threshold"],
    best_global["threshold"],
    tune_events,
    global_state,
    tune_demand,
)

group_state = initial_state()
run_policy(
    best_group["protected_threshold"],
    best_group["general_threshold"],
    tune_events,
    group_state,
    tune_demand,
)

global_eval = run_policy(
    best_global["threshold"],
    best_global["threshold"],
    eval_events,
    global_state,
    eval_demand,
)

group_eval = run_policy(
    best_group["protected_threshold"],
    best_group["general_threshold"],
    eval_events,
    group_state,
    eval_demand,
)

summary = {
    "formal_statistical_certification": False,
    "oracle_evaluation": True,
    "epsilon": EPSILON,

    "tune_events": len(tune_events),
    "eval_events": len(eval_events),

    "protected_rule": ">=2 lead-changing events during tuning window",
    "protected_articles": len(protected_ids),
    "protected_tune_read_share": protected_tune_share,
    "protected_eval_read_share": protected_eval_share,

    "global_policy": {
        "threshold": best_global["threshold"],
        **{k: v for k, v in global_eval.items() if k != "state"},
    },

    "group_policy": {
        "protected_threshold": best_group["protected_threshold"],
        "general_threshold": best_group["general_threshold"],
        **{k: v for k, v in group_eval.items() if k != "state"},
    },
}

with open(
    REPORTS / "phase0b_group_oracle_summary.json",
    "w",
) as f:
    json.dump(summary, f, indent=2)

comparison = pd.DataFrame([
    {
        "policy": "global",
        "protected_threshold": best_global["threshold"],
        "general_threshold": best_global["threshold"],
        "eval_savings": global_eval["savings"],
        "max_global": global_eval["max_global"],
        "max_weighted": global_eval["max_weighted"],
        "max_protected": global_eval["max_protected"],
    },
    {
        "policy": "group_aware",
        "protected_threshold": best_group["protected_threshold"],
        "general_threshold": best_group["general_threshold"],
        "eval_savings": group_eval["savings"],
        "max_global": group_eval["max_global"],
        "max_weighted": group_eval["max_weighted"],
        "max_protected": group_eval["max_protected"],
    },
])

comparison.to_csv(
    REPORTS / "phase0b_group_oracle_comparison.csv",
    index=False,
)

print("\n=== UNTOUCHED EVALUATION RESULTS ===\n")
print(comparison.to_string(index=False))

print("\n=== INTERPRETATION ===")

group_pass = (
    group_eval["max_global"] <= EPSILON
    and group_eval["max_weighted"] <= EPSILON
    and group_eval["max_protected"] <= EPSILON
)

global_pass = (
    global_eval["max_global"] <= EPSILON
    and global_eval["max_weighted"] <= EPSILON
    and global_eval["max_protected"] <= EPSILON
)

print(
    f"Global policy valid on eval:      {global_pass}"
)
print(
    f"Group-aware policy valid on eval: {group_pass}"
)

delta = (
    group_eval["savings"]
    - global_eval["savings"]
)

print(
    f"Savings improvement:              {pct(delta)}"
)

if group_pass and group_eval["savings"] >= 0.50:
    print(
        "\nRESULT: STRONG CONTINUE SIGNAL — "
        "group-aware controller preserved <=0.5% "
        "observed trace error while saving >=50% "
        "of full-model calls."
    )
elif group_pass and group_eval["savings"] > global_eval["savings"]:
    print(
        "\nRESULT: MODEST CONTINUE SIGNAL — "
        "group-aware controller improves on global gate, "
        "but savings remain below the 50% target."
    )
else:
    print(
        "\nRESULT: KILL / MAJOR WARNING — "
        "group-aware controller did not demonstrate the "
        "required out-of-sample advantage."
    )

print(
    "\nNo new Qwen inference was performed."
)
