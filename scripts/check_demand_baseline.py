import pandas as pd

labels = pd.read_csv("reports/probe_revision_labels.csv")
pv = pd.read_csv("data/raw/pageviews.csv")

labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
labels["article_id"] = labels["article_id"].astype(int)

pv["timestamp"] = pd.to_datetime(
    pv["timestamp"].astype(str),
    format="%Y%m%d%H",
    utc=True,
)
pv["views"] = pd.to_numeric(pv["views"], errors="coerce").fillna(0)

TUNE_START = pd.Timestamp("2026-08-03T00:00:00Z")
TUNE_END   = pd.Timestamp("2026-08-17T00:00:00Z")
EVAL_START = pd.Timestamp("2026-08-17T00:00:00Z")
EVAL_END   = pd.Timestamp("2026-08-31T00:00:00Z")

labels = labels.sort_values(
    ["timestamp", "article_id", "revision_id"]
)

# First labeled row/article = initial anchor.
first = set(labels.groupby("article_id").head(1).index)
events = labels.loc[~labels.index.isin(first)].copy()

tune_events = events[
    (events["timestamp"] >= TUNE_START) &
    (events["timestamp"] < TUNE_END)
]

eval_events = events[
    (events["timestamp"] >= EVAL_START) &
    (events["timestamp"] < EVAL_END)
].copy()

# Existing churn-defined group.
tune_counts = tune_events.groupby("article_id").size()

churn_ids = set(
    tune_counts[tune_counts >= 2].index
)

# Article/title map.
article_titles = (
    labels[["article_id", "title"]]
    .drop_duplicates("article_id")
)

# Tune and eval demand.
tune_views = (
    pv[
        (pv["timestamp"] >= TUNE_START) &
        (pv["timestamp"] < TUNE_END)
    ]
    .groupby("title")["views"]
    .sum()
)

eval_views = (
    pv[
        (pv["timestamp"] >= EVAL_START) &
        (pv["timestamp"] < EVAL_END)
    ]
    .groupby("title")["views"]
    .sum()
)

article_titles["tune_views"] = (
    article_titles["title"]
    .map(tune_views)
    .fillna(0)
)

article_titles["eval_views"] = (
    article_titles["title"]
    .map(eval_views)
    .fillna(0)
)

# Fair comparison:
# choose exactly 13 articles using ONLY tuning read demand.
demand_ids = set(
    article_titles
    .sort_values("tune_views", ascending=False)
    .head(len(churn_ids))["article_id"]
)

total_tune = article_titles["tune_views"].sum()
total_eval = article_titles["eval_views"].sum()


def report(name, ids):
    rows = article_titles[
        article_titles["article_id"].isin(ids)
    ]

    ev = eval_events[
        eval_events["article_id"].isin(ids)
    ].copy()

    flips = (
        ev["fresh_label"].astype(int)
        != ev["anchor_label"].astype(int)
    ).sum()

    print()
    print(name)
    print("-" * len(name))

    print("articles:", len(ids))

    print(
        "tune demand share:",
        f"{100 * rows['tune_views'].sum()/total_tune:.2f}%"
    )

    print(
        "eval demand share:",
        f"{100 * rows['eval_views'].sum()/total_eval:.2f}%"
    )

    print(
        "eval events:",
        len(ev),
        f"({100*len(ev)/len(eval_events):.2f}% of all eval changes)"
    )

    print(
        "eval changed articles:",
        ev["article_id"].nunique()
    )

    if flips is not None:
        print("eval anchor-relative flips:", int(flips))

    print(
        "eval events/article:",
        f"{len(ev)/len(ids):.4f}"
    )


print("=== SIMPLE-BASELINE CHECK ===")

print("Total evaluation events:", len(eval_events))

print(
    "Churn group / demand group overlap:",
    len(churn_ids & demand_ids),
    "of",
    len(churn_ids),
)

report(
    "CHURN-SELECTED 13",
    churn_ids,
)

report(
    "DEMAND-SELECTED 13",
    demand_ids,
)

print()
print("Demand-selected titles:")

print(
    article_titles[
        article_titles["article_id"].isin(demand_ids)
    ]
    .sort_values("tune_views", ascending=False)
    [["title", "tune_views", "eval_views"]]
    .to_string(index=False)
)
