import json
import pandas as pd

labels = pd.read_csv("reports/probe_revision_labels.csv")
labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
labels["article_id"] = labels["article_id"].astype(int)

TUNE_START = pd.Timestamp("2026-08-03T00:00:00Z")
TUNE_END   = pd.Timestamp("2026-08-17T00:00:00Z")
EVAL_START = pd.Timestamp("2026-08-17T00:00:00Z")
EVAL_END   = pd.Timestamp("2026-08-31T00:00:00Z")

labels = labels.sort_values(
    ["timestamp", "article_id", "revision_id"]
)

# First labeled state/article is initial anchor, remainder are lead-change states.
first_indices = set(
    labels.groupby("article_id").head(1).index
)

events = labels.loc[
    ~labels.index.isin(first_indices)
].copy()

tune = events[
    (events["timestamp"] >= TUNE_START) &
    (events["timestamp"] < TUNE_END)
]

eval_df = events[
    (events["timestamp"] >= EVAL_START) &
    (events["timestamp"] < EVAL_END)
].copy()

# Same frozen rule used in oracle.
tune_counts = tune.groupby("article_id").size()
protected = set(
    tune_counts[tune_counts >= 2].index
)

eval_df["protected"] = eval_df["article_id"].isin(protected)

print("=== PROTECTED POPULATION SUPPORT ===")
print("Protected articles:", len(protected))
print()

print("Evaluation events total:", len(eval_df))
print(
    "Protected events:",
    int(eval_df["protected"].sum()),
    f"({100*eval_df['protected'].mean():.2f}%)"
)
print(
    "General events:",
    int((~eval_df["protected"]).sum()),
    f"({100*(~eval_df['protected']).mean():.2f}%)"
)

print()
print("Articles with evaluation changes:")
print(
    "  protected:",
    eval_df.loc[
        eval_df["protected"], "article_id"
    ].nunique()
)
print(
    "  general:",
    eval_df.loc[
        ~eval_df["protected"], "article_id"
    ].nunique()
)

# Label flips in existing Qwen truth.
eval_df["label_flip"] = (
    eval_df["fresh_label"].astype(int) !=
    eval_df["anchor_label"].astype(int)
)

for name, sub in [
    ("protected", eval_df[eval_df["protected"]]),
    ("general", eval_df[~eval_df["protected"]]),
]:
    print()
    print(name.upper())
    print("  events:", len(sub))
    print("  anchor-relative flips:", int(sub["label_flip"].sum()))
    print(
        "  flip rate:",
        f"{100*sub['label_flip'].mean():.2f}%"
        if len(sub) else "n/a"
    )

print()
print("Protected article titles:")
titles = (
    labels[
        labels["article_id"].isin(protected)
    ][["article_id", "title"]]
    .drop_duplicates()
    .sort_values("title")
)

print(titles.to_string(index=False))
