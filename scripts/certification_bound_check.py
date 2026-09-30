import math
import pandas as pd
from scipy.stats import beta

EPSILON = 0.005
ALPHA = 0.05

TUNE_START = pd.Timestamp("2026-08-03T00:00:00Z")
TUNE_END   = pd.Timestamp("2026-08-17T00:00:00Z")

THRESHOLDS = [
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

df = pd.read_csv("reports/probe_revision_labels.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# IMPORTANT:
# Ignore v0.1 label_flip. Recompute anchor-relative truth.
df["flip"] = (
    df["fresh_label"].astype(int)
    != df["anchor_label"].astype(int)
)

# First row/article is initial anchor, not an event.
df = df.sort_values(["timestamp", "article_id", "revision_id"])
first = set(df.groupby("article_id").head(1).index)
events = df.loc[~df.index.isin(first)].copy()

tune = events[
    (events["timestamp"] >= TUNE_START)
    & (events["timestamp"] < TUNE_END)
].copy()

tune["anchor_distance"] = pd.to_numeric(
    tune["anchor_distance"], errors="coerce"
)
tune = tune[tune["anchor_distance"].notna()]


def cp_upper(k, n, alpha=0.05):
    """One-sided exact Clopper-Pearson upper confidence bound."""
    if n == 0:
        return float("nan")
    if k == n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def capacity(q, population_size):
    """
    With zero starting debt and q per indebted object:
        debt = (# indebted objects * q) / population_size
    Return max concurrent indebted objects before EPSILON is exceeded.
    """
    if not math.isfinite(q) or q <= 0:
        return None
    return math.floor(EPSILON * population_size / q)


print("=== CERTIFICATION BOUND CHECK ===")
print()
print("Tuning events:", len(tune))
print("Corrected anchor-relative flips:", int(tune["flip"].sum()))
print(
    "Observed tuning flip rate:",
    f"{100*tune['flip'].mean():.3f}%"
)
print()

# Fundamental sample-size limit.
n_zero = math.ceil(
    math.log(ALPHA) / math.log(1 - EPSILON)
)

best_possible = 1 - ALPHA ** (1 / len(tune))

print("=== FUNDAMENTAL SAMPLE-SIZE CHECK ===")
print(
    "Events needed for 95% upper bound <=0.5%",
    "with ZERO misses:",
    n_zero,
)
print(
    f"Best possible bound with only {len(tune)} events "
    f"and hypothetically ZERO misses:",
    f"{100*best_possible:.3f}%"
)
print()

print("=== SCORE-CONDITIONAL SKIP REGIONS ===")
print()
print(
    f"{'threshold':>10} "
    f"{'audits':>7} "
    f"{'misses':>7} "
    f"{'empirical':>10} "
    f"{'95% UB':>10} "
    f"{'cap N=431':>10} "
    f"{'cap N=13':>9}"
)

rows = []

for threshold in THRESHOLDS:

    # Operational interpretation:
    # score < threshold => gate would consider skipping.
    sub = tune[tune["anchor_distance"] < threshold]

    n = len(sub)
    k = int(sub["flip"].sum())

    if n:
        empirical = k / n
        ub = cp_upper(k, n)
        cap431 = capacity(ub, 431)
        cap13 = capacity(ub, 13)
    else:
        empirical = float("nan")
        ub = float("nan")
        cap431 = None
        cap13 = None

    rows.append({
        "threshold": threshold,
        "audits": n,
        "misses": k,
        "empirical_miss_rate": empirical,
        "upper95": ub,
        "capacity_431": cap431,
        "capacity_13": cap13,
    })

    emp_text = (
        f"{100*empirical:9.3f}%"
        if math.isfinite(empirical)
        else "       n/a"
    )

    ub_text = (
        f"{100*ub:9.3f}%"
        if math.isfinite(ub)
        else "       n/a"
    )

    print(
        f"{threshold:10g} "
        f"{n:7d} "
        f"{k:7d} "
        f"{emp_text} "
        f"{ub_text} "
        f"{str(cap431):>10} "
        f"{str(cap13):>9}"
    )


print()
print("=== ALL TUNING EVENTS, IGNORING SCORE ===")

n = len(tune)
k = int(tune["flip"].sum())
ub = cp_upper(k, n)

print("audits:", n)
print("misses:", k)
print("empirical miss rate:", f"{100*k/n:.3f}%")
print("95% upper miss bound:", f"{100*ub:.3f}%")
print(
    "N=431 max concurrent indebted objects:",
    capacity(ub, 431),
)
print(
    "N=13 max concurrent indebted objects:",
    capacity(ub, 13),
)

pd.DataFrame(rows).to_csv(
    "reports/certification_bound_check.csv",
    index=False,
)

print()
print("=== INTERPRETATION GUIDE ===")
print(
    "capacity_431 = how many distinct current objects could carry "
    "that q_i bound simultaneously before projected average debt "
    "alone exceeds 0.5%, assuming starting debt = 0."
)
print(
    "capacity_13 does the same calculation for the 13-object "
    "population."
)
print()
print(
    "This is a calibration-feasibility diagnostic, not final "
    "certification. It uses the existing anchor-distance field and "
    "does not yet account for dynamic anchor resets."
)
print()
print("Saved: reports/certification_bound_check.csv")
