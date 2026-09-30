from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json


def _bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes"}


def run_test0(project_dir: str | Path) -> dict:
    """Compute the never-recompute baseline from full fresh labels produced by the probe.

    This reports two quantities:
    - final-state stale fraction across articles;
    - lead-change-event-weighted stale fraction across evaluated states.

    The final state fraction is the closer Test-0 diagnostic to the current-state notion.
    Phase 0A later introduces explicit random state-audit sampling and population bounds.
    """
    root = Path(project_dir)
    path = root / "reports/probe_revision_labels.csv"
    if not path.exists():
        raise RuntimeError("Missing probe labels. Run: semcert run-probe")

    by_article: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["article_id"] = int(row["article_id"])
            row["revision_id"] = int(row["revision_id"])
            row["fresh_label"] = int(row["fresh_label"])
            row["anchor_label"] = int(row["anchor_label"])
            row["lead_changed"] = _bool(row["lead_changed"])
            row["pageviews"] = int(float(row.get("pageviews") or 0))
            by_article[row["article_id"]].append(row)

    final_stale = 0
    final_n = 0
    final_demand_stale = 0
    final_demand = 0
    event_stale = 0
    event_n = 0

    final_rows: list[dict] = []
    for aid, rows in by_article.items():
        rows.sort(key=lambda r: (r["timestamp"], r["revision_id"]))
        anchor_label = rows[0]["anchor_label"]
        last = rows[-1]
        stale = int(last["fresh_label"] != anchor_label)
        final_stale += stale
        final_n += 1
        demand = int(last.get("pageviews", 0))
        final_demand += demand
        final_demand_stale += demand * stale
        final_rows.append({
            "article_id": aid,
            "title": last["title"],
            "anchor_label": anchor_label,
            "final_fresh_label": last["fresh_label"],
            "final_stale": stale,
            "pageviews": demand,
        })
        for r in rows[1:]:
            if r["lead_changed"]:
                event_n += 1
                event_stale += int(r["fresh_label"] != anchor_label)

    result = {
        "articles": final_n,
        "final_state_stale_rate": final_stale / max(1, final_n),
        "pageview_weighted_final_state_stale_rate": final_demand_stale / max(1, final_demand),
        "lead_change_event_weighted_stale_rate": event_stale / max(1, event_n),
        "lead_change_states": event_n,
        "interpretation": (
            "Diagnostic Test 0 only. Phase 0A is required for statistically valid current-state "
            "population certification."
        ),
    }
    (root / "reports/test0_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (root / "reports/test0_final_states.csv").open("w", newline="", encoding="utf-8") as f:
        if final_rows:
            w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys()))
            w.writeheader(); w.writerows(final_rows)
    return result
