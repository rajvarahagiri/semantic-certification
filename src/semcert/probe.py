from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import random

import numpy as np

from .config import Config
from .io import read_jsonl, write_csv
from .models import BGEEmbedder, MockEmbedder, MockLabeler, ModelLock, QwenBinaryLabeler
from .scoring import cosine_distance


def _load_pageviews(path: Path) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    if not path.exists() or path.stat().st_size == 0:
        return totals
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            totals[row["title"]] += int(row.get("views") or 0)
    return totals


def _make_models(cfg: Config, lock_path: Path, mock: bool):
    if mock:
        return MockLabeler(), MockEmbedder()
    if not lock_path.exists():
        raise RuntimeError("Missing artifacts/model_lock.json. Run: semcert pin-models")
    lock = ModelLock.load(lock_path)
    full_cfg = cfg.models["full_model"]
    gate_cfg = cfg.models["gate_model"]
    labeler = QwenBinaryLabeler(
        lock.full_model_repo,
        lock.full_model_revision,
        device=full_cfg.get("device", "auto"),
        dtype=full_cfg.get("dtype", "auto"),
        max_new_tokens=int(full_cfg.get("max_new_tokens", 4)),
    )
    embedder = BGEEmbedder(
        lock.gate_model_repo,
        lock.gate_model_revision,
        device=gate_cfg.get("device", "auto"),
    )
    return labeler, embedder


def run_probe(cfg: Config, project_dir: str | Path, *, mock: bool = False) -> dict:
    root = Path(project_dir)
    rows = list(read_jsonl(root / "data/raw/revisions.jsonl"))
    if not rows:
        raise RuntimeError("No revisions found. Run collect-probe first.")
    pageviews = _load_pageviews(root / "data/raw/pageviews.csv")
    labeler, embedder = _make_models(cfg, root / "artifacts/model_lock.json", mock)

    include_bots = bool(cfg.probe.get("include_bots_primary", False))
    by_article: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        if not include_bots and r.get("is_bot"):
            continue
        by_article[int(r["article_id"])].append(r)
    for group in by_article.values():
        group.sort(key=lambda x: (x["timestamp"], x["revision_id"]))

    detailed: list[dict] = []
    anchor_labels: list[int] = []
    all_labels: list[int] = []
    total_revisions = 0
    changed_revisions = 0
    flips = 0
    comparable_changes = 0

    for aid, group in by_article.items():
        if not group:
            continue
        # First row is the anchor or earliest row collected.
        anchor = group[0]
        anchor_label = labeler.label(anchor["normalized_lead"])
        anchor_vec = embedder.embed(anchor["normalized_lead"])
        anchor_labels.append(anchor_label)
        prev_label = anchor_label
        prev_hash = anchor["lead_hash"]
        all_labels.append(anchor_label)
        detailed.append({
            "article_id": aid,
            "title": anchor["title"],
            "revision_id": anchor["revision_id"],
            "timestamp": anchor["timestamp"],
            "lead_changed": False,
            "fresh_label": anchor_label,
            "anchor_label": anchor_label,
            "anchor_distance": 0.0,
            "label_flip": False,
            "is_revert": anchor.get("is_revert", False),
            "pageviews": pageviews.get(anchor["title"], 0),
        })
        total_revisions += 1

        for r in group[1:]:
            total_revisions += 1
            changed = r["lead_hash"] != prev_hash
            prev_hash = r["lead_hash"]
            if not changed:
                continue
            changed_revisions += 1
            fresh = labeler.label(r["normalized_lead"])
            vec = embedder.embed(r["normalized_lead"])
            score = cosine_distance(anchor_vec, vec)
            flip = fresh != prev_label
            flips += int(flip)
            comparable_changes += 1
            all_labels.append(fresh)
            detailed.append({
                "article_id": aid,
                "title": r["title"],
                "revision_id": r["revision_id"],
                "timestamp": r["timestamp"],
                "lead_changed": True,
                "fresh_label": fresh,
                "anchor_label": anchor_label,
                "anchor_distance": score,
                "label_flip": flip,
                "is_revert": r.get("is_revert", False),
                "pageviews": pageviews.get(r["title"], 0),
            })
            prev_label = fresh

    noise_n = min(int(cfg.probe["noise_floor_examples"]), len(detailed))
    rng = random.Random(int(cfg.probe["seed"]))
    sample = rng.sample(detailed, noise_n) if noise_n else []
    disagreements = 0
    for x in sample:
        lead = next(
            r["normalized_lead"]
            for r in by_article[int(x["article_id"])]
            if int(r["revision_id"]) == int(x["revision_id"])
        )
        disagreements += int(labeler.label(lead) != labeler.label(lead))

    base_rate_anchor = float(np.mean(anchor_labels)) if anchor_labels else math.nan
    base_rate_all = float(np.mean(all_labels)) if all_labels else math.nan
    lead_change_rate = changed_revisions / max(1, total_revisions - len(by_article))
    flip_rate = flips / max(1, comparable_changes)
    noise_rate = disagreements / max(1, noise_n)
    never_recompute_event_stale = (
        sum(int(int(x["fresh_label"]) != int(x["anchor_label"])) for x in detailed if x["lead_changed"])
        / max(1, sum(int(x["lead_changed"]) for x in detailed))
    )

    exp = cfg.experiment
    go = (
        float(exp["base_rate_min"]) <= base_rate_anchor <= float(exp["base_rate_max"])
        and flip_rate >= float(exp["min_flip_rate_given_lead_change"])
        and noise_rate <= float(exp["epsilon"])
    )
    summary = {
        "mock_run": mock,
        "articles": len(by_article),
        "total_revision_rows": total_revisions,
        "lead_changing_revisions": changed_revisions,
        "lead_change_rate": lead_change_rate,
        "anchor_positive_base_rate": base_rate_anchor,
        "all_evaluated_positive_rate": base_rate_all,
        "label_flip_rate_given_lead_change": flip_rate,
        "noise_floor_n": noise_n,
        "noise_floor_disagreement_rate": noise_rate,
        "never_recompute_event_weighted_stale_rate": never_recompute_event_stale,
        "epsilon": float(exp["epsilon"]),
        "probe_go": go,
        "go_checks": {
            "base_rate_in_range": float(exp["base_rate_min"]) <= base_rate_anchor <= float(exp["base_rate_max"]),
            "flip_rate_sufficient": flip_rate >= float(exp["min_flip_rate_given_lead_change"]),
            "noise_floor_at_or_below_epsilon": noise_rate <= float(exp["epsilon"]),
        },
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_csv(reports / "probe_revision_labels.csv", detailed)
    (reports / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
