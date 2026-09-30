import csv
from pathlib import Path

from semcert.config import load_config
from semcert.io import write_jsonl
from semcert.probe import run_probe


def test_probe_mock(tmp_path: Path):
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    rows = [
        {"article_id": 1, "title": "A", "revision_id": 1, "timestamp": "2026-08-02T00:00:00Z", "is_bot": False, "is_revert": False, "normalized_lead": "Acme is a company [REF].", "lead_hash": "h1"},
        {"article_id": 1, "title": "A", "revision_id": 2, "timestamp": "2026-08-03T00:00:00Z", "is_bot": False, "is_revert": False, "normalized_lead": "Acme is the largest company.", "lead_hash": "h2"},
        {"article_id": 2, "title": "B", "revision_id": 3, "timestamp": "2026-08-02T00:00:00Z", "is_bot": False, "is_revert": False, "normalized_lead": "B is documented [REF].", "lead_hash": "h3"},
        {"article_id": 2, "title": "B", "revision_id": 4, "timestamp": "2026-08-04T00:00:00Z", "is_bot": False, "is_revert": False, "normalized_lead": "B is documented [REF].", "lead_hash": "h3"},
    ]
    write_jsonl(tmp_path / "data/raw/revisions.jsonl", rows)
    with (tmp_path / "data/raw/pageviews.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "timestamp", "views"])
        w.writeheader(); w.writerow({"title": "A", "timestamp": "2026080300", "views": 10}); w.writerow({"title": "B", "timestamp": "2026080300", "views": 5})
    cfg_path = Path(__file__).parents[1] / "config/phase0.yaml"
    cfg = load_config(cfg_path)
    result = run_probe(cfg, tmp_path, mock=True)
    assert result["articles"] == 2
    assert result["lead_changing_revisions"] == 1
    assert result["label_flip_rate_given_lead_change"] == 1.0


def test_test0_after_probe(tmp_path: Path):
    from semcert.test0 import run_test0
    (tmp_path / "data/raw").mkdir(parents=True)
    rows = [
        {"article_id": 1, "title": "A", "revision_id": 1, "timestamp": "2026-08-02T00:00:00Z", "is_bot": False, "is_revert": False, "normalized_lead": "Acme is a company [REF].", "lead_hash": "h1"},
        {"article_id": 1, "title": "A", "revision_id": 2, "timestamp": "2026-08-03T00:00:00Z", "is_bot": False, "is_revert": False, "normalized_lead": "Acme is the largest company.", "lead_hash": "h2"},
    ]
    write_jsonl(tmp_path / "data/raw/revisions.jsonl", rows)
    with (tmp_path / "data/raw/pageviews.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "timestamp", "views"]); w.writeheader(); w.writerow({"title":"A","timestamp":"2026080300","views":10})
    cfg_path = Path(__file__).parents[1] / "config/phase0.yaml"
    run_probe(load_config(cfg_path), tmp_path, mock=True)
    result = run_test0(tmp_path)
    assert result["final_state_stale_rate"] == 1.0
