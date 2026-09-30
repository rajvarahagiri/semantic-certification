from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from datetime import datetime
import json

from .config import Config
from .extract import extract_lead
from .io import write_jsonl, write_csv
from .wikimedia import WikimediaClient


def _ymd(iso_ts: str) -> str:
    return datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).strftime("%Y%m%d")


def collect_probe(cfg: Config, out_dir: str | Path, limit: int | None = None) -> dict:
    out = Path(out_dir)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = WikimediaClient(cfg)
    limit = int(limit or cfg.experiment["probe_articles"])
    articles = client.category_articles(
        cfg.wikimedia["root_category"],
        max_depth=int(cfg.wikimedia["category_depth"]),
        limit=limit,
    )
    write_csv(raw_dir / "articles.csv", articles)

    rev_rows: list[dict] = []
    pv_rows: list[dict] = []
    failures: list[dict] = []
    start = cfg.experiment["full_start"]
    end = cfg.experiment["full_end"]
    start_day, end_day = _ymd(start), _ymd(end)

    for idx, article in enumerate(articles, start=1):
        title = article["title"]
        try:
            revisions = client.revisions_with_anchor(title, start, end)
            prior_hash = None
            for r in revisions:
                lead = extract_lead(r.raw_wikitext, cfg.extraction)
                r.normalized_lead = lead.text
                r.lead_hash = lead.hash
                r.pipeline_version = lead.pipeline_version
                r.lead_changed = prior_hash is not None and lead.hash != prior_hash
                prior_hash = lead.hash
                d = r.to_dict()
                d.pop("raw_wikitext", None)  # private dataset stores only normalized model input by default
                rev_rows.append(d)
            pv_rows.extend(client.pageviews(title, start_day, end_day))
        except Exception as e:
            failures.append({"title": title, "error": repr(e)})
        if idx % 25 == 0:
            write_jsonl(raw_dir / "revisions.jsonl", rev_rows)
            write_csv(raw_dir / "pageviews.csv", pv_rows)
            write_csv(raw_dir / "failures.csv", failures)

    write_jsonl(raw_dir / "revisions.jsonl", rev_rows)
    write_csv(raw_dir / "pageviews.csv", pv_rows)
    write_csv(raw_dir / "failures.csv", failures)
    summary = {
        "articles_requested": limit,
        "articles_collected": len({r["article_id"] for r in rev_rows}),
        "revision_rows": len(rev_rows),
        "pageview_rows": len(pv_rows),
        "failures": len(failures),
    }
    (raw_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
