from __future__ import annotations

import time
from collections import deque
from datetime import datetime
from typing import Iterable
from urllib.parse import quote

import requests

from .config import Config, ensure_user_agent
from .schemas import RevisionRecord


class WikimediaClient:
    def __init__(self, cfg: Config):
        ensure_user_agent(cfg)
        w = cfg.wikimedia
        self.api_url = w["api_url"]
        self.analytics_url = w["analytics_url"].rstrip("/")
        self.project = w["project"]
        self.timeout = int(w.get("request_timeout_seconds", 30))
        self.pause = float(w.get("pause_seconds", 0.1))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": w["user_agent"], "Accept": "application/json"})

    def _get(self, url: str, params: dict | None = None) -> dict:
        for attempt in range(5):
            r = self.session.get(url, params=params, timeout=self.timeout)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(2**attempt, 16))
                continue
            r.raise_for_status()
            if self.pause:
                time.sleep(self.pause)
            return r.json()
        raise RuntimeError(f"Repeated Wikimedia request failure: {url}")

    def category_articles(self, root_category: str, *, max_depth: int, limit: int) -> list[dict]:
        queue = deque([(root_category, 0)])
        seen_cats = set()
        seen_pages: dict[int, dict] = {}

        while queue and len(seen_pages) < limit:
            category, depth = queue.popleft()
            if category in seen_cats or depth > max_depth:
                continue
            seen_cats.add(category)
            cont = None
            while len(seen_pages) < limit:
                params = {
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmtitle": category,
                    "cmtype": "page|subcat",
                    "cmlimit": "max",
                }
                if cont:
                    params["cmcontinue"] = cont
                data = self._get(self.api_url, params)
                for m in data.get("query", {}).get("categorymembers", []):
                    ns = int(m.get("ns", -1))
                    title = m.get("title", "")
                    if ns == 0:
                        seen_pages[int(m["pageid"])] = {"article_id": int(m["pageid"]), "title": title}
                    elif ns == 14 and depth < max_depth:
                        queue.append((title, depth + 1))
                    if len(seen_pages) >= limit:
                        break
                cont = data.get("continue", {}).get("cmcontinue")
                if not cont:
                    break
        return list(seen_pages.values())[:limit]

    def _revision_query(self, title: str, *, start: str, end: str, direction: str, limit: int | None = None) -> list[RevisionRecord]:
        out: list[RevisionRecord] = []
        rvcontinue = None
        while True:
            params = {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "prop": "revisions",
                "titles": title,
                "rvprop": "ids|timestamp|user|userid|comment|tags|sha1|content",
                "rvslots": "main",
                "rvlimit": str(min(50, limit)) if limit else "50",
                "rvdir": direction,
                "rvstart": start,
                "rvend": end,
            }
            if rvcontinue:
                params["rvcontinue"] = rvcontinue
            data = self._get(self.api_url, params)
            pages = data.get("query", {}).get("pages", [])
            if not pages:
                break
            page = pages[0]
            article_id = page.get("pageid")
            canonical_title = page.get("title", title)
            for rev in page.get("revisions", []) or []:
                slot = rev.get("slots", {}).get("main", {})
                content = slot.get("content", "")
                user = rev.get("user")
                tags = list(rev.get("tags", []) or [])
                out.append(
                    RevisionRecord(
                        article_id=int(article_id) if article_id is not None else None,
                        title=canonical_title,
                        revision_id=int(rev["revid"]),
                        parent_id=int(rev["parentid"]) if rev.get("parentid") is not None else None,
                        timestamp=rev["timestamp"],
                        user=user,
                        user_id=int(rev["userid"]) if rev.get("userid") is not None else None,
                        comment=rev.get("comment"),
                        tags=tags,
                        sha1=rev.get("sha1"),
                        raw_wikitext=content,
                        is_bot=self._looks_like_bot(user, tags),
                    )
                )
            if limit is not None and len(out) >= limit:
                return out[:limit]
            rvcontinue = data.get("continue", {}).get("rvcontinue")
            if not rvcontinue:
                break
        return out

    @staticmethod
    def _looks_like_bot(user: str | None, tags: list[str]) -> bool:
        if user and user.lower().endswith("bot"):
            return True
        lowered = {t.lower() for t in tags}
        return any("bot" in t for t in lowered)

    def revisions_with_anchor(self, title: str, start: str, end: str) -> list[RevisionRecord]:
        # One revision at/before start, then all revisions in the experiment window.
        anchor = self._revision_query(title, start=start, end="2001-01-01T00:00:00Z", direction="older", limit=1)
        window = self._revision_query(title, start=start, end=end, direction="newer")
        rows = anchor + window
        # Avoid double-counting if a revision timestamp equals the experiment start.
        dedup = {r.revision_id: r for r in rows}
        rows = list(dedup.values())
        rows.sort(key=lambda r: (r.timestamp, r.revision_id))
        self._mark_reverts(rows)
        return rows

    @staticmethod
    def _mark_reverts(rows: list[RevisionRecord]) -> None:
        seen_sha: dict[str, int] = {}
        for idx, row in enumerate(rows):
            tagset = {t.lower() for t in row.tags}
            tagged = any(
                x in tagset
                for x in {"mw-rollback", "mw-undo", "mw-manual-revert", "mw-reverted"}
            )
            repeated_sha = bool(row.sha1 and row.sha1 in seen_sha and seen_sha[row.sha1] < idx - 1)
            row.is_revert = tagged or repeated_sha
            if row.sha1:
                seen_sha[row.sha1] = idx

    def pageviews(self, title: str, start_yyyymmdd: str, end_yyyymmdd: str) -> list[dict]:
        encoded = quote(title.replace(" ", "_"), safe="")
        url = (
            f"{self.analytics_url}/metrics/pageviews/per-article/{self.project}/"
            f"all-access/all-agents/{encoded}/daily/{start_yyyymmdd}/{end_yyyymmdd}"
        )
        data = self._get(url)
        return [
            {"title": title, "timestamp": i["timestamp"], "views": int(i["views"])}
            for i in data.get("items", [])
        ]
