from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class RevisionRecord:
    article_id: int | None
    title: str
    revision_id: int
    parent_id: int | None
    timestamp: str
    user: str | None
    user_id: int | None
    comment: str | None
    tags: list[str]
    sha1: str | None
    raw_wikitext: str
    is_bot: bool = False
    is_revert: bool = False
    normalized_lead: str = ""
    lead_hash: str = ""
    lead_changed: bool = False
    pipeline_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d
