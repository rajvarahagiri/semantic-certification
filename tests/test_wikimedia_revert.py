from semcert.schemas import RevisionRecord
from semcert.wikimedia import WikimediaClient


def r(i, sha, tags=None):
    return RevisionRecord(None, "A", i, i-1 if i>1 else None, f"2026-08-0{i}T00:00:00Z", "u", None, None, tags or [], sha, "")


def test_repeated_sha_marks_revert():
    rows = [r(1, "a"), r(2, "b"), r(3, "a")]
    WikimediaClient._mark_reverts(rows)
    assert rows[2].is_revert
