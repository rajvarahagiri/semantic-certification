from semcert.extract import normalize_lead, content_hash


def test_extracts_only_lead_and_preserves_reference_markers():
    w = """{{Short description|Example}}
'''Acme''' is a company.<ref name=one>Source</ref> It is the largest widget maker.{{citation needed|date=Jan 2026}}

== History ==
Body text should not appear.
"""
    out = normalize_lead(w)
    assert "Acme is a company." in out
    assert "[REF]" in out
    assert "[CITATION_NEEDED]" in out
    assert "Body text" not in out


def test_hash_changes_with_pipeline_version():
    assert content_hash("x", "v1") != content_hash("x", "v2")
