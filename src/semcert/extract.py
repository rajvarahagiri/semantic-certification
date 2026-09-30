from __future__ import annotations

import hashlib
import html
import re
from dataclasses import dataclass


_HEADING_RE = re.compile(r"^==[^=].*?==\s*$", re.MULTILINE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_BLOCK_RE = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
_REF_SELF_RE = re.compile(r"<ref\b[^>]*/\s*>", re.IGNORECASE)
_CITATION_NEEDED_RE = re.compile(
    r"\{\{\s*(citation needed|cn|fact|citeneeded)\b.*?\}\}", re.IGNORECASE | re.DOTALL
)
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_FILE_RE = re.compile(r"\[\[(File|Image):.*?\]\]", re.IGNORECASE | re.DOTALL)
_CATEGORY_RE = re.compile(r"\[\[Category:.*?\]\]", re.IGNORECASE)
_EXTERNAL_LINK_RE = re.compile(r"\[(https?://\S+)\s+([^\]]+)\]")
_EXTERNAL_BARE_RE = re.compile(r"\[https?://[^\]]+\]")
_WIKILINK_WITH_TEXT_RE = re.compile(r"\[\[[^\]|]+\|([^\]]+)\]\]")
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_BOLD_ITALIC_RE = re.compile(r"'{2,5}")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{3,}")


def _remove_templates(text: str) -> str:
    """Remove balanced-ish {{...}} blocks, preserving markers substituted earlier.

    This intentionally favors reproducibility over perfect MediaWiki rendering. Nested
    templates are removed iteratively from innermost to outermost.
    """
    prior = None
    out = text
    simple = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
    while prior != out:
        prior = out
        out = simple.sub(" ", out)
    return out


def lead_wikitext(wikitext: str) -> str:
    m = _HEADING_RE.search(wikitext)
    return wikitext[: m.start()] if m else wikitext


def normalize_lead(
    wikitext: str,
    *,
    ref_marker: str = "[REF]",
    citation_needed_marker: str = "[CITATION_NEEDED]",
    max_chars: int = 24000,
) -> str:
    text = lead_wikitext(wikitext or "")
    text = _COMMENT_RE.sub(" ", text)
    text = _CITATION_NEEDED_RE.sub(f" {citation_needed_marker} ", text)
    text = _REF_BLOCK_RE.sub(f" {ref_marker} ", text)
    text = _REF_SELF_RE.sub(f" {ref_marker} ", text)
    text = _TABLE_RE.sub(" ", text)
    text = _FILE_RE.sub(" ", text)
    text = _CATEGORY_RE.sub(" ", text)
    text = _remove_templates(text)
    text = _EXTERNAL_LINK_RE.sub(r"\2", text)
    text = _EXTERNAL_BARE_RE.sub(" ", text)
    text = _WIKILINK_WITH_TEXT_RE.sub(r"\1", text)
    text = _WIKILINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _BOLD_ITALIC_RE.sub("", text)
    text = html.unescape(text)
    lines = []
    for line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    text = "\n".join(lines)
    text = _BLANKS_RE.sub("\n\n", text).strip()
    return text[:max_chars]


def content_hash(text: str, pipeline_version: str) -> str:
    payload = f"{pipeline_version}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ExtractedLead:
    text: str
    hash: str
    pipeline_version: str


def extract_lead(wikitext: str, cfg: dict) -> ExtractedLead:
    version = str(cfg["pipeline_version"])
    text = normalize_lead(
        wikitext,
        ref_marker=str(cfg["preserve_ref_marker"]),
        citation_needed_marker=str(cfg["preserve_citation_needed_marker"]),
        max_chars=int(cfg.get("max_chars", 24000)),
    )
    return ExtractedLead(text=text, hash=content_hash(text, version), pipeline_version=version)
