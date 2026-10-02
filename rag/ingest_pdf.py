"""PDF -> cleaned, section-aware chunks. Pure functions; no DB / model code."""
from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

try:
    import pymupdf
except ImportError:  # older PyMuPDF releases
    import fitz as pymupdf

logger = logging.getLogger("rag.ingest")


@dataclass
class PipelineConfig:
    max_chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_size: int = 200
    header_footer_zone_lines: int = 3
    header_footer_min_page_ratio: float = 0.35
    header_footer_min_pages_for_detection: int = 3
    toc_dotted_line_min_ratio: float = 0.4
    toc_max_scan_pages: int = 15
    duplicate_similarity_window: int = 3
    heading_max_words: int = 14
    numbered_heading_pattern: str = r"^(\d{1,2}(\.\d{1,2}){0,3})\.?\s+\S.*"
    min_chars_per_page_for_text_pdf: int = 20


DEFAULT_CONFIG = PipelineConfig()


@dataclass
class Page:
    page_number: int
    text: str
    source: str
    total_pages: int
    is_empty: bool = False
    lines: list[str] = field(default_factory=list)


@dataclass
class Section:
    title: str
    level: int
    page_start: int
    page_end: int
    parent_title: str | None
    paragraphs: list[str] = field(default_factory=list)
    para_pages: list[int] = field(default_factory=list)  # page of each paragraph


# ------------------------------------------------------------------ loading
def load_pdf(pdf_path: str) -> list[Page]:
    if not isinstance(pdf_path, str) or not pdf_path.strip():
        raise ValueError("pdf_path must be a non-empty string.")
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"Not a PDF file: {pdf_path}")
    try:
        with pymupdf.open(pdf_path) as doc:
            texts = [page.get_text("text") or "" for page in doc]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse PDF '{pdf_path}': {exc}") from exc
    if not texts:
        raise ValueError(f"No pages found in: {pdf_path}")

    source = os.path.basename(pdf_path)
    pages = []
    for i, text in enumerate(texts, start=1):
        if not text.strip():
            logger.warning("Page %d of %s is empty or unreadable.", i, source)
        pages.append(Page(i, text, source, len(texts), is_empty=not text.strip()))
    logger.info("Loaded %d pages from %s", len(pages), source)
    return pages


def check_needs_ocr(pages: list[Page], config: PipelineConfig = DEFAULT_CONFIG) -> bool:
    if not pages:
        return False
    thin = sum(1 for p in pages if len(p.text.strip()) < config.min_chars_per_page_for_text_pdf)
    ratio = thin / len(pages)
    if ratio >= 0.8:
        logger.warning("%.0f%% of pages have no text: PDF looks scanned (needs OCR).", ratio * 100)
        return True
    return False


# ------------------------------------------------------------ TOC detection
_TOC_LINE_RE = re.compile(
    r"^(?P<title>.{2,80}?)[\s.\-·•]{2,}(?P<page>\d{1,4})\s*$"
)
_TOC_TITLE_WORDS = ("contents", "table of contents", "index", "list of figures", "list of tables", "toc")


def detect_toc_pages(pages: list[Page], config: PipelineConfig = DEFAULT_CONFIG) -> set[int]:
    toc: set[int] = set()
    for page in pages[: config.toc_max_scan_pages]:
        lines = [ln.strip() for ln in page.text.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        dotted = sum(1 for ln in lines if _TOC_LINE_RE.match(ln)) / len(lines)
        first = lines[0].lower().strip(" :")
        title_hit = any(first.startswith(w) for w in _TOC_TITLE_WORDS)
        num_end = sum(1 for ln in lines if re.search(r"\d{1,4}\s*$", ln)) / len(lines)
        if dotted >= config.toc_dotted_line_min_ratio or (title_hit and num_end >= 0.3):
            toc.add(page.page_number)
            logger.info("Detected TOC page %d", page.page_number)
    return toc


# --------------------------------------------------- header / footer detection
def _normalize_for_comparison(line: str) -> str:
    norm = re.sub(r"\d+", "#", line.strip().lower())
    return re.sub(r"\s+", " ", norm)


def detect_headers_footers(
    pages: list[Page], config: PipelineConfig = DEFAULT_CONFIG
) -> tuple[set[str], set[str]]:
    if len(pages) < config.header_footer_min_pages_for_detection:
        return set(), set()
    n = config.header_footer_zone_lines
    head, foot = Counter(), Counter()
    live = [p for p in pages if not p.is_empty]
    if not live:
        return set(), set()
    for p in live:
        lines = [ln for ln in p.text.splitlines() if ln.strip()]
        for t in {_normalize_for_comparison(x) for x in lines[:n]}:
            if t:
                head[t] += 1
        for t in {_normalize_for_comparison(x) for x in lines[-n:]}:
            if t:
                foot[t] += 1
    min_count = max(2, int(len(live) * config.header_footer_min_page_ratio))
    h = {t for t, c in head.items() if c >= min_count}
    f = {t for t, c in foot.items() if c >= min_count}
    logger.info("Detected %d header and %d footer template(s).", len(h), len(f))
    return h, f


# ------------------------------------------------------------ noise detection
_PURE_PAGE_NUM_RE = re.compile(r"^\d{1,4}$")
_PAGE_LABEL_RE = re.compile(
    r"^(page\s+\d{1,4}(\s+of\s+\d{1,4})?|p\.?\s*\d{1,4}|-\s*\d{1,4}\s*-)$", re.IGNORECASE
)


def _is_page_number_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and bool(_PURE_PAGE_NUM_RE.match(s) or _PAGE_LABEL_RE.match(s))


def _is_isolated_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if len(s) == 1 and not s.isalnum():
        return True
    return bool(re.fullmatch(r"[_\-=~.]{3,}", s))


# ------------------------------------------------------------------ headings
_BULLET_RE = re.compile(r"^\s*([•\-\*\u2022\u25CF\u25AA]|\d+[\.\)])\s+")
_HEADING_NUMBERED_RE = re.compile(DEFAULT_CONFIG.numbered_heading_pattern)
# "Part of the ..." must NOT be a heading, so require a number / roman numeral / letter.
_HEADING_KEYWORDS_RE = re.compile(r"^(?i:chapter|section|part|appendix)\s+(?:\d+|[IVXLC]+|[A-Z])\b")


def _classify_heading_level(line: str) -> int | None:
    s = line.strip()
    if not s or len(s) > 120:
        return None
    words = s.split()
    sentence_end = s.endswith((".", ",", ";", ":")) and not s.endswith("...")
    short = len(words) <= DEFAULT_CONFIG.heading_max_words

    if short and not sentence_end and _HEADING_KEYWORDS_RE.match(s):
        return 1
    m = _HEADING_NUMBERED_RE.match(s)
    if m and short and not sentence_end:
        return min(m.group(1).count(".") + 1, 3)
    if s.isupper() and 1 <= len(words) <= 12:
        return 1
    if (
        not sentence_end
        and short
        and s[:1].isupper()
        and sum(1 for w in words if w[:1].isupper()) / len(words) >= 0.6
    ):
        return 2
    return None


def _is_structural_line(line: str) -> bool:
    s = line.strip()
    return bool(s) and (bool(_BULLET_RE.match(s)) or _classify_heading_level(s) is not None)


def _rejoin_wrapped_paragraphs(lines: list[str]) -> list[str]:
    merged: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            merged.append(" ".join(buf).strip())
            buf.clear()

    for line in lines:
        s = line.strip()
        if not s:
            flush()
            merged.append("")
        elif _is_structural_line(s):
            flush()
            merged.append(s)
        else:
            buf.append(s)
    flush()
    return merged


# ------------------------------------------------------------------ cleaning
def clean_pages(
    pages: list[Page],
    toc_pages: set[int],
    header_templates: set[str],
    footer_templates: set[str],
    config: PipelineConfig = DEFAULT_CONFIG,
) -> list[Page]:
    out: list[Page] = []
    n = config.header_footer_zone_lines
    for page in pages:
        if page.page_number in toc_pages:
            continue
        if page.is_empty:
            out.append(page)
            continue

        raw = page.text.splitlines()
        non_empty = [i for i, ln in enumerate(raw) if ln.strip()]
        zone = set(non_empty[:n]) | set(non_empty[-n:])  # only strip inside header/footer zones

        kept: list[str] = []
        for i, line in enumerate(raw):
            s = line.strip()
            if not s:
                kept.append("")
                continue
            norm = _normalize_for_comparison(s)
            if i in zone and (norm in header_templates or norm in footer_templates):
                continue
            if _is_page_number_line(s) or _is_isolated_noise_line(s):
                continue
            kept.append(s)

        collapsed: list[str] = []
        prev_blank = True
        for p in _rejoin_wrapped_paragraphs(kept):
            if p == "":
                if not prev_blank:
                    collapsed.append("")
                prev_blank = True
            else:
                collapsed.append(re.sub(r"\s{2,}", " ", p))
                prev_blank = False
        while collapsed and collapsed[-1] == "":
            collapsed.pop()

        page.text = "\n".join(collapsed)
        page.lines = [c for c in collapsed if c]
        page.is_empty = not page.text.strip()
        out.append(page)
    logger.info("Cleaned %d pages (dropped %d TOC pages).", len(out), len(toc_pages))
    return out


# ---------------------------------------------------------------- duplicates
def _fingerprint(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text.strip().lower()).encode("utf-8")).hexdigest()


def remove_duplicates(pages: list[Page], config: PipelineConfig = DEFAULT_CONFIG) -> list[Page]:
    recent: list[str] = []
    window = config.duplicate_similarity_window
    for page in pages:
        if page.is_empty:
            continue
        new_lines: list[str] = []
        last_fp = None
        for line in page.text.splitlines():
            if not line.strip():
                new_lines.append(line)
                continue
            fp = _fingerprint(line)
            if fp == last_fp:
                continue  # immediate repeat
            # cross-window dedupe only for longer lines, so short repeated labels survive
            if len(line) >= 20 and fp in recent[-window:]:
                continue
            new_lines.append(line)
            last_fp = fp
            recent.append(fp)
            recent = recent[-500:]
        page.text = "\n".join(new_lines)
        page.lines = [ln for ln in new_lines if ln.strip()]
    return pages


# ------------------------------------------------------------------ sections
def detect_sections(pages: list[Page], config: PipelineConfig = DEFAULT_CONFIG) -> list[Section]:
    sections: list[Section] = []
    stack: dict[int, str] = {}
    current: Section | None = None

    def start(title: str, level: int, pg: int) -> Section:
        sec = Section(title, level, pg, pg, stack.get(level - 1))
        sections.append(sec)
        stack[level] = title
        for deeper in [lv for lv in stack if lv > level]:
            del stack[deeper]
        return sec

    for page in pages:
        if page.is_empty:
            continue
        for line in page.text.splitlines():
            s = line.strip()
            if not s:
                continue
            level = _classify_heading_level(s)
            if level is not None:
                current = start(s, level, page.page_number)
            else:
                if current is None:
                    current = start("Document Start", 1, page.page_number)
                current.paragraphs.append(s)
                current.para_pages.append(page.page_number)
                current.page_end = page.page_number
    logger.info("Detected %d section(s).", len(sections))
    return sections


# ------------------------------------------------------------------ chunking
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u2018\u201C])")


def _split_into_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """Last resort for a single 'sentence' longer than the chunk budget."""
    if len(text) <= size:
        return [text]
    out, cur = [], ""
    for w in text.split():
        while len(w) > size:
            if cur:
                out.append(cur)
                cur = ""
            out.append(w[:size])
            w = w[size:]
        cand = f"{cur} {w}".strip()
        if len(cand) > size:
            out.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def _overlap_tail(text: str, n: int) -> str:
    """Last ~n chars, starting on a word boundary (never mid-word)."""
    if n <= 0 or len(text) <= n:
        return text if n > 0 else ""
    tail = text[-n:]
    i = tail.find(" ")
    return tail[i + 1:] if i != -1 else tail


def _names(sec: Section) -> tuple[str, str | None]:
    if sec.parent_title and sec.parent_title != sec.title:
        return sec.parent_title, sec.title
    return sec.title, None


def semantic_chunking(sections: list[Section], config: PipelineConfig = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    max_size = config.max_chunk_size
    raw: list[dict[str, Any]] = []

    for sec in sections:
        if not sec.paragraphs:
            continue
        section, subsection = _names(sec)
        crumb = f"Section: {section}" + (f"\nSubsection: {subsection}" if subsection else "")

        def emit(body: str, pages: list[int]) -> None:
            body = body.strip()
            if body and pages:
                raw.append({
                    "text": f"{crumb}\n\n{body}", "body_len": len(body),
                    "section": section, "subsection": subsection,
                    "page_start": min(pages), "page_end": max(pages),
                })

        units: list[tuple[str, int, bool]] = []  # (text, page, starts_new_paragraph)
        for para, pg in zip(sec.paragraphs, sec.para_pages):
            if len(para) <= max_size:
                units.append((para, pg, True))
                continue
            first = True
            for sent in _split_into_sentences(para):
                for piece in _hard_split(sent, max_size):
                    units.append((piece, pg, first))
                    first = False

        cur, pages = "", []
        for text, pg, new_para in units:
            cand = (cur + ("\n" if new_para else " ") + text) if cur else text
            if cur and len(cand) > max_size:
                emit(cur, pages)
                tail = _overlap_tail(cur, config.chunk_overlap)
                cur = f"{tail} {text}" if tail else text
                pages = [pg]
            else:
                cur = cand
                pages.append(pg)
        emit(cur, pages)

    merged: list[dict[str, Any]] = []
    for ch in raw:
        if (merged and ch["body_len"] < config.min_chunk_size
                and len(merged[-1]["text"]) + len(ch["text"]) <= max_size * 1.5):
            merged[-1]["text"] += "\n\n" + ch["text"]
            merged[-1]["page_end"] = max(merged[-1]["page_end"], ch["page_end"])
        else:
            merged.append(ch)
    logger.info("Produced %d chunk(s) from %d section(s).", len(merged), len(sections))
    return merged


def build_rag_dictionary(document_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """chunk_id is a 0-based int (same convention as the original notebook's VECTOR_DB)."""
    formatted = [
        {
            "chunk_id": i,
            "text": c["text"],
            "char_count": len(c["text"]),
            "metadata": {
                "source": document_id,
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "section": c.get("section"),
                "subsection": c.get("subsection"),
            },
        }
        for i, c in enumerate(chunks)
    ]
    return {"document_id": document_id, "total_chunks": len(formatted), "chunks": formatted}


def process_pdf(
    pdf_path: str,
    max_chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    overrides = {k: v for k, v in
                 {"max_chunk_size": max_chunk_size, "chunk_overlap": chunk_overlap}.items() if v is not None}
    cfg = replace(config or PipelineConfig(), **overrides)  # never mutates caller's config
    if cfg.chunk_overlap >= cfg.max_chunk_size:
        raise ValueError("chunk_overlap must be smaller than max_chunk_size.")

    doc_id = os.path.basename(pdf_path)
    pages = load_pdf(pdf_path)
    if check_needs_ocr(pages, cfg):
        return {"document_id": doc_id, "needs_ocr": True, "total_chunks": 0, "chunks": []}

    toc = detect_toc_pages(pages, cfg)
    headers, footers = detect_headers_footers(pages, cfg)
    pages = clean_pages(pages, toc, headers, footers, cfg)
    pages = remove_duplicates(pages, cfg)
    sections = detect_sections(pages, cfg)
    result = build_rag_dictionary(doc_id, semantic_chunking(sections, cfg))
    logger.info("Pipeline complete: %d chunk(s) from %s.", result["total_chunks"], doc_id)
    return result
