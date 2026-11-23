"""Build rag_corpus.json from the Package-of-Practices PDFs.

Pages are split into prose and table chunks (table rows keep their headers), tagged with the
locations and region they mention, then deduplicated and filtered. Scanned pages fall back to OCR.

    python -m scripts.build_rag
"""
import os
import re
import glob
import json
import logging
from collections import Counter

import numpy as np
import pdfplumber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build_rag")

_WS_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Collapse whitespace and drop soft hyphens so each chunk is one clean line."""
    if not text:
        return ""
    text = text.replace("­", "")
    return _WS_RE.sub(" ", text).strip()

DEFAULT_PDF_DIR = os.path.join(ROOT, "Package_Of_Practices")
DEFAULT_OUTPUT = "rag_corpus.json"
DEFAULT_CHUNK_SIZE = 250   # words, for prose chunks
DEFAULT_OVERLAP = 40       # words, for prose chunks
DEFAULT_TABLE_ROWS_PER_CHUNK = 6  # data rows per table chunk (header repeated in each)
# small batches keep a table chunk from mixing rows from different districts

# locations used in the experiments, with spellings found in the PDFs (e.g. Dharwar)
LOCATION_ALIASES = {
    "Mandya": ["Mandya"],
    "Dharwad": ["Dharwad", "Dharwar"],
    "Davanagere": ["Davanagere", "Davangere", "Davanagara"],
    "Haveri": ["Haveri"],
}
_LOCATION_PATTERNS = {
    canonical: re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)
    for canonical, aliases in LOCATION_ALIASES.items()
}


def _detect_locations(text: str) -> list:
    """Known locations named in the chunk text, if any."""
    return [canonical for canonical, pattern in _LOCATION_PATTERNS.items() if pattern.search(text)]


# other Karnataka districts, used only for the coarser region tag
OTHER_KARNATAKA_DISTRICT_ALIASES = {
    "Belagavi": ["Belagavi", "Belgaum"],
    "Mysuru": ["Mysuru", "Mysore"],
    "Ballari": ["Ballari", "Bellary"],
    "Shivamogga": ["Shivamogga", "Shimoga"],
    "Chitradurga": ["Chitradurga"],
    "Hassan": ["Hassan"],
}
_ALL_KARNATAKA_TERMS = ["Karnataka"] + [
    alias
    for aliases in list(LOCATION_ALIASES.values()) + list(OTHER_KARNATAKA_DISTRICT_ALIASES.values())
    for alias in aliases
]
_KARNATAKA_REGION_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _ALL_KARNATAKA_TERMS) + r")\b", re.IGNORECASE
)

def _detect_region(text: str) -> str:
    """Return "Karnataka" if the chunk mentions the state or any of its districts, else None."""
    return "Karnataka" if _KARNATAKA_REGION_PATTERN.search(text) else None


def _not_within_bboxes(obj, bboxes):
    """pdfplumber recipe: True if a text object's center falls outside every
    given table bounding box, i.e. it belongs to the page's prose, not a table.
    """
    v_mid = (obj["top"] + obj["bottom"]) / 2
    h_mid = (obj["x0"] + obj["x1"]) / 2

    def in_bbox(bbox):
        x0, top, x1, bottom = bbox
        return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom)

    return not any(in_bbox(b) for b in bboxes)


def _is_degenerate_header(header: list, max_header_len: int = 120, dup_frac_threshold: float = 0.5) -> bool:
    """Detect multi-column prose that find_tables() mistook for a table (header cells repeat one passage)."""
    non_empty = [h for h in header if h]
    if not non_empty:
        return True
    if any(len(h) > max_header_len for h in non_empty):
        return True
    most_common_count = Counter(non_empty).most_common(1)[0][1]
    return (most_common_count / len(non_empty)) > dup_frac_threshold


def _render_table_rows(table_rows, source_file, page_num):
    """Render table rows as 'Header1: val1 | Header2: val2 | ...' chunks."""
    chunks = []
    if not table_rows or len(table_rows) < 2:
        return chunks

    header = [_clean_text(h) for h in table_rows[0]]
    if _is_degenerate_header(header):
        log.warning(f"{source_file} page {page_num}: table header looks like a misdetected "
                    f"prose/directory block (repeated or oversized cells) -- skipping rather "
                    f"than rendering it as garbage rows.")
        return chunks
    data_rows = table_rows[1:]

    rendered_rows = []
    for row in data_rows:
        cells = [_clean_text(c) for c in row]
        pairs = [f"{h}: {v}" for h, v in zip(header, cells) if h and v]
        if pairs:
            rendered_rows.append(" | ".join(pairs))

    for i in range(0, len(rendered_rows), DEFAULT_TABLE_ROWS_PER_CHUNK):
        batch = rendered_rows[i:i + DEFAULT_TABLE_ROWS_PER_CHUNK]
        text = "\n".join(batch)
        if len(text.strip()) > 50:
            chunks.append({
                "file": source_file, "page": page_num, "type": "table", "text": text,
            })
    return chunks


def _chunk_prose(paragraphs, source_file, page_num, chunk_size, overlap):
    """Word-window chunking that keeps whole paragraphs together where possible."""
    chunks = []
    buffer_words = []

    def flush():
        if len(buffer_words) == 0:
            return
        text = " ".join(buffer_words)
        if len(text.strip()) > 50:
            chunks.append({
                "file": source_file, "page": page_num, "type": "prose", "text": text,
            })

    for para in paragraphs:
        para_words = para.split()
        if not para_words:
            continue

        if len(para_words) > chunk_size:
            # oversized paragraph: flush the buffer and window it on its own
            flush()
            buffer_words = []
            step = max(chunk_size - overlap, 1)
            for i in range(0, len(para_words), step):
                window = para_words[i:i + chunk_size]
                text = " ".join(window)
                if len(text.strip()) > 50:
                    chunks.append({
                        "file": source_file, "page": page_num, "type": "prose", "text": text,
                    })
            continue

        if len(buffer_words) + len(para_words) > chunk_size:
            flush()
            # Carry the overlap tail forward so context isn't lost at the boundary.
            buffer_words = buffer_words[-overlap:] if overlap else []
        buffer_words.extend(para_words)

    flush()
    return chunks


_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_LABEL_PATTERN = re.compile(r"\b(?:phone|fax|mobile|ph\.?)\s*[:\-]?", re.IGNORECASE)
_UNIT_PATTERN = re.compile(
    r"\b(?:kgs?|kg/ha|kg/acre|q/ha|t/ha|quintal|mm|cm|ha|acre|%|days?|das|dat|"
    r"litre|liter|ml|g/m2|g/plant|plants?/m2|weeks?|months?)\b",
    re.IGNORECASE,
)


def _looks_like_contact_directory(text: str) -> bool:
    """Detect personnel or contact listings (names, designations, emails, phone numbers)."""
    n_emails = len(_EMAIL_PATTERN.findall(text))
    n_phone_labels = len(_PHONE_LABEL_PATTERN.findall(text))
    return n_emails >= 2 or n_phone_labels >= 3


def _looks_like_bare_data_table(text: str) -> bool:
    """Table chunk with no agronomic unit (kg, mm, %, ha, days, ...) in it."""
    return not _UNIT_PATTERN.search(text)


def _apply_content_quality_filter(chunks: list) -> tuple:
    """Drop contact directories and bare numeric tables. Returns (kept, n_contact_removed, n_bare_table_removed)."""
    kept, n_contact, n_bare_table = [], 0, 0
    for c in chunks:
        text = c.get("text", "")
        if _looks_like_contact_directory(text):
            n_contact += 1
            continue
        if c.get("type") == "table" and _looks_like_bare_data_table(text):
            n_bare_table += 1
            continue
        kept.append(c)
    return kept, n_contact, n_bare_table


def _shingles(text: str, k: int = 8) -> set:
    """Word k-shingles of a chunk, for near-duplicate detection."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def _dedupe_near_duplicates(chunks: list, threshold: float = 0.8) -> tuple:
    """Drop chunks whose shingle Jaccard similarity to an earlier chunk from the same file exceeds the threshold."""
    kept, n_removed = [], 0
    seen_shingles_by_file = {}
    for c in chunks:
        sh = _shingles(c["text"])
        prior = seen_shingles_by_file.setdefault(c["file"], [])
        if any(_jaccard(sh, other) >= threshold for other in prior):
            n_removed += 1
            continue
        prior.append(sh)
        kept.append(c)
    return kept, n_removed


_ocr_reader = None
_ocr_unavailable = False


def _get_ocr_reader():
    """Create the EasyOCR reader on first use; None if easyocr is not installed."""
    global _ocr_reader, _ocr_unavailable
    if _ocr_unavailable:
        return None
    if _ocr_reader is None:
        try:
            import easyocr
        except ImportError:
            log.warning("easyocr is not installed -- scanned/image-only PDF pages will be "
                        "skipped instead of OCR'd. Install with `pip install easyocr`.")
            _ocr_unavailable = True
            return None
        log.info("Loading EasyOCR reader (first use only; downloads model weights on first run)...")
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _ocr_page(page, source_file, page_num, chunk_size, overlap):
    """OCR a page that has no text layer and chunk the result like prose."""
    reader = _get_ocr_reader()
    if reader is None:
        return []
    try:
        image = page.to_image(resolution=200).original
        blocks = reader.readtext(np.array(image), detail=0, paragraph=True)
    except Exception as e:
        log.warning(f"OCR failed on {source_file} page {page_num}: {e}")
        return []

    paragraphs = [_clean_text(b) for b in blocks if b and b.strip()]
    chunks = _chunk_prose(paragraphs, source_file, page_num, chunk_size, overlap)
    for c in chunks:
        c["type"] = "ocr"
    return chunks


def _extract_page_prose(page, chunk_size, overlap, source_file, page_num, table_bboxes=None):
    if table_bboxes:
        prose_page = page.filter(lambda obj: _not_within_bboxes(obj, table_bboxes))
    else:
        prose_page = page
    prose_text = prose_page.extract_text() or ""

    paragraphs = [_clean_text(p) for p in prose_text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1 and prose_text.strip():
        # no blank-line breaks, fall back to line breaks
        paragraphs = [_clean_text(p) for p in prose_text.split("\n") if p.strip()]

    return _chunk_prose(paragraphs, source_file, page_num, chunk_size, overlap)


def chunk_pdf(pdf_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
    source_file = os.path.basename(pdf_path)
    chunks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # some PDFs draw bold text by printing each glyph twice
            page = page.dedupe_chars(tolerance=1)
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            page_chunks = []
            for table in tables:
                rows = table.extract()
                page_chunks.extend(_render_table_rows(rows, source_file, page_num))
            page_chunks.extend(
                _extract_page_prose(page, chunk_size, overlap, source_file, page_num, table_bboxes)
            )

            if not page_chunks and table_bboxes:
                # a page detected as one big table: re-extract it as prose
                full_text = page.extract_text() or ""
                if full_text.strip():
                    log.warning(f"{source_file} page {page_num}: detected table produced no "
                                f"usable chunks and swallowed the page's prose -- falling back "
                                f"to full-page prose extraction.")
                    page_chunks = _extract_page_prose(page, chunk_size, overlap, source_file, page_num)

            if not page_chunks:
                # scanned page, use OCR
                page_chunks = _ocr_page(page, source_file, page_num, chunk_size, overlap)

            chunks.extend(page_chunks)

    for c in chunks:
        c["locations"] = _detect_locations(c["text"])
        # region is tagged per chunk from its own text
        c["region"] = _detect_region(c["text"])

    return chunks


def main():
    pdf_dir = DEFAULT_PDF_DIR
    out_path = DEFAULT_OUTPUT
    chunk_size = DEFAULT_CHUNK_SIZE
    overlap = DEFAULT_OVERLAP

    log.info(f"Scanning {pdf_dir} for PDFs...")
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    log.info(f"Found {len(pdf_files)} PDF(s) in {pdf_dir}")

    all_chunks = []
    for pdf_file in pdf_files:
        try:
            chunks = chunk_pdf(pdf_file, chunk_size, overlap)
            all_chunks.extend(chunks)
            n_table = sum(1 for c in chunks if c["type"] == "table")
            n_prose = sum(1 for c in chunks if c["type"] == "prose")
            n_ocr = sum(1 for c in chunks if c["type"] == "ocr")
            log.info(f"  {os.path.basename(pdf_file)}: {len(chunks)} chunks "
                      f"({n_prose} prose, {n_table} table, {n_ocr} ocr)")
        except Exception as e:
            log.error(f"Error parsing {pdf_file}: {e}")

    # drop exact duplicates
    n_before = len(all_chunks)
    seen_texts = set()
    deduped_chunks = []
    for c in all_chunks:
        if c["text"] in seen_texts:
            continue
        seen_texts.add(c["text"])
        deduped_chunks.append(c)
    all_chunks = deduped_chunks
    log.info(f"Removed {n_before - len(all_chunks)} exact-duplicate chunk(s) "
              f"({n_before} -> {len(all_chunks)})")

    # drop near duplicates
    n_before_near = len(all_chunks)
    all_chunks, n_near_dup = _dedupe_near_duplicates(all_chunks)
    log.info(f"Removed {n_near_dup} near-duplicate chunk(s) "
              f"({n_before_near} -> {len(all_chunks)})")

    # drop contact directories and bare numeric tables
    n_before_quality = len(all_chunks)
    all_chunks, n_contact, n_bare_table = _apply_content_quality_filter(all_chunks)
    log.info(f"Removed {n_contact} contact-directory chunk(s) and {n_bare_table} bare-data-table "
              f"chunk(s) with no agronomic guidance value ({n_before_quality} -> {len(all_chunks)})")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    log.info(f"Wrote {len(all_chunks)} chunks total to {out_path}")

    n_tagged = sum(1 for c in all_chunks if c["locations"])
    log.info(f"{n_tagged}/{len(all_chunks)} chunks tagged with a known location")
    for loc in LOCATION_ALIASES:
        n = sum(1 for c in all_chunks if loc in c["locations"])
        log.info(f"  {loc}: {n} chunks")

    n_region = sum(1 for c in all_chunks if c.get("region"))
    log.info(f"{n_region}/{len(all_chunks)} chunks tagged region=Karnataka (per-chunk mention only)")

    # per-document share of the corpus
    file_counts = Counter(c["file"] for c in all_chunks)
    log.info("Chunk share by source file (top 5):")
    for f, n in file_counts.most_common(5):
        log.info(f"  {100 * n / len(all_chunks):5.1f}%  ({n:4d} chunks)  {f}")


if __name__ == "__main__":
    main()
