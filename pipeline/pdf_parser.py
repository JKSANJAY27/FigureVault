"""
pipeline/pdf_parser.py — Phase 1: PDF text & metadata extraction

Uses PyMuPDF (fitz) to extract:
  • Full paper text (per page)
  • Document metadata (title, author, DOI, abstract, year)
  • Text blocks with position info (bbox, font size)
  • Section-level text (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
  • In-text figure references with surrounding context
  • High-resolution page renders (PNG) for downstream processing

All public methods mirror the API described in the FigureVault dev bible.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from config import IMAGE_DPI, FIGURES_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------
_DOI_RE = re.compile(
    r"(?:doi[:\s]*|https?://(?:dx\.)?doi\.org/)"
    r"(10\.\d{4,9}/[-._;()/:A-Z0-9a-z]+)",
    re.IGNORECASE,
)
_DOI_BARE_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9a-z]+)", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_SECTION_HEADERS = re.compile(
    r"^\s*(abstract|introduction|methods?|materials?\s+and\s+methods?|"
    r"results?|discussion|conclusions?|acknowledgements?|references?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FIG_REF_RE = re.compile(
    r"\b(?:Fig(?:ure|s?\.?)?)\s*\.?\s*([A-Z]?\d+[a-z]?(?:\s*[–-]\s*\d+)?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PaperMetadata:
    """Structured metadata extracted from a PDF."""
    pdf_path: str
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    doi: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    abstract: Optional[str] = None
    page_count: int = 0
    full_text: str = ""           # concatenated plain text of all pages
    page_texts: list[str] = field(default_factory=list)  # per-page text


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class PDFParser:
    """Parse a scientific PDF and return text, metadata, and rendered page images.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the input PDF file.
    dpi : int
        Render resolution for page images (default: IMAGE_DPI from config).

    Context-manager usage::

        with PDFParser("paper.pdf") as p:
            meta = p.parse()
    """

    def __init__(self, pdf_path: str | Path, dpi: int = IMAGE_DPI) -> None:
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi
        self._doc: Optional[fitz.Document] = None

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "PDFParser":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying fitz document if open."""
        if self._doc is not None:
            try:
                self._doc.close()
            except Exception:
                pass
            self._doc = None

    # ------------------------------------------------------------------
    # Public high-level API
    # ------------------------------------------------------------------

    def parse(self) -> PaperMetadata:
        """Open the PDF and extract all text & metadata.

        Returns
        -------
        PaperMetadata
            Structured metadata ready for downstream pipeline phases.
        """
        logger.info("Parsing PDF: %s", self.pdf_path)
        try:
            doc = self._open_doc()
        except fitz.EmptyFileError as exc:
            raise ValueError(f"Cannot parse PDF: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Cannot parse PDF: {exc}") from exc

        meta = self._extract_metadata_from_doc(doc)
        meta.page_texts = self._extract_page_texts(doc)
        meta.full_text = "\n\n".join(meta.page_texts)
        meta.page_count = doc.page_count

        # Infer missing fields from body text
        if not meta.doi:
            meta.doi = self._sniff_doi(meta.full_text)
        if not meta.year:
            meta.year = self._sniff_year(meta.full_text[:4000])
        if not meta.title:
            meta.title = self._sniff_title(doc)
        if not meta.abstract:
            meta.abstract = self._extract_abstract(meta.full_text)

        logger.info(
            "Parsed %d pages — doi=%s  title=%.60s",
            meta.page_count, meta.doi, meta.title or "(unknown)",
        )
        return meta

    def extract_metadata(self) -> dict:
        """Extract paper metadata from the PDF.

        Returns
        -------
        dict
            Keys: title, doi, authors, abstract, journal, year
        """
        doc = self._open_doc()
        meta = self._extract_metadata_from_doc(doc)
        page_texts = self._extract_page_texts(doc)
        full_text = "\n\n".join(page_texts)

        if not meta.doi:
            meta.doi = self._sniff_doi(full_text)
        if not meta.year:
            meta.year = self._sniff_year(full_text[:4000])
        if not meta.title:
            meta.title = self._sniff_title(doc)
        if not meta.abstract:
            meta.abstract = self._extract_abstract(full_text)

        return {
            "title": meta.title,
            "doi": meta.doi,
            "authors": meta.authors,
            "abstract": meta.abstract,
            "journal": meta.journal,
            "year": meta.year,
        }

    def extract_full_text(self) -> dict[int, str]:
        """Return dict mapping page_number (0-indexed) → page text.

        Also flags pages that appear to be scanned (text length < 100 chars).

        Returns
        -------
        dict[int, str]
            {0: "page 1 text", 1: "page 2 text", ...}
        """
        doc = self._open_doc()
        result: dict[int, str] = {}
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if len(text.strip()) < 100:
                logger.warning(
                    "Page %d of %s may be scanned (text length=%d)",
                    i + 1, self.pdf_path.name, len(text.strip()),
                )
            result[i] = text
        return result

    def extract_text_blocks(self) -> dict[int, list[dict]]:
        """Extract text blocks with position info for each page.

        Each block dict: {"text": str, "bbox": [x0,y0,x1,y1], "page": int, "font_size": float}
        Blocks with font_size < 4 are discarded (PDF rendering artefacts).

        Returns
        -------
        dict[int, list[dict]]
            {page_index: [block, ...]}
        """
        doc = self._open_doc()
        result: dict[int, list[dict]] = {}
        for page_idx, page in enumerate(doc):
            raw = page.get_text("dict")
            blocks: list[dict] = []
            for block in raw.get("blocks", []):
                if block.get("type") != 0:          # 0 = text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_size = span.get("size", 0)
                        if font_size < 4:
                            continue
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        bbox = list(span.get("bbox", [0, 0, 0, 0]))
                        blocks.append({
                            "text": text,
                            "bbox": bbox,
                            "page": page_idx,
                            "font_size": font_size,
                        })
            result[page_idx] = blocks
        return result

    def render_pages(self, output_dir: Optional[str | Path] = None) -> list[str]:
        """Render all pages as PNG images at self.dpi resolution.

        Parameters
        ----------
        output_dir : str | Path, optional
            Directory where PNGs are written.  Defaults to FIGURES_DIR / pdf stem.

        Returns
        -------
        list[str]
            Sorted list of rendered page image file paths.
        """
        out = Path(output_dir or FIGURES_DIR) / self.pdf_path.stem
        out.mkdir(parents=True, exist_ok=True)

        doc = self._open_doc()
        paths: list[str] = []
        mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 dpi is PDF default

        for i, page in enumerate(doc):
            img_path = out / f"page_{i:03d}.png"
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pix.save(str(img_path))
            paths.append(str(img_path))
            logger.debug("Rendered page %d → %s", i, img_path)

        logger.info("Rendered %d pages to %s", len(paths), out)
        return paths

    def extract_section_text(self) -> dict[str, str]:
        """Identify major paper sections and return their text.

        Sections detected: Abstract, Introduction, Methods, Results,
        Discussion, Conclusion (case-insensitive, standalone lines).

        Returns
        -------
        dict[str, str]
            {section_name: section_text}  — section_name is title-cased.
        """
        full_text = "\n\n".join(self._get_page_texts())
        sections: dict[str, str] = {}
        section_starts: list[tuple[int, str]] = []

        for m in _SECTION_HEADERS.finditer(full_text):
            name = m.group(1).strip().title()
            # Normalise aliases
            name = re.sub(r"Materials?\s+And\s+Methods?", "Methods", name, flags=re.IGNORECASE)
            section_starts.append((m.end(), name))

        for i, (start, name) in enumerate(section_starts):
            end = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(full_text)
            body = full_text[start:end].strip()
            if name in sections:
                sections[name] += "\n\n" + body
            else:
                sections[name] = body

        return sections

    def find_figure_references(self) -> list[dict]:
        """Find all in-text references to figures.

        Matches: "Figure 1", "Fig. 1", "Fig 1A", "Figure S1", etc.

        Returns
        -------
        list[dict]
            Each: {"figure_ref": str, "page": int, "context": str (±200 chars)}
        """
        page_texts = self._get_page_texts()
        results: list[dict] = []

        for page_idx, text in enumerate(page_texts):
            for m in _FIG_REF_RE.finditer(text):
                start = max(0, m.start() - 200)
                end   = min(len(text), m.end() + 200)
                results.append({
                    "figure_ref": m.group(0).strip(),
                    "page": page_idx,
                    "context": text[start:end].strip(),
                })

        return results

    def get_page_text(self, page_number: int) -> str:
        """Return plain text from a single page (1-indexed).

        Parameters
        ----------
        page_number : int
            1-based page index.
        """
        doc = self._open_doc()
        page = doc[page_number - 1]
        return page.get_text("text")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_doc(self) -> fitz.Document:
        """Open (or return cached) fitz Document, with validation."""
        if self._doc is not None and not self._doc.is_closed:
            return self._doc
        try:
            doc = fitz.open(str(self.pdf_path))
        except Exception as exc:
            raise ValueError(f"Cannot parse PDF: {exc}") from exc

        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        if doc.page_count == 0:
            raise ValueError("PDF appears to be empty")

        self._doc = doc
        return doc

    def _get_page_texts(self) -> list[str]:
        """Return cached per-page texts, opening doc if necessary."""
        doc = self._open_doc()
        return [page.get_text("text") for page in doc]

    @staticmethod
    def _extract_metadata_from_doc(doc: fitz.Document) -> PaperMetadata:
        """Pull metadata from the PDF info dictionary."""
        info = doc.metadata or {}
        title: Optional[str] = info.get("title") or None
        author_raw: str = info.get("author") or ""
        authors = [a.strip() for a in re.split(r"[;,]", author_raw) if a.strip()]
        creator = info.get("creator") or info.get("producer") or ""
        # Some publishers embed the journal name in creator/producer
        journal: Optional[str] = None
        if creator and "Microsoft" not in creator and "Adobe" not in creator:
            journal = creator[:120] or None
        return PaperMetadata(
            pdf_path=doc.name,
            title=title or None,
            authors=authors,
            journal=journal,
        )

    @staticmethod
    def _extract_page_texts(doc: fitz.Document) -> list[str]:
        """Extract plain text from every page."""
        return [page.get_text("text") for page in doc]

    @classmethod
    def _sniff_doi(cls, text: str) -> Optional[str]:
        """Attempt to find a DOI in the extracted text."""
        # Try with prefix first (more reliable)
        m = _DOI_RE.search(text[:6000])
        if m:
            return m.group(1).rstrip(".")
        # Fall back to bare DOI pattern
        m = _DOI_BARE_RE.search(text[:6000])
        return m.group(1).rstrip(".") if m else None

    @classmethod
    def _sniff_year(cls, text: str) -> Optional[int]:
        """Heuristically extract a publication year from early pages."""
        matches = _YEAR_RE.findall(text)
        if not matches:
            return None
        most_common = Counter(matches).most_common(1)[0][0]
        return int(most_common)

    @staticmethod
    def _sniff_title(doc: fitz.Document) -> Optional[str]:
        """Try to infer the paper title from the largest font text on page 1."""
        if doc.page_count == 0:
            return None
        page = doc[0]
        raw = page.get_text("dict")
        biggest: tuple[float, str] = (0.0, "")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    text = span.get("text", "").strip()
                    if size > biggest[0] and len(text) > 10:
                        biggest = (size, text)
        return biggest[1] or None

    @staticmethod
    def _extract_abstract(full_text: str) -> Optional[str]:
        """Extract the abstract by finding text between 'Abstract' and 'Introduction'."""
        m = re.search(
            r"\bAbstract\b(.{50,3000}?)\b(?:Introduction|Keywords?|Background)\b",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        return None


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_pdf>")
        sys.exit(1)

    logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    pdf_path = sys.argv[1]

    with PDFParser(pdf_path) as p:
        meta = p.parse()
        print("=" * 60)
        print(f"File      : {meta.pdf_path}")
        print(f"Pages     : {meta.page_count}")
        print(f"Title     : {meta.title}")
        print(f"Authors   : {', '.join(meta.authors) or '(none)'}")
        print(f"DOI       : {meta.doi}")
        print(f"Year      : {meta.year}")
        print(f"Journal   : {meta.journal}")
        print(f"Abstract  : {(meta.abstract or '')[:200]}...")
        print()
        sections = p.extract_section_text()
        print(f"Sections found: {list(sections.keys())}")
        refs = p.find_figure_references()
        print(f"Figure references found: {len(refs)}")
        if refs:
            print(f"  First ref: {refs[0]['figure_ref']} (page {refs[0]['page']})")
        print("=" * 60)
