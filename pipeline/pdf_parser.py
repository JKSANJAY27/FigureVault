"""
pipeline/pdf_parser.py — Phase 1: PDF text & metadata extraction

Uses PyMuPDF (fitz) to extract:
  • Full paper text (per page)
  • Document metadata (title, author, DOI heuristic)
  • Page renders at configured DPI (used as fallback figure source)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from config import IMAGE_DPI, FIGURES_DIR

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Structured metadata extracted from a PDF."""
    pdf_path: str
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    doi: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    page_count: int = 0
    full_text: str = ""           # concatenated plain text of all pages
    page_texts: list[str] = field(default_factory=list)  # per-page text


class PDFParser:
    """Parse a scientific PDF and return text, metadata and rendered page images.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the input PDF file.
    dpi : int
        Render resolution for page images (default: IMAGE_DPI from config).
    """

    # Regex to sniff DOIs from raw text
    _DOI_RE = re.compile(
        r"\b(?:doi[:\s]*|https?://(?:dx\.)?doi\.org/)"
        r"(10\.\d{4,9}/\S+)",
        re.IGNORECASE,
    )
    # Rough year detector (4-digit number in 1900-2099)
    _YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

    def __init__(self, pdf_path: str | Path, dpi: int = IMAGE_DPI) -> None:
        self.pdf_path = Path(pdf_path)
        self.dpi = dpi

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self) -> PaperMetadata:
        """Open the PDF, extract all text and metadata.

        Returns
        -------
        PaperMetadata
            Structured metadata object ready for downstream pipeline phases.
        """
        logger.info("Parsing PDF: %s", self.pdf_path)
        doc = fitz.open(str(self.pdf_path))
        try:
            meta = self._extract_metadata(doc)
            meta.page_texts = self._extract_page_texts(doc)
            meta.full_text = "\n\n".join(meta.page_texts)
            meta.page_count = doc.page_count

            # Try to infer missing fields from text
            if not meta.doi:
                meta.doi = self._sniff_doi(meta.full_text)
            if not meta.year:
                meta.year = self._sniff_year(meta.full_text[:3000])
        finally:
            doc.close()

        logger.info(
            "Parsed %d pages — doi=%s  title=%s",
            meta.page_count, meta.doi, meta.title,
        )
        return meta

    def render_pages(self, output_dir: Optional[Path] = None) -> list[Path]:
        """Render every page as a PNG image and return the list of file paths.

        Used as a fallback when PDFFigures2 is unavailable.

        Parameters
        ----------
        output_dir : Path, optional
            Directory where PNGs are written.  Defaults to FIGURES_DIR / pdf stem.

        Returns
        -------
        list[Path]
            Sorted list of rendered page image paths.
        """
        out = Path(output_dir or FIGURES_DIR) / self.pdf_path.stem
        out.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(self.pdf_path))
        paths: list[Path] = []
        try:
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)  # 72 dpi is PDF default
            for i, page in enumerate(doc):
                img_path = out / f"page_{i + 1:04d}.png"
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pix.save(str(img_path))
                paths.append(img_path)
                logger.debug("Rendered page %d → %s", i + 1, img_path)
        finally:
            doc.close()

        logger.info("Rendered %d pages to %s", len(paths), out)
        return paths

    def get_page_text(self, page_number: int) -> str:
        """Return plain text from a single page (1-indexed).

        Parameters
        ----------
        page_number : int
            1-based page index.

        Returns
        -------
        str
            Extracted plain text for the requested page.
        """
        doc = fitz.open(str(self.pdf_path))
        try:
            page = doc[page_number - 1]
            return page.get_text("text")
        finally:
            doc.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(doc: fitz.Document) -> PaperMetadata:
        """Pull metadata from the PDF info dictionary."""
        info = doc.metadata or {}
        title: Optional[str] = info.get("title") or None
        author_raw: str = info.get("author") or ""
        authors = [a.strip() for a in re.split(r"[;,]", author_raw) if a.strip()]
        return PaperMetadata(
            pdf_path=doc.name,
            title=title or None,
            authors=authors,
        )

    @staticmethod
    def _extract_page_texts(doc: fitz.Document) -> list[str]:
        """Extract plain text from every page."""
        return [page.get_text("text") for page in doc]

    @classmethod
    def _sniff_doi(cls, text: str) -> Optional[str]:
        """Attempt to find a DOI in the extracted text."""
        match = cls._DOI_RE.search(text[:5000])  # DOI usually near the top
        return match.group(1).rstrip(".") if match else None

    @classmethod
    def _sniff_year(cls, text: str) -> Optional[int]:
        """Heuristically extract a publication year from early pages."""
        matches = cls._YEAR_RE.findall(text)
        if matches:
            # Most common year in the first 3000 chars is likely the pub year
            from collections import Counter
            most_common = Counter(matches).most_common(1)[0][0]
            return int(most_common)
        return None
