"""
pipeline/figure_extractor.py — Phase 2: Figure image extraction

Strategy (with automatic fallback):
  1. Try PDFFigures2 (Java JAR) — battle-tested on 1 M+ papers, gives tight
     bounding boxes and captions out of the box.
  2. If PDFFigures2 is unavailable (no Java / JAR missing), fall back to a
     PyMuPDF heuristic that detects image blocks inside PDF pages.

Produces one PNG per detected figure + a FigureRecord dataclass per image.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from config import FIGURES_DIR, IMAGE_DPI, PDFFIGURES2_JAR

logger = logging.getLogger(__name__)


@dataclass
class FigureRecord:
    """Represents a single extracted figure (or sub-panel)."""
    page_number: int
    figure_number: str                   # e.g. "3", "S2"
    panel_label: str = ""               # e.g. "A", "B"
    caption: str = ""
    image_path: Optional[Path] = None   # path to saved PNG
    bounding_box: dict = field(default_factory=dict)  # {x, y, w, h} in PDF points
    confidence: float = 1.0             # set low for heuristic extractions


class FigureExtractor:
    """Extract figure images from a scientific PDF.

    Parameters
    ----------
    pdf_path : str | Path
        Source PDF.
    output_dir : Path, optional
        Where to write extracted figure PNGs.
    dpi : int
        Render DPI for extracted figures.
    use_pdffigures2 : bool
        If True (default), attempt to use the PDFFigures2 JAR first.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        output_dir: Optional[Path] = None,
        dpi: int = IMAGE_DPI,
        use_pdffigures2: bool = True,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir or FIGURES_DIR) / self.pdf_path.stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.use_pdffigures2 = use_pdffigures2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> list[FigureRecord]:
        """Extract all figures from the PDF.

        Returns
        -------
        list[FigureRecord]
            One record per detected figure/panel, with image paths set.
        """
        if self.use_pdffigures2 and self._pdffigures2_available():
            logger.info("Using PDFFigures2 for extraction")
            try:
                return self._extract_with_pdffigures2()
            except Exception as exc:
                logger.warning(
                    "PDFFigures2 failed (%s) — falling back to PyMuPDF heuristic", exc
                )
        logger.info("Using PyMuPDF heuristic for figure extraction")
        return self._extract_with_pymupdf()

    # ------------------------------------------------------------------
    # PDFFigures2 strategy
    # ------------------------------------------------------------------

    @staticmethod
    def _pdffigures2_available() -> bool:
        """Return True if Java is on PATH and the PDFFigures2 JAR exists."""
        if not PDFFIGURES2_JAR.exists():
            logger.debug("PDFFigures2 JAR not found at %s", PDFFIGURES2_JAR)
            return False
        if shutil.which("java") is None:
            logger.debug("Java not found on PATH")
            return False
        return True

    def _extract_with_pdffigures2(self) -> list[FigureRecord]:
        """Run PDFFigures2 and parse its JSON output.

        PDFFigures2 outputs:
          • <stem>-figures.json  — list of figure metadata
          • <stem>-<n>.png       — cropped figure images
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            cmd = [
                "java", "-jar", str(PDFFIGURES2_JAR),
                str(self.pdf_path),
                "--figure-data-folder", str(tmp_dir),
                "--save-figures", str(tmp_dir),
                "--dpi", str(self.dpi),
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"PDFFigures2 exited {result.returncode}: {result.stderr[:500]}"
                )

            json_files = list(tmp_dir.glob("*-figures.json"))
            if not json_files:
                return []

            data = json.loads(json_files[0].read_text(encoding="utf-8"))
            records: list[FigureRecord] = []

            for fig in data:
                fig_num = str(fig.get("figType", "?") + str(fig.get("number", "")))
                caption = fig.get("caption", "")
                page = int(fig.get("page", 0))
                bb = fig.get("regionBoundary", {})
                bounding_box = {
                    "x": bb.get("x1", 0), "y": bb.get("y1", 0),
                    "w": bb.get("x2", 0) - bb.get("x1", 0),
                    "h": bb.get("y2", 0) - bb.get("y1", 0),
                }

                # Locate the rendered image produced by PDFFigures2
                src_images = list(tmp_dir.glob(f"*-{fig.get('number', '')}*.png"))
                dest_image: Optional[Path] = None
                if src_images:
                    dest_image = self.output_dir / f"fig_{fig_num}.png"
                    shutil.copy(src_images[0], dest_image)

                records.append(FigureRecord(
                    page_number=page,
                    figure_number=fig_num,
                    caption=caption,
                    image_path=dest_image,
                    bounding_box=bounding_box,
                ))

        logger.info("PDFFigures2 extracted %d figures", len(records))
        return records

    # ------------------------------------------------------------------
    # PyMuPDF heuristic fallback
    # ------------------------------------------------------------------

    def _extract_with_pymupdf(self) -> list[FigureRecord]:
        """Detect image blocks inside each PDF page and crop them out.

        This is a best-effort heuristic: it finds raster images embedded in
        the PDF and captures the surrounding area.  Caption extraction is
        limited to text blocks immediately below the image bounding box.

        Returns
        -------
        list[FigureRecord]
        """
        doc = fitz.open(str(self.pdf_path))
        records: list[FigureRecord] = []
        fig_counter = 1

        try:
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            for page_idx, page in enumerate(doc):
                image_list = page.get_images(full=True)
                for img_info in image_list:
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue
                    rect = rects[0]

                    # Skip tiny images (logos, icons, etc.)
                    area = rect.width * rect.height
                    if area < 5000:  # pt² — roughly 2cm×2cm at 72 dpi
                        continue

                    # Render the cropped region
                    clip = rect + fitz.Rect(-5, -5, 5, 5)  # small padding
                    clip &= page.rect  # clamp to page bounds
                    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                    img_path = self.output_dir / f"fig_{fig_counter:03d}.png"
                    pix.save(str(img_path))

                    # Heuristic caption extraction (text in 50pt below figure)
                    caption_rect = fitz.Rect(rect.x0, rect.y1, rect.x1, rect.y1 + 50)
                    caption_rect &= page.rect
                    caption = page.get_text("text", clip=caption_rect).strip()

                    records.append(FigureRecord(
                        page_number=page_idx + 1,
                        figure_number=str(fig_counter),
                        caption=caption,
                        image_path=img_path,
                        bounding_box={
                            "x": rect.x0, "y": rect.y0,
                            "w": rect.width, "h": rect.height,
                        },
                        confidence=0.6,  # lower confidence for heuristic
                    ))
                    fig_counter += 1
        finally:
            doc.close()

        logger.info("PyMuPDF heuristic extracted %d figures", len(records))
        return records
