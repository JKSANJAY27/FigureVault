"""
pipeline/figure_extractor.py — Phase 2: Figure image extraction

Strategy (with automatic fallback):
  1. Try PDFFigures2 (Java JAR, built from allenai/pdffigures2 via sbt assembly).
     The JAR is called with the FigureExtractorBatchCli main class.
     It outputs:
       • <output_prefix><stem>.json  — figure metadata (caption, bbox, page, type)
       • <image_prefix><stem>-Figure<N>-<page>.png  — cropped figure images
  2. If PDFFigures2 is unavailable (no Java / JAR missing), fall back to a
     PyMuPDF heuristic that detects image blocks inside PDF pages.

Produces one PNG per detected figure + a FigureRecord dataclass per image.

PDFFigures2 CLI reference (from allenai/pdffigures2 README):
  java -cp pdffigures2.jar org.allenai.pdffigures2.FigureExtractorBatchCli \\
       /path/to/pdf_dir/ \\
       -m <image_output_prefix> \\
       -d <data_output_prefix> \\
       -s stat_file.json

When running on a SINGLE pdf file the prefix-based output naming still applies.
We work around this by creating a temp directory per-call and collecting results.
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

# PDFFigures2 main class for batch extraction
_PDFFIGURES2_MAIN = "org.allenai.pdffigures2.FigureExtractorBatchCli"


@dataclass
class FigureRecord:
    """Represents a single extracted figure (or sub-panel)."""
    page_number: int
    figure_number: str                   # e.g. "3", "S2", "Figure3"
    panel_label: str = ""                # e.g. "A", "B"
    caption: str = ""
    figure_type: Optional[str] = None    # set by classifier in Phase 3
    image_path: Optional[Path] = None   # path to saved PNG
    bounding_box: dict = field(default_factory=dict)  # {x, y, w, h} in PDF points
    confidence: float = 1.0             # set lower for heuristic extractions


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
                records = self._extract_with_pdffigures2()
                if records:
                    return records
                logger.info("PDFFigures2 returned 0 figures — falling back to PyMuPDF")
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

        PDFFigures2 FigureExtractorBatchCli flags:
          -m <prefix>  — image output prefix (images saved as <prefix><stem>-Figure<N>-<page>.png)
          -d <prefix>  — data output prefix  (JSON saved as <prefix><stem>.json)
          -r <dpi>     — render DPI (default 72 inside the JAR, we pass our DPI)
          -q           — quiet mode

        The JAR processes a directory of PDFs.  We create a temporary single-PDF
        directory so we can use the batch CLI on a single file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_tmp_dir = tmp_dir / "input"
            pdf_tmp_dir.mkdir()
            img_prefix_dir = tmp_dir / "images"
            img_prefix_dir.mkdir()
            data_prefix_dir = tmp_dir / "data"
            data_prefix_dir.mkdir()

            # Symlink / copy the PDF into a temp folder so the batch CLI can find it
            pdf_link = pdf_tmp_dir / self.pdf_path.name
            shutil.copy2(self.pdf_path, pdf_link)

            img_prefix = str(img_prefix_dir) + "/"
            data_prefix = str(data_prefix_dir) + "/"

            cmd = [
                "java",
                "-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider",
                "-cp", str(PDFFIGURES2_JAR),
                _PDFFIGURES2_MAIN,
                str(pdf_tmp_dir) + "/",   # trailing slash = directory of PDFs
                "-m", img_prefix,
                "-d", data_prefix,
                "-q",                     # quiet — suppress verbose Scala logging
            ]

            logger.debug("PDFFigures2 cmd: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                stderr_snippet = result.stderr[:600] if result.stderr else "(no stderr)"
                raise RuntimeError(
                    f"PDFFigures2 exited {result.returncode}: {stderr_snippet}"
                )

            # Locate the JSON output file — named after the PDF stem
            json_candidates = list(data_prefix_dir.glob("*.json"))
            if not json_candidates:
                logger.warning("PDFFigures2 produced no JSON output")
                return []

            data = json.loads(json_candidates[0].read_text(encoding="utf-8"))
            records: list[FigureRecord] = []

            for fig in data:
                fig_type_raw = fig.get("figType", "Figure")  # "Figure" or "Table"
                fig_num = str(fig.get("name", fig.get("number", len(records) + 1)))
                caption = fig.get("caption", "")
                page = int(fig.get("page", 0))

                bb_raw = fig.get("regionBoundary", fig.get("captionBoundary", {}))
                bounding_box = {
                    "x": bb_raw.get("x1", 0),
                    "y": bb_raw.get("y1", 0),
                    "w": bb_raw.get("x2", 0) - bb_raw.get("x1", 0),
                    "h": bb_raw.get("y2", 0) - bb_raw.get("y1", 0),
                }

                # PDFFigures2 saves images as: <prefix><stem>-<figType><num>-<page>.png
                stem = self.pdf_path.stem
                img_glob = f"{stem}-{fig_type_raw}{fig_num}*.png"
                src_images = list(img_prefix_dir.glob(img_glob))

                dest_image: Optional[Path] = None
                if src_images:
                    fname = f"fig_{fig_type_raw}{fig_num}_p{page}.png"
                    dest_image = self.output_dir / fname
                    shutil.copy(src_images[0], dest_image)
                else:
                    # Fall back: render the bounding-box region ourselves
                    dest_image = self._render_bbox_region(page, bb_raw, fig_num)

                records.append(FigureRecord(
                    page_number=page,
                    figure_number=fig_num,
                    caption=caption,
                    image_path=dest_image,
                    bounding_box=bounding_box,
                    confidence=0.95,  # PDFFigures2 is highly reliable
                ))

        logger.info("PDFFigures2 extracted %d figures", len(records))
        return records

    def _render_bbox_region(
        self,
        page_number: int,
        bb: dict,
        fig_label: str,
    ) -> Optional[Path]:
        """Render a bounding-box region from a PDF page using PyMuPDF.

        Used as a fallback when PDFFigures2 does not save an image.
        bb keys: x1, y1, x2, y2 (PDF points at 72 DPI).
        """
        try:
            doc = fitz.open(str(self.pdf_path))
            if page_number >= doc.page_count:
                doc.close()
                return None
            page = doc[page_number]
            rect = fitz.Rect(
                bb.get("x1", 0), bb.get("y1", 0),
                bb.get("x2", 1), bb.get("y2", 1),
            )
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
            img_path = self.output_dir / f"fig_{fig_label}_render.png"
            pix.save(str(img_path))
            doc.close()
            return img_path
        except Exception as exc:
            logger.warning("Fallback region render failed: %s", exc)
            return None

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

                    # Skip tiny images (logos, icons, decorations)
                    area = rect.width * rect.height
                    if area < 5000:   # pt² — roughly 2 cm × 2 cm at 72 dpi
                        continue

                    # Render the cropped region with small padding
                    clip = rect + fitz.Rect(-5, -5, 5, 5)
                    clip &= page.rect   # clamp to page bounds
                    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                    img_path = self.output_dir / f"fig_{fig_counter:03d}.png"
                    pix.save(str(img_path))

                    # Heuristic caption: text in ~60pt below the figure
                    caption_rect = fitz.Rect(
                        rect.x0, rect.y1,
                        rect.x1, rect.y1 + 60,
                    )
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
                        confidence=0.6,   # lower confidence for heuristic
                    ))
                    fig_counter += 1
        finally:
            doc.close()

        logger.info("PyMuPDF heuristic extracted %d figures", len(records))
        return records
