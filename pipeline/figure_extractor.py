"""
pipeline/figure_extractor.py — Phase 2: Figure Image Extraction Engine

Two-strategy approach:
  1. PDFFigures2 (Java JAR, primary method) — extremely accurate for CS/bio papers.
     Calls the JAR via subprocess and parses its JSON output.
  2. PyMuPDF fallback (pure Python) — handles papers PDFFigures2 misses.
     Detects both raster images and vector graphics drawn as PDF operators.

After extraction, both strategies run multi-panel detection on each figure:
  - If the caption contains panel labels (A), (B) etc., OpenCV splits the image
    along white-space gaps into individual sub-panel PNGs.

Output directory layout:
  output_dir/
    figures/
      <paper_slug>/
        fig_001_full.png
        fig_001A.png
        fig_001B.png
    metadata/
      <paper_slug>/
        figures.json

Public entrypoint: FigureExtractor.extract_all(pdf_path, output_dir, page_renders)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from config import FIGURES_DIR, IMAGE_DPI, MIN_FIGURE_AREA_PX, PDFFIGURES2_JAR

logger = logging.getLogger(__name__)

# PDFFigures2 main class
_PDFFIGURES2_MAIN = "org.allenai.pdffigures2.FigureExtractorBatchCli"

# Regex to detect panel labels inside captions
_PANEL_LABEL_RE = re.compile(
    r'[\(\[]([A-H])[\)\]]'        # (A), [B]
    r'|(?<!\w)([A-H])\)'          # A), B)
    r'|(?<!\w)([A-H])\.\s',       # A., B.
    re.IGNORECASE,
)

# Caption-start pattern used by PyMuPDF heuristic
_CAPTION_START_RE = re.compile(
    r'^(Figure|Fig\.|FIGURE|Supplementary\s+Figure|Supp\.?\s*Fig\.?)\s*[\dS]',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FigureRecord:
    """Represents a single extracted figure or sub-panel."""
    page_number: int
    figure_number: str            # e.g. "1", "S3", "Figure1"
    panel_label: str = ""         # e.g. "A", "B"  (empty = full figure)
    caption: str = ""
    figure_type: Optional[str] = None   # set later by classifier
    image_path: Optional[Path] = None  # path to saved PNG
    bounding_box: dict = field(default_factory=dict)  # {x, y, w, h}
    source: str = "unknown"        # "pdffigures2" | "pymupdf"
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "figure_number": self.figure_number,
            "panel_label": self.panel_label,
            "caption": self.caption,
            "page": self.page_number,
            "image_path": str(self.image_path) if self.image_path else None,
            "bounding_box": self.bounding_box,
            "source": self.source,
            "figure_type": self.figure_type,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Strategy 1: PDFFigures2
# ---------------------------------------------------------------------------

class PDFFiguresExtractor:
    """Wrap the PDFFigures2 Java JAR for high-accuracy figure extraction."""

    def __init__(self, jar_path: str | Path = PDFFIGURES2_JAR) -> None:
        self.jar_path = Path(jar_path)

    def is_available(self) -> bool:
        """Return True if Java is on PATH and the JAR exists."""
        if not self.jar_path.exists():
            logger.debug("PDFFigures2 JAR not found: %s", self.jar_path)
            return False
        try:
            subprocess.run(
                ["java", "-version"],
                capture_output=True, timeout=10, check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("Java not found on PATH")
            return False

    def extract(self, pdf_path: str | Path, output_dir: str | Path) -> list[dict]:
        """Run PDFFigures2 on a single PDF and return a list of figure dicts.

        Parameters
        ----------
        pdf_path:
            Path to the input PDF.
        output_dir:
            Directory where extracted PNG images and metadata will be written.

        Returns
        -------
        list[dict]  Each dict has keys:
            figure_number, caption, page, image_path, bounding_box, source
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        figures_out = output_dir / "figures" / pdf_path.stem
        metadata_out = output_dir / "metadata" / pdf_path.stem
        figures_out.mkdir(parents=True, exist_ok=True)
        metadata_out.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_tmp_dir = tmp_dir / "input"
            pdf_tmp_dir.mkdir()
            img_prefix_dir = tmp_dir / "images"
            img_prefix_dir.mkdir()
            data_prefix_dir = tmp_dir / "data"
            data_prefix_dir.mkdir()

            shutil.copy2(pdf_path, pdf_tmp_dir / pdf_path.name)

            cmd = [
                "java",
                "-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider",
                "-cp", str(self.jar_path),
                _PDFFIGURES2_MAIN,
                str(pdf_tmp_dir) + "/",
                "-m", str(img_prefix_dir) + "/",
                "-d", str(data_prefix_dir) + "/",
                "-q",
            ]
            logger.debug("PDFFigures2 cmd: %s", " ".join(cmd))

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"PDFFigures2 exited {result.returncode}: "
                    f"{result.stderr[:600] or '(no stderr)'}"
                )

            json_files = list(data_prefix_dir.glob("*.json"))
            if not json_files:
                logger.warning("PDFFigures2 produced no JSON output for %s", pdf_path.name)
                return []

            raw_data = json.loads(json_files[0].read_text(encoding="utf-8"))
            results: list[dict] = []

            for fig in raw_data:
                fig_type_raw = fig.get("figType", "Figure")
                fig_num = str(fig.get("name", fig.get("number", len(results) + 1)))
                caption = fig.get("caption", "")
                page = int(fig.get("page", 0))

                bb_raw = fig.get("regionBoundary", fig.get("captionBoundary", {}))
                bounding_box = [
                    bb_raw.get("x1", 0), bb_raw.get("y1", 0),
                    bb_raw.get("x2", 0), bb_raw.get("y2", 0),
                ]

                # Locate extracted PNG from PDFFigures2
                stem = pdf_path.stem
                src_candidates = list(img_prefix_dir.glob(
                    f"{stem}-{fig_type_raw}{fig_num}*.png"
                ))
                image_path: Optional[Path] = None
                if src_candidates:
                    dest_name = f"fig_{fig_num.zfill(3)}_full.png"
                    image_path = figures_out / dest_name
                    shutil.copy(src_candidates[0], image_path)
                    logger.info(
                        "PDFFigures2 → figure %s | page %d | %s",
                        fig_num, page, image_path.name,
                    )
                else:
                    # Render the bounding-box region with PyMuPDF
                    image_path = _render_pdf_region(
                        pdf_path, page, bb_raw, figures_out / f"fig_{fig_num.zfill(3)}_full.png"
                    )
                    if image_path:
                        logger.info(
                            "PDFFigures2 bbox render → figure %s | page %d", fig_num, page
                        )

                results.append({
                    "figure_number": fig_num,
                    "caption": caption,
                    "page": page,
                    "image_path": str(image_path) if image_path else None,
                    "bounding_box": bounding_box,
                    "source": "pdffigures2",
                })

        return results


# ---------------------------------------------------------------------------
# Strategy 2: PyMuPDF fallback
# ---------------------------------------------------------------------------

class PyMuPDFFigureExtractor:
    """Pure-Python figure extractor using PyMuPDF.

    Handles both raster images embedded in the PDF and vector graphics
    (charts drawn as PDF operators).
    """

    def extract(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        page_renders: Optional[list[str]] = None,
    ) -> list[dict]:
        """Extract figures from a PDF using PyMuPDF heuristics.

        Parameters
        ----------
        pdf_path:
            Path to the PDF.
        output_dir:
            Destination directory (figures/ and metadata/ sub-dirs created here).
        page_renders:
            Optional list of pre-rendered page PNG paths (not used directly here,
            kept for API compatibility with callers that pass them).

        Returns
        -------
        list[dict]  Same schema as PDFFiguresExtractor.extract().
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        figures_out = output_dir / "figures" / pdf_path.stem
        figures_out.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        results: list[dict] = []
        fig_counter = 1
        dpi = IMAGE_DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)

        try:
            for page_idx in range(doc.page_count):
                page = doc[page_idx]
                page_num = page_idx + 1

                image_list = page.get_images(full=True)
                page_has_raster = False

                for img_info in image_list:
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue
                    rect = rects[0]

                    # Filter tiny images (logos, icons)
                    area_px = rect.width * rect.height * (dpi / 72) ** 2
                    if area_px < MIN_FIGURE_AREA_PX:
                        continue

                    page_has_raster = True

                    # Expand clip by 50px and clamp to page
                    pad = 50 * (72 / dpi)
                    clip = fitz.Rect(
                        rect.x0 - pad, rect.y1 - pad,
                        rect.x1 + pad, rect.y1 + pad,
                    )
                    clip &= page.rect

                    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                    img_path = figures_out / f"fig_{fig_counter:03d}_full.png"
                    pix.save(str(img_path))

                    # Caption: look ~80pt below image
                    caption = self._find_caption_below(page, rect)

                    logger.info(
                        "PyMuPDF → figure %d | page %d | %.0fx%.0f px | %s",
                        fig_counter, page_num,
                        pix.width, pix.height, img_path.name,
                    )

                    results.append({
                        "figure_number": str(fig_counter),
                        "caption": caption,
                        "page": page_num,
                        "image_path": str(img_path),
                        "bounding_box": [rect.x0, rect.y0, rect.x1, rect.y1],
                        "source": "pymupdf",
                    })
                    fig_counter += 1

                # Vector-graphics detection: if no raster images but many short
                # line segments, the whole page is likely a vector figure.
                if not page_has_raster:
                    paths = page.get_drawings()
                    if len(paths) > 30:  # heuristic threshold
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        img_path = figures_out / f"fig_{fig_counter:03d}_full.png"
                        pix.save(str(img_path))

                        logger.info(
                            "PyMuPDF vector → page %d (%d paths) | %s",
                            page_num, len(paths), img_path.name,
                        )
                        results.append({
                            "figure_number": str(fig_counter),
                            "caption": "",
                            "page": page_num,
                            "image_path": str(img_path),
                            "bounding_box": [0, 0, page.rect.width, page.rect.height],
                            "source": "pymupdf",
                            "needs_segmentation": True,
                        })
                        fig_counter += 1
        finally:
            doc.close()

        logger.info("PyMuPDF extracted %d figures from %s", len(results), pdf_path.name)
        return results

    @staticmethod
    def _find_caption_below(page: fitz.Page, rect: fitz.Rect) -> str:
        """Return caption text found below the figure bounding box."""
        search_rect = fitz.Rect(rect.x0, rect.y1, rect.x1, rect.y1 + 80)
        search_rect &= page.rect
        blocks = page.get_text("blocks", clip=search_rect)
        for block in sorted(blocks, key=lambda b: b[1]):  # sort by y0
            text = block[4].strip()
            if _CAPTION_START_RE.match(text):
                return text
        # Fallback: return any text in the area
        return page.get_text("text", clip=search_rect).strip()

    def detect_panels(self, figure_image_path: str | Path, caption: str) -> list[dict]:
        """Split a multi-panel figure into sub-panel images.

        Uses the caption to detect panel labels (A, B, C…), then uses OpenCV
        to find large horizontal/vertical white-space gaps to split the image.

        Parameters
        ----------
        figure_image_path:
            Path to the full figure PNG.
        caption:
            The figure caption text (used to detect panel labels).

        Returns
        -------
        list[dict]
            [{"panel_label": "A", "image_path": str, "caption_segment": str}, …]
            If no multi-panel structure detected, returns a single-item list
            wrapping the original figure.
        """
        import cv2  # type: ignore
        import numpy as np
        from PIL import Image

        figure_image_path = Path(figure_image_path)

        # Detect panel labels from caption
        panel_labels = sorted(set(
            (m.group(1) or m.group(2) or m.group(3)).upper()
            for m in _PANEL_LABEL_RE.finditer(caption)
        ))
        expected_n = len(panel_labels)

        single_panel = [{
            "panel_label": "",
            "image_path": str(figure_image_path),
            "caption_segment": caption,
        }]

        if expected_n < 2:
            return single_panel

        try:
            img_pil = Image.open(figure_image_path).convert("RGB")
            img_np = np.array(img_pil)
        except Exception as exc:
            logger.warning("Cannot open figure for panel detection: %s", exc)
            return single_panel

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # Threshold: find near-white rows/columns (white-space gaps)
        _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)

        output_dir = figure_image_path.parent
        stem = figure_image_path.stem  # e.g. "fig_001_full"

        # --- Try horizontal split first (panels stacked vertically) ---
        row_sums = thresh.sum(axis=1)  # sum per row (max = 255*w)
        white_rows = np.where(row_sums >= 0.98 * 255 * w)[0]
        h_cuts = _find_cuts(white_rows, h, expected_n - 1)

        if len(h_cuts) == expected_n - 1:
            panels = _slice_panels(img_np, "horizontal", h_cuts, h, w)
        else:
            # --- Try vertical split (panels side by side) ---
            col_sums = thresh.sum(axis=0)
            white_cols = np.where(col_sums >= 0.98 * 255 * h)[0]
            v_cuts = _find_cuts(white_cols, w, expected_n - 1)
            if len(v_cuts) == expected_n - 1:
                panels = _slice_panels(img_np, "vertical", v_cuts, h, w)
            else:
                logger.debug(
                    "Could not split %d panels for %s — returning full image",
                    expected_n, figure_image_path.name,
                )
                return single_panel

        results = []
        for i, (label, panel_arr) in enumerate(zip(panel_labels, panels)):
            out_path = output_dir / f"{stem.replace('_full', '')}_{label}.png"
            Image.fromarray(panel_arr).save(out_path)
            results.append({
                "panel_label": label,
                "image_path": str(out_path),
                "caption_segment": _extract_panel_caption(caption, label),
            })
            logger.info("Saved panel %s → %s", label, out_path.name)

        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_cuts(white_indices: "np.ndarray", total: int, n_cuts: int) -> list[int]:
    """Find n_cuts evenly-spread cut positions from contiguous white bands."""
    if len(white_indices) == 0 or n_cuts <= 0:
        return []
    # Group contiguous white pixels into bands
    bands: list[tuple[int, int]] = []
    start = white_indices[0]
    prev = white_indices[0]
    for idx in white_indices[1:]:
        if idx - prev > 5:
            bands.append((start, prev))
            start = idx
        prev = idx
    bands.append((start, prev))

    if len(bands) < n_cuts:
        return []

    # Pick the widest bands as cut positions
    bands_sorted = sorted(bands, key=lambda b: b[1] - b[0], reverse=True)
    chosen = sorted(bands_sorted[:n_cuts], key=lambda b: b[0])
    return [(b[0] + b[1]) // 2 for b in chosen]


def _slice_panels(
    img: "np.ndarray",
    direction: str,
    cuts: list[int],
    h: int,
    w: int,
) -> list["np.ndarray"]:
    """Slice image into sub-panels given cut positions."""
    positions = [0] + cuts + [h if direction == "horizontal" else w]
    panels = []
    for a, b in zip(positions, positions[1:]):
        if direction == "horizontal":
            panels.append(img[a:b, :])
        else:
            panels.append(img[:, a:b])
    return panels


def _extract_panel_caption(caption: str, label: str) -> str:
    """Attempt to extract the sub-caption for a given panel label."""
    pattern = re.compile(
        rf'[\(\[]{re.escape(label)}[\)\]].*?(?=[\(\[][A-H][\)\]]|$)',
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(caption)
    return m.group(0).strip() if m else caption


def _render_pdf_region(
    pdf_path: Path,
    page_number: int,
    bb: dict,
    dest_path: Path,
    dpi: int = IMAGE_DPI,
) -> Optional[Path]:
    """Render a bounding-box region from a PDF page to a PNG file."""
    try:
        doc = fitz.open(str(pdf_path))
        if page_number >= doc.page_count:
            doc.close()
            return None
        page = doc[page_number]
        rect = fitz.Rect(bb.get("x1", 0), bb.get("y1", 0), bb.get("x2", 1), bb.get("y2", 1))
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
        pix.save(str(dest_path))
        doc.close()
        return dest_path
    except Exception as exc:
        logger.warning("Bbox region render failed: %s", exc)
        return None


def _validate_image(path: Optional[Path]) -> bool:
    """Return True if path points to a valid image larger than 100×100 px."""
    if path is None or not path.exists():
        return False
    try:
        from PIL import Image
        with Image.open(path) as img:
            w, h = img.size
            return w > 100 and h > 100
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Strategy Combiner — public entrypoint
# ---------------------------------------------------------------------------

class FigureExtractor:
    """Main figure extraction class used by the FigureVault pipeline.

    Tries PDFFigures2 first; falls back to PyMuPDF if unavailable or empty.
    Runs multi-panel detection on every extracted figure.
    """

    def __init__(self) -> None:
        self._p2 = PDFFiguresExtractor()
        self._pymupdf = PyMuPDFFigureExtractor()

    def extract_all(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        page_renders: Optional[list[str]] = None,
    ) -> list[FigureRecord]:
        """Extract all figures from a PDF.

        Parameters
        ----------
        pdf_path:
            Path to the PDF file.
        output_dir:
            Root directory for outputs.
        page_renders:
            Pre-rendered page PNGs (optional, passed to PyMuPDF extractor).

        Returns
        -------
        list[FigureRecord]
            One record per figure (or sub-panel).
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        # Choose strategy
        if self._p2.is_available():
            logger.info("Using PDFFigures2 (primary strategy)")
            try:
                raw = self._p2.extract(pdf_path, output_dir)
                if raw:
                    logger.info("PDFFigures2 extracted %d figures", len(raw))
                    source = "pdffigures2"
                else:
                    logger.info("PDFFigures2 returned 0 figures — falling back to PyMuPDF")
                    raw = self._pymupdf.extract(pdf_path, output_dir, page_renders)
                    source = "pymupdf"
            except Exception as exc:
                logger.warning("PDFFigures2 failed (%s) — falling back to PyMuPDF", exc)
                raw = self._pymupdf.extract(pdf_path, output_dir, page_renders)
                source = "pymupdf"
        else:
            logger.info("Using PyMuPDF heuristic (PDFFigures2 unavailable)")
            raw = self._pymupdf.extract(pdf_path, output_dir, page_renders)
            source = "pymupdf"

        records: list[FigureRecord] = []

        for fig in raw:
            img_path = Path(fig["image_path"]) if fig.get("image_path") else None
            if not _validate_image(img_path):
                logger.debug(
                    "Skipping figure %s — image missing or too small", fig["figure_number"]
                )
                continue

            bb_raw = fig.get("bounding_box", [])
            if isinstance(bb_raw, list) and len(bb_raw) == 4:
                bounding_box = {
                    "x": bb_raw[0], "y": bb_raw[1],
                    "w": bb_raw[2] - bb_raw[0], "h": bb_raw[3] - bb_raw[1],
                }
            else:
                bounding_box = bb_raw if isinstance(bb_raw, dict) else {}

            caption = fig.get("caption", "")
            confidence = 0.95 if source == "pdffigures2" else 0.8

            # Multi-panel detection
            panels = self._pymupdf.detect_panels(img_path, caption)

            if len(panels) <= 1:
                records.append(FigureRecord(
                    page_number=fig.get("page", 0),
                    figure_number=fig["figure_number"],
                    panel_label="",
                    caption=caption,
                    image_path=img_path,
                    bounding_box=bounding_box,
                    source=source,
                    confidence=confidence,
                ))
            else:
                for panel in panels:
                    panel_path = Path(panel["image_path"])
                    if not _validate_image(panel_path):
                        continue
                    records.append(FigureRecord(
                        page_number=fig.get("page", 0),
                        figure_number=fig["figure_number"],
                        panel_label=panel["panel_label"],
                        caption=panel.get("caption_segment", caption),
                        image_path=panel_path,
                        bounding_box=bounding_box,
                        source=source,
                        confidence=confidence * 0.95,  # slight penalty for sub-panel
                    ))

        # Write metadata JSON
        self._save_metadata(pdf_path, output_dir, records)

        logger.info(
            "FigureExtractor: %d total records from %s (source: %s)",
            len(records), pdf_path.name, source,
        )
        return records

    # ------------------------------------------------------------------
    # Legacy compatibility (used by existing code/tests)
    # ------------------------------------------------------------------

    def extract(self) -> list[FigureRecord]:
        """Backwards-compatible single-argument extract.

        Requires self.pdf_path and self.output_dir to be set by __init__.
        """
        if not hasattr(self, "pdf_path"):
            raise AttributeError(
                "Use FigureExtractor().extract_all(pdf_path, output_dir) instead"
            )
        return self.extract_all(self.pdf_path, self.output_dir)

    @staticmethod
    def standardize_figure_data(raw_figures: list[dict]) -> list[dict]:
        """Ensure all figure dicts share the same schema."""
        out = []
        for fig in raw_figures:
            bb = fig.get("bounding_box", [])
            if isinstance(bb, list) and len(bb) == 4:
                bb_dict = {"x": bb[0], "y": bb[1], "w": bb[2] - bb[0], "h": bb[3] - bb[1]}
            else:
                bb_dict = bb if isinstance(bb, dict) else {}

            out.append({
                "figure_number": str(fig.get("figure_number", "")),
                "panel_label": fig.get("panel_label", ""),
                "caption": fig.get("caption", ""),
                "page": int(fig.get("page", 0)),
                "image_path": fig.get("image_path"),
                "bounding_box": bb_dict,
                "source": fig.get("source", "unknown"),
                "figure_type": fig.get("figure_type"),
                "confidence": float(
                    0.95 if fig.get("source") == "pdffigures2" else 0.8
                ),
            })
        return out

    @staticmethod
    def _save_metadata(
        pdf_path: Path,
        output_dir: Path,
        records: list[FigureRecord],
    ) -> None:
        metadata_dir = output_dir / "metadata" / pdf_path.stem
        metadata_dir.mkdir(parents=True, exist_ok=True)
        meta_file = metadata_dir / "figures.json"
        data = [r.to_dict() for r in records]
        meta_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Metadata written to %s", meta_file)
