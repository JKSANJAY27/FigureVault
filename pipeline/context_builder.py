"""
pipeline/context_builder.py — Phase 4: Context assembly for the extraction model

Combines:
  • Figure image path
  • Figure caption
  • Surrounding paper text (paragraphs referencing the figure)
  • Figure number and type label

Produces a PromptContext object consumed by the Phase 5 extractor.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from pipeline.figure_extractor import FigureRecord
from pipeline.pdf_parser import PaperMetadata

logger = logging.getLogger(__name__)

# Max characters of surrounding text to include in the prompt
_MAX_SURROUNDING_CHARS = 1500


@dataclass
class PromptContext:
    """All information needed to prompt Gemma4 for data extraction."""
    figure: FigureRecord
    surrounding_text: str = ""
    paper_title: Optional[str] = None
    doi: Optional[str] = None

    def build_extraction_prompt(self) -> str:
        """Render the full extraction prompt string."""
        lines = [
            f"Paper: {self.paper_title or 'Unknown'}",
            f"DOI: {self.doi or 'Unknown'}",
            f"Figure {self.figure.figure_number}"
            + (f" ({self.figure.figure_type})" if getattr(self.figure, 'figure_type', None) else ""),
            "",
            "=== CAPTION ===",
            self.figure.caption or "(no caption available)",
            "",
        ]
        if self.surrounding_text:
            lines += ["=== SURROUNDING TEXT ===", self.surrounding_text, ""]
        lines += [
            "=== TASK ===",
            "Extract ALL numerical data from this figure image.",
            "For each data series output a JSON object with keys:",
            "  series_name, x_label, x_unit, y_label, y_unit,",
            "  axis_scale_x (linear|log), axis_scale_y (linear|log),",
            "  data_points ([{x, y, err_x, err_y},...]),",
            "  statistical_annotations ([{marker, x, p_value},...]).",
            "Output a JSON array of series objects. No extra text.",
        ]
        return "\n".join(lines)


class ContextBuilder:
    """Assemble prompt context for each figure from paper metadata.

    Parameters
    ----------
    paper_meta : PaperMetadata
        Parsed paper metadata including per-page text.
    window_chars : int
        How many characters of surrounding text to include.
    """

    def __init__(
        self,
        paper_meta: PaperMetadata,
        window_chars: int = _MAX_SURROUNDING_CHARS,
    ) -> None:
        self.meta = paper_meta
        self.window_chars = window_chars

    def build(self, figure: FigureRecord | dict) -> PromptContext:
        """Build a PromptContext for a single figure.

        Parameters
        ----------
        figure : FigureRecord or dict

        Returns
        -------
        PromptContext
        """
        if isinstance(figure, dict):
            from pathlib import Path
            img_path = figure.get("image_path")
            fig_obj = FigureRecord(
                page_number=figure.get("page", 0),
                figure_number=figure.get("figure_number", ""),
                panel_label=figure.get("panel_label", ""),
                caption=figure.get("caption", ""),
                figure_type=figure.get("classification", {}).get("figure_type") or figure.get("figure_type"),
                image_path=Path(img_path) if img_path else None,
                bounding_box=figure.get("bounding_box", {}),
                source=figure.get("source", ""),
                confidence=figure.get("confidence", 1.0)
            )
        else:
            fig_obj = figure

        surrounding = self._find_surrounding_text(fig_obj)
        ctx = PromptContext(
            figure=fig_obj,
            surrounding_text=surrounding,
            paper_title=self.meta.title,
            doi=self.meta.doi,
        )
        logger.debug(
            "Built context for figure %s (surrounding_chars=%d)",
            fig_obj.figure_number, len(surrounding),
        )
        return ctx

    def build_all(self, figures: list[FigureRecord | dict]) -> list[PromptContext]:
        """Build prompt contexts for every figure in a list."""
        return [self.build(fig) for fig in figures]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_surrounding_text(self, figure: FigureRecord) -> str:
        """Search the paper text for paragraphs that mention this figure."""
        fig_num = figure.figure_number

        # Patterns like "Figure 3", "Fig. 3", "Fig 3", "Fig. S2"
        patterns = [
            rf"\bFig(?:ure)?\.?\s*{re.escape(fig_num)}\b",
            rf"\bFigure\s+{re.escape(fig_num)}\b",
        ]
        combined_re = re.compile("|".join(patterns), re.IGNORECASE)

        # Search from the page the figure is on, expanding outward
        page_texts = self.meta.page_texts
        start_page = max(0, figure.page_number - 2)
        search_text = "\n".join(page_texts[start_page : figure.page_number + 1])

        match = combined_re.search(search_text)
        if match:
            start = max(0, match.start() - self.window_chars // 2)
            end = min(len(search_text), match.end() + self.window_chars // 2)
            return search_text[start:end].strip()

        # Fall back: return text from the figure's page
        if figure.page_number > 0 and figure.page_number <= len(page_texts):
            page_text = page_texts[figure.page_number - 1]
            return page_text[: self.window_chars].strip()

        return ""
