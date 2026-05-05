"""
pipeline/context_builder.py — Phase 4: Rich Context Assembly

Before each figure is sent to the data extractor (Phase 5), this module
assembles the richest possible context package:

  • Figure image path + caption
  • In-text references (± 300 chars around every "Figure X" mention)
  • Methods section text relevant to this figure
  • Results section text that directly references this figure
  • A single, token-efficient context_summary string injected into the Gemma4 prompt

This is the key quality differentiator: Gemma4 with richer context extracts
better data because it understands WHAT the numbers represent, not just what
they look like.

Backward-compatible API:
  ContextBuilder(paper_meta)          ← old usage (still works)
  ContextBuilder(max_context_chars=3000)   ← new standalone usage with PDFParser
  ctx = builder.build(figure_record)  ← returns PromptContext (old API)
  ctx_dict = builder.build_context(figure_dict, pdf_parser)  ← new rich API
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_SURROUNDING_CHARS = 1500
_CONTEXT_WINDOW = 300        # chars either side of a fig ref
_MAX_REFS = 3                # max in-text references to keep
_MAX_SECTION_CHARS = 1000    # cap on methods / results text
_MIN_REF_UNIQUE_CHARS = 50   # drop refs that are too similar to a kept one

# Section name aliases used in extract_section_text() output
_METHODS_KEYS  = {"methods", "method", "materials and methods", "materials & methods",
                  "experimental", "experimental section"}
_RESULTS_KEYS  = {"results", "results and discussion"}

# Regex for figure references: "Figure 3", "Fig. 3", "Fig 3A", "Figure S2"
_FIG_REF_RE = re.compile(
    r"\b(?:Fig(?:ure|s?\.?)?)\.?\s*([A-Z]?(?:S\s*)?\d+\s*[a-z]?"
    r"(?:\s*[–\-]\s*\d+)?)\b",
    re.IGNORECASE,
)

# Matches simple "content" words (nouns / descriptors) from captions
_WORD_RE = re.compile(r"\b[a-zA-Z]{4,}\b")
# Stop-words to exclude from caption keyword extraction
_STOP_WORDS = {
    "figure", "shows", "shown", "panel", "panels", "data", "with", "from",
    "each", "this", "that", "their", "were", "have", "been", "also", "upon",
    "using", "used", "left", "right", "upper", "lower", "scale", "bars",
    "images", "image", "error", "bars", "mean", "median", "total", "values",
}


# ---------------------------------------------------------------------------
# PromptContext dataclass (legacy + extended)
# ---------------------------------------------------------------------------

@dataclass
class PromptContext:
    """All information needed to prompt Gemma4 for data extraction."""

    # Core fields (legacy)
    figure: "FigureRecord"   # type: ignore[name-defined]
    surrounding_text: str = ""
    paper_title: Optional[str] = None
    doi: Optional[str] = None

    # Rich Phase-4 extensions
    in_text_references: list[str] = field(default_factory=list)
    methods_context: str = ""
    results_context: str = ""
    context_summary: str = ""

    def build_extraction_prompt(self) -> str:
        """Render the full extraction prompt string sent to Gemma4."""
        lines = [
            f"Paper: {self.paper_title or 'Unknown'}",
            f"DOI: {self.doi or 'Unknown'}",
            f"Figure {self.figure.figure_number}"
            + (f" ({self.figure.figure_type})" if getattr(self.figure, "figure_type", None) else ""),
            "",
        ]

        # Prefer rich context_summary; fall back to surrounding_text
        context_block = self.context_summary or self.surrounding_text
        if context_block:
            lines += ["=== CONTEXT ===", context_block, ""]
        else:
            lines += [
                "=== CAPTION ===",
                self.figure.caption or "(no caption available)",
                "",
            ]

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


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """Assemble rich prompt context for each figure before data extraction.

    Supports two usage styles:

    **Legacy** (paper metadata already parsed):
        builder = ContextBuilder(paper_meta)
        ctx = builder.build(figure_record)   # → PromptContext

    **Rich / Phase-4** (PDFParser available):
        builder = ContextBuilder(max_context_chars=3000)
        ctx_dict = builder.build_context(figure_dict, pdf_parser)
        ctxs = builder.build_batch_contexts(figures, pdf_parser)

    Both styles can be mixed; the ``build`` method also accepts a
    ``pdf_parser`` keyword argument to enable the rich path.
    """

    def __init__(
        self,
        paper_meta=None,         # PaperMetadata | None
        window_chars: int = _MAX_SURROUNDING_CHARS,
        max_context_chars: int = 3000,
    ) -> None:
        self.meta = paper_meta
        self.window_chars = window_chars
        self.max_context_chars = max_context_chars
        self._section_cache: dict = {}   # keyed by pdf path

    # ------------------------------------------------------------------
    # Legacy public API (PromptContext-based)
    # ------------------------------------------------------------------

    def build(
        self,
        figure,                        # FigureRecord | dict
        pdf_parser=None,               # PDFParser | None — enables rich path
    ) -> PromptContext:
        """Build a PromptContext for a single figure.

        If *pdf_parser* is supplied the full rich context (in-text refs,
        Methods, Results) is assembled.  Otherwise only surrounding text
        from self.meta.page_texts is used.
        """
        from pipeline.figure_extractor import FigureRecord  # local import avoids circularity

        # Normalise input to a FigureRecord
        if isinstance(figure, dict):
            img_path = figure.get("image_path")
            fig_obj = FigureRecord(
                page_number=figure.get("page", 0),
                figure_number=figure.get("figure_number", ""),
                panel_label=figure.get("panel_label", ""),
                caption=figure.get("caption", ""),
                figure_type=(
                    figure.get("classification", {}).get("figure_type")
                    or figure.get("figure_type")
                ),
                image_path=Path(img_path) if img_path else None,
                bounding_box=figure.get("bounding_box", {}),
                source=figure.get("source", ""),
                confidence=figure.get("confidence", 1.0),
            )
        else:
            fig_obj = figure

        # ---- Rich path (PDFParser supplied) -----
        if pdf_parser is not None:
            ctx_dict = self.build_context(
                {
                    "figure_number": fig_obj.figure_number,
                    "caption": fig_obj.caption or "",
                    "page": fig_obj.page_number,
                    "image_path": str(fig_obj.image_path) if fig_obj.image_path else "",
                    "panel_label": fig_obj.panel_label or "",
                },
                pdf_parser,
            )
            return PromptContext(
                figure=fig_obj,
                surrounding_text=ctx_dict.get("context_summary", ""),
                paper_title=ctx_dict.get("paper_title"),
                doi=ctx_dict.get("paper_doi"),
                in_text_references=ctx_dict.get("in_text_references", []),
                methods_context=ctx_dict.get("methods_context", ""),
                results_context=ctx_dict.get("results_context", ""),
                context_summary=ctx_dict.get("context_summary", ""),
            )

        # ---- Legacy path (PaperMetadata from self.meta) -----
        surrounding = self._find_surrounding_text(fig_obj)
        paper_title = self.meta.title if self.meta else None
        doi = self.meta.doi if self.meta else None

        ctx = PromptContext(
            figure=fig_obj,
            surrounding_text=surrounding,
            paper_title=paper_title,
            doi=doi,
            context_summary=surrounding,
        )
        logger.debug(
            "Built context for figure %s (surrounding_chars=%d)",
            fig_obj.figure_number, len(surrounding),
        )
        return ctx

    def build_all(self, figures: list, pdf_parser=None) -> list[PromptContext]:
        """Build PromptContexts for every figure in a list."""
        return [self.build(fig, pdf_parser=pdf_parser) for fig in figures]

    # ------------------------------------------------------------------
    # Rich Phase-4 public API (dict-based)
    # ------------------------------------------------------------------

    def build_context(self, figure: dict, pdf_parser) -> dict:
        """Build the full rich context dict for a single figure.

        Parameters
        ----------
        figure : dict
            Must have: ``figure_number``, ``caption``.
            Optional: ``page``, ``image_path``, ``panel_label``.
        pdf_parser : PDFParser
            An already-opened PDFParser for the paper.

        Returns
        -------
        dict with keys:
            image_path, caption, figure_number, panel_label,
            in_text_references, methods_context, results_context,
            paper_title, paper_doi, context_summary
        """
        fig_num = str(figure.get("figure_number", "")).strip()
        caption = figure.get("caption", "")

        # Ensure sections are cached
        self._warm_section_cache(pdf_parser)

        in_text_refs = self.find_in_text_references(fig_num, pdf_parser)
        methods_ctx  = self.get_methods_context(fig_num, caption, pdf_parser)
        results_ctx  = self.get_results_context(fig_num, pdf_parser)

        summary = self.build_context_summary(
            caption, in_text_refs, methods_ctx, results_ctx
        )

        # Try to get paper metadata from the parser if available
        paper_title: Optional[str] = None
        paper_doi:   Optional[str] = None
        try:
            meta = self._get_meta(pdf_parser)
            paper_title = meta.get("title") if isinstance(meta, dict) else getattr(meta, "title", None)
            paper_doi   = meta.get("doi")   if isinstance(meta, dict) else getattr(meta, "doi",   None)
        except Exception:
            pass

        return {
            "image_path":          figure.get("image_path", ""),
            "caption":             caption,
            "figure_number":       fig_num,
            "panel_label":         figure.get("panel_label"),
            "in_text_references":  in_text_refs,
            "methods_context":     methods_ctx,
            "results_context":     results_ctx,
            "paper_title":         paper_title,
            "paper_doi":           paper_doi,
            "context_summary":     summary,
        }

    def build_batch_contexts(self, figures: list[dict], pdf_parser) -> list[dict]:
        """Build rich context dicts for every figure.  Shows a tqdm progress bar."""
        results = []
        for fig in tqdm(figures, desc="Building contexts", unit="fig"):
            try:
                results.append(self.build_context(fig, pdf_parser))
            except Exception as exc:
                logger.error(
                    "build_batch_contexts: failed for figure %s: %s",
                    fig.get("figure_number", "?"), exc,
                )
                results.append(self._empty_context(fig))
        return results

    # ------------------------------------------------------------------
    # Sub-steps (public so callers can use them directly)
    # ------------------------------------------------------------------

    def find_in_text_references(
        self, figure_number: str, pdf_parser
    ) -> list[str]:
        """Return up to _MAX_REFS distinct in-text snippets mentioning *figure_number*.

        Searches for variants: "Figure 1", "Fig. 1", "Fig 1A", etc.
        Extracts ±_CONTEXT_WINDOW chars around each match.  Keeps only the
        longest non-redundant snippets.
        """
        full_text = self._get_full_text(pdf_parser)
        if not full_text or not figure_number:
            return []

        # Build a regex that matches any variant of this figure number
        esc = re.escape(figure_number.strip())
        pattern = re.compile(
            rf"\b(?:Fig(?:ure|s?\.?)?)\.?\s*{esc}\b",
            re.IGNORECASE,
        )

        snippets: list[str] = []
        for m in pattern.finditer(full_text):
            start = max(0, m.start() - _CONTEXT_WINDOW)
            end   = min(len(full_text), m.end() + _CONTEXT_WINDOW)
            snippet = full_text[start:end].strip().replace("\n", " ")
            snippets.append(snippet)

        if not snippets:
            return []

        # De-duplicate and rank by length (most informative first)
        snippets.sort(key=len, reverse=True)
        kept: list[str] = []
        for s in snippets:
            # Skip if very similar to an already-kept snippet
            if any(
                self._overlap_ratio(s, k) > 0.6
                for k in kept
            ):
                continue
            kept.append(s)
            if len(kept) >= _MAX_REFS:
                break

        return kept

    def get_methods_context(
        self, figure_number: str, caption: str, pdf_parser
    ) -> str:
        """Return up to _MAX_SECTION_CHARS from the Methods section most
        relevant to this figure.

        Strategy:
        1. Extract keywords from the caption (content nouns, 4+ chars).
        2. Score each sentence in Methods by keyword hits + figure-ref match.
        3. Return the top sentences up to the char limit.
        """
        methods_text = self._get_section(pdf_parser, _METHODS_KEYS)
        if not methods_text:
            return ""

        keywords = self._extract_caption_keywords(caption)
        fig_re = self._fig_num_re(figure_number)

        return self._score_and_trim(methods_text, keywords, fig_re, _MAX_SECTION_CHARS)

    def get_results_context(self, figure_number: str, pdf_parser) -> str:
        """Return up to _MAX_SECTION_CHARS from the Results section that
        directly mentions this figure.
        """
        results_text = self._get_section(pdf_parser, _RESULTS_KEYS)
        if not results_text:
            return ""

        fig_re = self._fig_num_re(figure_number)
        # Split into paragraphs and keep only those that mention the figure
        paragraphs = re.split(r"\n{2,}", results_text)
        relevant = [p.strip() for p in paragraphs if fig_re.search(p)]

        if not relevant:
            # Fall back: first 1000 chars of Results
            return results_text[:_MAX_SECTION_CHARS].strip()

        combined = "\n\n".join(relevant)
        return combined[:_MAX_SECTION_CHARS].strip()

    def build_context_summary(
        self,
        caption: str,
        in_text_references: list[str],
        methods_context: str,
        results_context: str,
    ) -> str:
        """Combine all context pieces into a single string for the model prompt.

        Total length is capped at ``self.max_context_chars``.
        """
        parts: list[str] = []

        if caption:
            parts.append(f"FIGURE CAPTION: {caption}")

        if in_text_references:
            ref_str = "; ".join(in_text_references[:2])
            parts.append(f"IN-TEXT DESCRIPTION: {ref_str}")

        if methods_context:
            parts.append(f"METHODS CONTEXT: {methods_context[:500]}")

        if results_context:
            parts.append(f"RESULTS CONTEXT: {results_context[:500]}")

        summary = "\n\n".join(parts)
        return summary[: self.max_context_chars].strip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_surrounding_text(self, figure: "FigureRecord") -> str:  # type: ignore[name-defined]
        """Legacy helper: search self.meta.page_texts for mentions of this figure."""
        if self.meta is None:
            return ""

        fig_num = figure.figure_number
        patterns = [
            rf"\bFig(?:ure)?\.?\s*{re.escape(fig_num)}\b",
            rf"\bFigure\s+{re.escape(fig_num)}\b",
        ]
        combined_re = re.compile("|".join(patterns), re.IGNORECASE)

        page_texts = self.meta.page_texts
        start_page = max(0, figure.page_number - 2)
        search_text = "\n".join(page_texts[start_page: figure.page_number + 1])

        match = combined_re.search(search_text)
        if match:
            start = max(0, match.start() - self.window_chars // 2)
            end   = min(len(search_text), match.end() + self.window_chars // 2)
            return search_text[start:end].strip()

        if figure.page_number > 0 and figure.page_number <= len(page_texts):
            page_text = page_texts[figure.page_number - 1]
            return page_text[: self.window_chars].strip()

        return ""

    def _warm_section_cache(self, pdf_parser) -> None:
        """Populate self._section_cache from pdf_parser if not yet done."""
        cache_key = getattr(pdf_parser, "pdf_path", id(pdf_parser))
        if cache_key in self._section_cache:
            return
        try:
            sections = pdf_parser.extract_section_text()
            full_text_pages = self._get_full_text(pdf_parser)
        except Exception as exc:
            logger.warning("Could not extract sections: %s", exc)
            sections = {}
            full_text_pages = ""

        self._section_cache[cache_key] = {
            "sections": sections,
            "full_text": full_text_pages,
        }

    def _get_section(self, pdf_parser, keys: set[str]) -> str:
        """Return the text of the first matching section (case-insensitive key lookup)."""
        cache_key = getattr(pdf_parser, "pdf_path", id(pdf_parser))
        if cache_key not in self._section_cache:
            self._warm_section_cache(pdf_parser)

        sections: dict = self._section_cache[cache_key].get("sections", {})
        for section_name, text in sections.items():
            if section_name.lower() in keys:
                return text
        return ""

    def _get_full_text(self, pdf_parser) -> str:
        """Return cached full-text string for the paper."""
        cache_key = getattr(pdf_parser, "pdf_path", id(pdf_parser))
        if cache_key not in self._section_cache:
            self._warm_section_cache(pdf_parser)
        return self._section_cache[cache_key].get("full_text", "")

    def _get_meta(self, pdf_parser) -> dict:
        """Extract lightweight metadata dict from the parser."""
        if hasattr(pdf_parser, "extract_metadata"):
            return pdf_parser.extract_metadata()
        if self.meta is not None:
            return {"title": self.meta.title, "doi": self.meta.doi}
        return {}

    @staticmethod
    def _fig_num_re(figure_number: str) -> re.Pattern:
        """Compile a pattern matching all variants of *figure_number*."""
        esc = re.escape(figure_number.strip())
        return re.compile(
            rf"\b(?:Fig(?:ure|s?\.?)?)\.?\s*{esc}\b",
            re.IGNORECASE,
        )

    @staticmethod
    def _extract_caption_keywords(caption: str) -> set[str]:
        """Extract meaningful content words from a caption string."""
        words = _WORD_RE.findall(caption.lower())
        return {w for w in words if w not in _STOP_WORDS}

    @staticmethod
    def _score_and_trim(
        text: str,
        keywords: set[str],
        fig_re: re.Pattern,
        max_chars: int,
    ) -> str:
        """Score sentences by keyword overlap and figure-ref presence, return top ones."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        scored: list[tuple[int, str]] = []
        for sent in sentences:
            s_lower = sent.lower()
            kw_hits = sum(1 for k in keywords if k in s_lower)
            fig_hit = 2 if fig_re.search(sent) else 0
            score = kw_hits + fig_hit
            if score > 0:
                scored.append((score, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        result_parts: list[str] = []
        total = 0
        for _, sent in scored:
            if total + len(sent) > max_chars:
                break
            result_parts.append(sent)
            total += len(sent)

        return " ".join(result_parts).strip()

    @staticmethod
    def _overlap_ratio(a: str, b: str) -> float:
        """Estimate the character overlap ratio between two strings."""
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if not longer:
            return 0.0
        hits = sum(1 for ch in shorter if ch in longer)
        return hits / len(longer)

    @staticmethod
    def _empty_context(figure: dict) -> dict:
        return {
            "image_path":          figure.get("image_path", ""),
            "caption":             figure.get("caption", ""),
            "figure_number":       figure.get("figure_number", ""),
            "panel_label":         figure.get("panel_label"),
            "in_text_references":  [],
            "methods_context":     "",
            "results_context":     "",
            "paper_title":         None,
            "paper_doi":           None,
            "context_summary":     figure.get("caption", ""),
        }
