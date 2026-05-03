"""
pipeline/extractor.py — Phase 5: Main data extraction via Gemma4

The core of FigureVault. Takes a PromptContext (figure image + caption +
surrounding text) and asks Gemma4 to output all quantitative data as JSON.

Returns a list of ExtractedSeries objects (one per data series in the figure).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from config import CONFIDENCE_THRESHOLD
from models.ollama_client import OllamaClient
from pipeline.context_builder import PromptContext

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a scientific data extraction expert.
Given a figure image from a scientific paper along with its caption and context,
extract ALL numerical data from the figure.

IMPORTANT RULES:
1. Output ONLY a valid JSON array. No markdown, no explanation, no extra text.
2. Each element represents one data series.
3. For bar charts: each bar group or bar is one series.
4. For line plots: each line is one series.
5. For scatter plots: each point cloud is one series.
6. data_points must be an array of {x, y, err_x, err_y} objects (null if unknown).
7. axis_scale_x and axis_scale_y must be "linear" or "log".
8. Include a confidence score (0.0-1.0) for each series.

Output format (JSON array):
[
  {
    "series_name": "Control",
    "x_label": "Time",
    "x_unit": "min",
    "y_label": "Absorbance",
    "y_unit": "OD600",
    "axis_scale_x": "linear",
    "axis_scale_y": "linear",
    "data_points": [{"x": 0, "y": 0.1, "err_x": null, "err_y": 0.02}, ...],
    "statistical_annotations": [{"marker": "*", "x": 5.0, "p_value": 0.04}],
    "confidence": 0.85
  }
]
"""


@dataclass
class ExtractedSeries:
    """One data series extracted from a figure."""
    series_name: str = ""
    x_label: str = ""
    x_unit: str = ""
    y_label: str = ""
    y_unit: str = ""
    axis_scale_x: str = "linear"
    axis_scale_y: str = "linear"
    data_points: list[dict[str, Any]] = field(default_factory=list)
    statistical_annotations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    extraction_method: str = "gemma4_multimodal"


class DataExtractor:
    """Extract structured data from figures using Gemma4 multimodal inference.

    Parameters
    ----------
    client : OllamaClient, optional
        Reuse an existing client to avoid re-instantiation.
    confidence_threshold : float
        Series below this confidence are flagged but still returned.
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.client = client or OllamaClient()
        self.confidence_threshold = confidence_threshold

    def extract(self, ctx: PromptContext) -> list[ExtractedSeries]:
        """Extract data series from a single figure.

        Parameters
        ----------
        ctx : PromptContext
            Context assembled by Phase 4's ContextBuilder.

        Returns
        -------
        list[ExtractedSeries]
            One entry per detected data series.
        """
        if ctx.figure.image_path is None or not ctx.figure.image_path.exists():
            logger.warning("No image for figure %s — skipping extraction", ctx.figure.figure_number)
            return []

        prompt = ctx.build_extraction_prompt()

        try:
            raw = self.client.query_multimodal(
                prompt=prompt,
                image_path=ctx.figure.image_path,
                system=_SYSTEM_PROMPT,
                temperature=0.1,
            )
            series_list = self._parse_response(raw)
        except Exception as exc:
            logger.error("Extraction failed for figure %s: %s", ctx.figure.figure_number, exc)
            series_list = []

        logger.info(
            "Extracted %d series from figure %s",
            len(series_list), ctx.figure.figure_number,
        )
        return series_list

    def extract_batch(self, contexts: list[PromptContext]) -> dict[str, list[ExtractedSeries]]:
        """Extract data from multiple figures.

        Returns
        -------
        dict
            Mapping of figure_number → list of ExtractedSeries.
        """
        results: dict[str, list[ExtractedSeries]] = {}
        for ctx in contexts:
            fig_num = ctx.figure.figure_number
            results[fig_num] = self.extract(ctx)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> list[ExtractedSeries]:
        """Parse the model's JSON response into ExtractedSeries objects."""
        # Strip common markdown code fences the model sometimes adds
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        # Try to extract a JSON array from the output
        array_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not array_match:
            logger.warning("No JSON array found in model response")
            return []

        try:
            data = json.loads(array_match.group())
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse JSON array: %s", exc)
            return []

        series_out: list[ExtractedSeries] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            s = ExtractedSeries(
                series_name=str(item.get("series_name", "")),
                x_label=str(item.get("x_label", "")),
                x_unit=str(item.get("x_unit", "")),
                y_label=str(item.get("y_label", "")),
                y_unit=str(item.get("y_unit", "")),
                axis_scale_x=str(item.get("axis_scale_x", "linear")),
                axis_scale_y=str(item.get("axis_scale_y", "linear")),
                data_points=item.get("data_points", []),
                statistical_annotations=item.get("statistical_annotations", []),
                confidence=float(item.get("confidence", 0.5)),
            )
            series_out.append(s)

        return series_out
