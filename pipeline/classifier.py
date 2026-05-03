"""
pipeline/classifier.py — Phase 3: Figure Type Classification

Classifies each extracted figure image + caption into one of the FIGURE_TYPES
defined in config.py using Gemma4 E4B via Ollama multimodal inference.

The classification drives which specialized extractor is used downstream
(plot_digitizer, bar_extractor, spectrum_extractor, etc.).

Module-level constants CLASSIFICATION_SYSTEM_PROMPT and
CLASSIFICATION_USER_TEMPLATE are kept outside methods for easy iteration
during fine-tuning prompt engineering.

Usage:
    from models.ollama_client import OllamaClient
    from pipeline.classifier import FigureClassifier

    client = OllamaClient()
    clf = FigureClassifier(client)
    result = clf.classify("path/to/fig.png", "Figure 2A. Growth curves...")
    # → {"figure_type": "line_plot", "confidence": 0.94, ...}
"""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Optional

from tqdm import tqdm

from config import CONFIDENCE_THRESHOLD, FIGURE_TYPES
from models.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level prompt constants (easy to iterate for fine-tuning)
# ---------------------------------------------------------------------------

CLASSIFICATION_SYSTEM_PROMPT = """You are a scientific figure classification expert. \
You analyze figures from research papers and classify them precisely. \
You ALWAYS respond with valid JSON only, no other text."""


CLASSIFICATION_USER_TEMPLATE = """\
Figure caption: {caption}

Classify this scientific figure. Return ONLY this JSON with no other text:
{{
  "figure_type": "<type from list>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence explaining your classification>",
  "has_quantitative_data": <true|false>,
  "sub_types": ["<optional sub-classification>"]
}}

Valid figure_type values:
line_plot, bar_chart, scatter_plot, heatmap, spectrum_nmr, spectrum_ir, spectrum_ms,
western_blot, microscopy, flowchart, other

Rules:
- line_plot: line graph, time series, dose-response curves, kinetics plots
- bar_chart: bar graphs, column charts, grouped bars, stacked bars
- scatter_plot: XY plot WITHOUT connected lines; individual points only
- heatmap: grid of colored cells; correlation matrix, expression matrix, z-score
- spectrum_nmr: NMR spectrum — look for ppm on x-axis, sharp peaks (1H, 13C, 2D NMR)
- spectrum_ir: Infrared or Raman spectrum — wavenumber (cm⁻¹) on x-axis, broad bands
- spectrum_ms: Mass spectrum — m/z on x-axis, intensity on y-axis
- western_blot: horizontal dark bands on light/dark background, often labelled by molecular weight
- microscopy: Microscopy image (fluorescence, electron, brightfield); may have scale bar
- flowchart: Diagram, schematic, flowchart — no numerical axes, no quantitative data
- other: Pie chart, box plot, violin plot, survival curve, ROC curve, or unrecognizable

Additional guidance:
- If you see labeled numerical axes → has_quantitative_data = true
- Spectra always have m/z, ppm, or cm⁻¹ on the x-axis
- Western blots have horizontal bands; they do NOT have x/y numerical axes with data points
- If unsure between two types, pick the one with higher confidence and note both in reasoning
- microscopy images often lack axes entirely"""


# Mapping from figure_type → extraction strategy tag
_EXTRACTION_STRATEGY_MAP: dict[str, str] = {
    "line_plot":    "plot_digitizer",
    "scatter_plot": "plot_digitizer",
    "bar_chart":    "bar_extractor",
    "heatmap":      "heatmap_extractor",
    "spectrum_nmr": "spectrum_extractor",
    "spectrum_ir":  "spectrum_extractor",
    "spectrum_ms":  "spectrum_extractor",
    "western_blot": "gel_extractor",
    "microscopy":   "microscopy_extractor",
    "flowchart":    "description_only",
    "other":        "description_only",
}

# Fallback result when classification completely fails
_FALLBACK_RESULT: dict = {
    "figure_type": "other",
    "confidence": 0.0,
    "reasoning": "Classification failed — defaulting to 'other'.",
    "has_quantitative_data": False,
    "sub_types": [],
}


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------

class FigureClassifier:
    """Classify figure images using Gemma4 multimodal inference.

    Parameters
    ----------
    ollama_client : OllamaClient
        A configured OllamaClient instance.
    confidence_threshold : float
        Results below this threshold are still returned but flagged with
        figure_type = 'other' when confidence is too low.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        client: Optional[OllamaClient] = None,  # backwards-compat alias
    ) -> None:
        # Support both keyword argument forms
        self.client: OllamaClient = ollama_client or client or OllamaClient()
        self.confidence_threshold = confidence_threshold
        self._valid_types = set(FIGURE_TYPES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, image_path: str, caption: str) -> dict:
        """Classify a single figure image.

        Parameters
        ----------
        image_path : str
            Path to the extracted PNG file.
        caption : str
            The figure caption text.

        Returns
        -------
        dict with keys:
            figure_type, confidence, reasoning,
            has_quantitative_data, sub_types
        """
        prompt = CLASSIFICATION_USER_TEMPLATE.format(
            caption=caption or "(no caption provided)"
        )
        try:
            raw = self.client.query_multimodal(
                prompt=prompt,
                image_path=image_path,
                system=CLASSIFICATION_SYSTEM_PROMPT,
                temperature=0.05,  # Low temp → deterministic classification
            )
            result = self._parse_response(raw)
        except Exception as exc:
            logger.error("Classification error for %s: %s", image_path, exc)
            result = dict(_FALLBACK_RESULT)

        logger.info(
            "Classified %s → %s (conf=%.2f) | quantitative=%s",
            image_path, result["figure_type"],
            result["confidence"], result["has_quantitative_data"],
        )
        return result

    def classify_batch(self, figures: list[dict]) -> list[dict]:
        """Classify a list of figure dicts in-place.

        Each figure dict must have at minimum:
            "image_path" (str) and optionally "caption" (str).

        The "classification" key is added to each dict with the result.
        A single failure never stops the batch.

        Parameters
        ----------
        figures : list[dict]
            List of figure metadata dicts (from FigureExtractor).

        Returns
        -------
        list[dict]
            The same list, each item enriched with a "classification" key.
        """
        for fig in tqdm(figures, desc="Classifying figures", unit="fig"):
            image_path = fig.get("image_path") or ""
            caption = fig.get("caption") or ""
            try:
                fig["classification"] = self.classify(image_path, caption)
            except Exception as exc:
                logger.error(
                    "classify_batch: unhandled error for figure %s: %s",
                    fig.get("figure_number", "?"), exc,
                )
                fig["classification"] = dict(_FALLBACK_RESULT)
        return figures

    # Also handle FigureRecord objects (duck-typing compatibility)
    def classify_figure_records(self, figures: list) -> list:
        """Classify a list of FigureRecord dataclass instances.

        Populates `record.figure_type` and `record.confidence` in-place.
        """
        for fig in tqdm(figures, desc="Classifying figures", unit="fig"):
            image_path = str(fig.image_path) if fig.image_path else ""
            caption = fig.caption or ""
            try:
                result = self.classify(image_path, caption)
                fig.figure_type = result["figure_type"]
                fig.confidence = result["confidence"]
            except Exception as exc:
                logger.error(
                    "classify_figure_records: error for figure %s: %s",
                    fig.figure_number, exc,
                )
                fig.figure_type = "other"
                fig.confidence = 0.0
        return figures

    @staticmethod
    def get_extraction_strategy(figure_type: str) -> str:
        """Map a figure_type label to an extraction strategy tag.

        Parameters
        ----------
        figure_type : str
            One of the FIGURE_TYPES values.

        Returns
        -------
        str
            Strategy tag: "plot_digitizer", "bar_extractor", "spectrum_extractor",
            "heatmap_extractor", "gel_extractor", "microscopy_extractor",
            or "description_only".
        """
        return _EXTRACTION_STRATEGY_MAP.get(figure_type, "description_only")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> dict:
        """Parse the model's JSON response, with multiple fallback layers."""
        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

        data: dict = {}

        # Attempt 1: direct parse
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt 2: extract JSON object with regex
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if not data:
            logger.warning("Could not parse JSON from model response: %s", raw[:200])
            return dict(_FALLBACK_RESULT)

        # Normalise and validate
        figure_type = str(data.get("figure_type", "other")).lower().strip()
        if figure_type not in self._valid_types:
            # Fuzzy match: see if any valid type is a substring of the returned label
            figure_type = next(
                (ft for ft in FIGURE_TYPES if ft in figure_type), "other"
            )

        confidence = float(data.get("confidence", 0.3))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        # Downgrade very low-confidence predictions to "other"
        if confidence < self.confidence_threshold:
            logger.debug(
                "Confidence %.2f below threshold %.2f → reclassifying as 'other'",
                confidence, self.confidence_threshold,
            )
            figure_type = "other"

        has_quant = bool(data.get("has_quantitative_data", True))
        # Microscopy and flowcharts almost never have quantitative data
        if figure_type in ("microscopy", "flowchart"):
            has_quant = data.get("has_quantitative_data", False)

        return {
            "figure_type": figure_type,
            "confidence": confidence,
            "reasoning": str(data.get("reasoning", "")),
            "has_quantitative_data": has_quant,
            "sub_types": list(data.get("sub_types", [])),
        }


# ---------------------------------------------------------------------------
# Test function (run directly for smoke-testing)
# ---------------------------------------------------------------------------

def test_classifier(image_path: str, caption: str) -> None:
    """Smoke-test the classifier against a single image.

    Example:
        python -m pipeline.classifier path/to/fig.png "Figure 1. Growth curves..."
    """
    client = OllamaClient()
    if not client.is_available():
        print("[ERROR] Ollama is not running or the model is not available.")
        print("  Start Ollama: ollama serve")
        print("  Pull model:   ollama pull gemma4:e4b")
        sys.exit(1)

    clf = FigureClassifier(ollama_client=client)
    result = clf.classify(image_path, caption)

    print("\n=== Classification Result ===")
    print(json.dumps(result, indent=2))
    print(f"\nExtraction strategy: {FigureClassifier.get_extraction_strategy(result['figure_type'])}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.classifier <image_path> <caption>")
        sys.exit(1)
    test_classifier(sys.argv[1], sys.argv[2])
