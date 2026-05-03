"""
pipeline/classifier.py — Phase 3: Figure type classification

Sends each extracted figure image + its caption to Gemma 4 via Ollama
and returns one of the FIGURE_TYPES labels defined in config.py.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from config import CONFIDENCE_THRESHOLD, FIGURE_TYPES
from models.ollama_client import OllamaClient
from pipeline.figure_extractor import FigureRecord

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a scientific figure classification expert.\n"
    "Classify the figure into ONE of the categories listed.\n"
    "Respond ONLY with JSON: "
    '{"label":"<category>","confidence":<0-1>,"reasoning":"<one sentence>"}'
)


class FigureClassifier:
    """Classify figure images using Gemma 4 multimodal inference."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.client = client or OllamaClient()
        self.confidence_threshold = confidence_threshold
        self._categories_str = ", ".join(FIGURE_TYPES)

    def classify(self, figure: FigureRecord) -> FigureRecord:
        """Classify a single FigureRecord and populate figure_type & confidence."""
        if figure.image_path is None or not figure.image_path.exists():
            figure.figure_type = "other"
            return figure

        prompt = (
            f"Categories: {self._categories_str}\n\n"
            f"Caption: {figure.caption or '(no caption)'}\n\n"
            "Classify this scientific figure."
        )
        try:
            raw = self.client.query_multimodal(
                prompt=prompt,
                image_path=figure.image_path,
                system=_SYSTEM_PROMPT,
                temperature=0.05,
            )
            label, confidence = self._parse_response(raw)
        except Exception as exc:
            logger.error("Classification failed for figure %s: %s", figure.figure_number, exc)
            label, confidence = "other", 0.0

        figure.figure_type = label
        figure.confidence = confidence
        logger.info("Figure %s → %s (conf=%.2f)", figure.figure_number, label, confidence)
        return figure

    def classify_batch(self, figures: list[FigureRecord]) -> list[FigureRecord]:
        """Classify a list of figures sequentially."""
        for fig in figures:
            self.classify(fig)
        return figures

    def _parse_response(self, raw: str) -> tuple[str, float]:
        """Parse JSON response from model, with fallbacks."""
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            data = json.loads(m.group()) if m else {}

        label = data.get("label", "other")
        confidence = float(data.get("confidence", 0.3))

        if label not in FIGURE_TYPES:
            label = next((ft for ft in FIGURE_TYPES if ft in label.lower()), "other")
        if confidence < self.confidence_threshold:
            label = "other"

        return label, confidence
