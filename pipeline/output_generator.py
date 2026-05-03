"""
pipeline/output_generator.py — Phase 7 & 8: CSV / JSON / SQLite output

Orchestrates the final output for a single paper:
  • Per-figure CSV files
  • Full-paper JSON with provenance
  • Summary report (text)
  • Database persistence via DatabaseManager
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from config import OUTPUT_DIR
from database.db import DatabaseManager
from pipeline.context_builder import PromptContext
from pipeline.extractor import ExtractedSeries
from pipeline.figure_extractor import FigureRecord
from pipeline.pdf_parser import PaperMetadata

logger = logging.getLogger(__name__)


class OutputGenerator:
    """Generate all output artefacts for a processed paper.

    Parameters
    ----------
    paper_meta : PaperMetadata
        Metadata extracted by PDFParser.
    figures : list[FigureRecord]
        All figures extracted from the paper.
    series_map : dict[str, list[ExtractedSeries]]
        Mapping of figure_number → extracted data series.
    db : DatabaseManager, optional
        If supplied, data is also persisted to SQLite.
    output_dir : Path, optional
        Root output directory (defaults to OUTPUT_DIR from config).
    """

    def __init__(
        self,
        paper_meta: PaperMetadata,
        figures: list[FigureRecord],
        series_map: dict[str, list[ExtractedSeries]],
        db: Optional[DatabaseManager] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.meta = paper_meta
        self.figures = figures
        self.series_map = series_map
        self.db = db
        self.paper_stem = Path(paper_meta.pdf_path).stem
        self.out_dir = Path(output_dir or OUTPUT_DIR) / self.paper_stem
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> dict[str, list[Path]]:
        """Run all output generators and return paths grouped by type.

        Returns
        -------
        dict with keys 'csv', 'json', 'report', each containing a list of Paths.
        """
        csv_paths = self.generate_csvs()
        json_path = self.generate_json()
        report_path = self.generate_report()

        if self.db:
            self._persist_to_db()

        return {"csv": csv_paths, "json": [json_path], "report": [report_path]}

    def generate_csvs(self) -> list[Path]:
        """Write one CSV per data series and return their paths."""
        paths: list[Path] = []
        for fig in self.figures:
            series_list = self.series_map.get(fig.figure_number, [])
            for s in series_list:
                if not s.data_points:
                    continue
                df = pd.DataFrame(s.data_points)
                safe_series = (s.series_name or "series").replace(" ", "_").replace("/", "-")
                fname = f"fig{fig.figure_number}_{safe_series}.csv"
                path = self.out_dir / fname
                df.to_csv(path, index=False)
                paths.append(path)
                logger.debug("CSV → %s  (%d rows)", path, len(df))
        logger.info("Generated %d CSV files in %s", len(paths), self.out_dir)
        return paths

    def generate_json(self) -> Path:
        """Write a single structured JSON file with full provenance."""
        output: dict = {
            "paper": {
                "pdf_path": str(self.meta.pdf_path),
                "doi": self.meta.doi,
                "title": self.meta.title,
                "authors": self.meta.authors,
                "journal": self.meta.journal,
                "year": self.meta.year,
                "page_count": self.meta.page_count,
            },
            "figures": [],
        }

        for fig in self.figures:
            fig_dict = {
                "figure_number": fig.figure_number,
                "panel_label": fig.panel_label,
                "page_number": fig.page_number,
                "figure_type": getattr(fig, "figure_type", None),
                "caption": fig.caption,
                "image_path": str(fig.image_path) if fig.image_path else None,
                "bounding_box": fig.bounding_box,
                "confidence": fig.confidence,
                "extracted_series": [
                    self._series_to_dict(s)
                    for s in self.series_map.get(fig.figure_number, [])
                ],
            }
            output["figures"].append(fig_dict)

        json_path = self.out_dir / "extraction_full.json"
        json_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("JSON → %s", json_path)
        return json_path

    def generate_report(self) -> Path:
        """Write a human-readable summary report."""
        lines = [
            "=" * 60,
            "FigureVault Extraction Report",
            "=" * 60,
            f"Paper   : {self.meta.title or 'Unknown'}",
            f"DOI     : {self.meta.doi or 'Unknown'}",
            f"PDF     : {self.meta.pdf_path}",
            f"Pages   : {self.meta.page_count}",
            f"Figures : {len(self.figures)}",
            "",
        ]
        total_series = 0
        total_points = 0
        for fig in self.figures:
            series_list = self.series_map.get(fig.figure_number, [])
            n_pts = sum(len(s.data_points) for s in series_list)
            total_series += len(series_list)
            total_points += n_pts
            fig_type = getattr(fig, "figure_type", "unknown")
            lines.append(
                f"  Fig {fig.figure_number:<5} | {fig_type:<20} | "
                f"{len(series_list)} series | {n_pts} data points"
            )

        lines += [
            "",
            f"Total series extracted : {total_series}",
            f"Total data points      : {total_points}",
            "",
            f"Outputs written to: {self.out_dir}",
            "=" * 60,
        ]

        report_path = self.out_dir / "report.txt"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Report → %s", report_path)
        return report_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _persist_to_db(self) -> None:
        """Save all extracted data to the SQLite database."""
        assert self.db is not None
        paper_id = self.db.insert_paper(
            pdf_path=self.meta.pdf_path,
            doi=self.meta.doi,
            title=self.meta.title,
            authors=self.meta.authors,
            journal=self.meta.journal,
            year=self.meta.year,
        )
        for fig in self.figures:
            fig_id = self.db.insert_figure(
                paper_id=paper_id,
                figure_number=fig.figure_number,
                panel_label=fig.panel_label,
                page_number=fig.page_number,
                figure_type=getattr(fig, "figure_type", None),
                caption=fig.caption,
                image_path=str(fig.image_path) if fig.image_path else None,
                bounding_box=fig.bounding_box,
                confidence_score=fig.confidence,
            )
            for s in self.series_map.get(fig.figure_number, []):
                self.db.insert_extracted_data(
                    figure_id=fig_id,
                    series_name=s.series_name,
                    x_label=s.x_label,
                    x_unit=s.x_unit,
                    y_label=s.y_label,
                    y_unit=s.y_unit,
                    data_points=s.data_points,
                    axis_scale_x=s.axis_scale_x,
                    axis_scale_y=s.axis_scale_y,
                    statistical_annotations=s.statistical_annotations,
                    extraction_method=s.extraction_method,
                    confidence_score=s.confidence,
                )
        logger.info("Persisted paper id=%d to database", paper_id)

    @staticmethod
    def _series_to_dict(s: ExtractedSeries) -> dict:
        return {
            "series_name": s.series_name,
            "x_label": s.x_label,
            "x_unit": s.x_unit,
            "y_label": s.y_label,
            "y_unit": s.y_unit,
            "axis_scale_x": s.axis_scale_x,
            "axis_scale_y": s.axis_scale_y,
            "data_points": s.data_points,
            "statistical_annotations": s.statistical_annotations,
            "confidence": s.confidence,
            "extraction_method": s.extraction_method,
        }
