"""
eval/benchmark.py — Accuracy benchmarking for FigureVault

Compares FigureVault extracted data against WebPlotDigitizer ground truth
or synthetic datasets with known values.

Metrics reported:
  • RMSE per figure type
  • R² correlation
  • Point-level precision / recall
  • Series count accuracy
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import BENCHMARK_RMSE_THRESHOLD
from eval.metrics import compute_rmse, compute_r2, match_series

logger = logging.getLogger(__name__)


@dataclass
class FigureBenchmarkResult:
    """Benchmark result for a single figure."""
    figure_id: str
    figure_type: str
    rmse: Optional[float] = None
    r2: Optional[float] = None
    series_precision: float = 0.0
    series_recall: float = 0.0
    passed: bool = False
    notes: str = ""


@dataclass
class BenchmarkReport:
    """Aggregate benchmark report across all test figures."""
    total_figures: int = 0
    passed: int = 0
    failed: int = 0
    mean_rmse: float = 0.0
    mean_r2: float = 0.0
    results_by_type: dict[str, list[FigureBenchmarkResult]] = field(default_factory=dict)


class Benchmarker:
    """Benchmark FigureVault against a ground-truth test set.

    The test set directory must contain pairs of files:
      • ``<stem>.png``            — the figure image
      • ``<stem>_ground_truth.json``  — ground truth JSON with same schema as
                                        DataExtractor output

    Parameters
    ----------
    test_dir : Path
        Directory containing figure+ground-truth pairs.
    rmse_threshold : float
        RMSE below which a figure extraction is counted as "passed".
    """

    def __init__(
        self,
        test_dir: Path,
        rmse_threshold: float = BENCHMARK_RMSE_THRESHOLD,
    ) -> None:
        self.test_dir = Path(test_dir)
        self.rmse_threshold = rmse_threshold

    def run(self) -> BenchmarkReport:
        """Run the benchmark over all test figures.

        Returns
        -------
        BenchmarkReport
        """
        from models.ollama_client import OllamaClient
        from pipeline.context_builder import PromptContext
        from pipeline.extractor import DataExtractor
        from pipeline.figure_extractor import FigureRecord

        gt_files = list(self.test_dir.glob("*_ground_truth.json"))
        logger.info("Benchmarking %d figures in %s", len(gt_files), self.test_dir)

        client = OllamaClient()
        extractor = DataExtractor(client=client)
        report = BenchmarkReport(total_figures=len(gt_files))

        all_rmse: list[float] = []
        all_r2: list[float] = []

        for gt_file in gt_files:
            stem = gt_file.name.replace("_ground_truth.json", "")
            fig_path = gt_file.parent / f"{stem}.png"

            if not fig_path.exists():
                logger.warning("Missing figure for %s", gt_file)
                continue

            # Build a minimal FigureRecord
            fig = FigureRecord(
                page_number=1,
                figure_number=stem,
                image_path=fig_path,
            )

            # Load ground truth
            try:
                gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Cannot load ground truth %s: %s", gt_file, exc)
                continue

            # Run extraction
            ctx = PromptContext(figure=fig)
            predicted = extractor.extract(ctx)

            # Evaluate
            result = self._evaluate(stem, gt_data, predicted)
            report.results_by_type.setdefault(result.figure_type, []).append(result)

            if result.passed:
                report.passed += 1
            else:
                report.failed += 1

            if result.rmse is not None:
                all_rmse.append(result.rmse)
            if result.r2 is not None:
                all_r2.append(result.r2)

        report.mean_rmse = float(np.mean(all_rmse)) if all_rmse else 0.0
        report.mean_r2 = float(np.mean(all_r2)) if all_r2 else 0.0
        logger.info(
            "Benchmark complete: %d/%d passed, mean RMSE=%.4f, mean R²=%.4f",
            report.passed, report.total_figures, report.mean_rmse, report.mean_r2,
        )
        return report

    def print_report(self, report: BenchmarkReport) -> None:
        """Print a formatted benchmark report to stdout."""
        print("\n" + "=" * 60)
        print("FigureVault Benchmark Report")
        print("=" * 60)
        print(f"Total figures : {report.total_figures}")
        print(f"Passed        : {report.passed}")
        print(f"Failed        : {report.failed}")
        print(f"Pass rate     : {report.passed / max(report.total_figures, 1) * 100:.1f}%")
        print(f"Mean RMSE     : {report.mean_rmse:.4f}  (threshold={self.rmse_threshold})")
        print(f"Mean R²       : {report.mean_r2:.4f}")
        print("\nResults by figure type:")
        for fig_type, results in sorted(report.results_by_type.items()):
            n = len(results)
            passed = sum(1 for r in results if r.passed)
            mean_rmse = np.mean([r.rmse for r in results if r.rmse is not None]) if results else 0.0
            print(f"  {fig_type:<20} {passed}/{n} passed  RMSE={mean_rmse:.4f}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        figure_id: str,
        ground_truth: list[dict],
        predicted: list,
    ) -> FigureBenchmarkResult:
        """Compute accuracy metrics for one figure."""
        fig_type = ground_truth[0].get("figure_type", "unknown") if ground_truth else "unknown"
        result = FigureBenchmarkResult(figure_id=figure_id, figure_type=fig_type)

        if not ground_truth or not predicted:
            result.notes = "Missing ground truth or predictions"
            return result

        # Match predicted series to ground truth series
        matched_pairs = match_series(ground_truth, [vars(p) if hasattr(p, '__dict__') else p for p in predicted])

        all_gt_y: list[float] = []
        all_pred_y: list[float] = []

        for gt_series, pred_series in matched_pairs:
            gt_pts = gt_series.get("data_points", [])
            pred_pts = pred_series.get("data_points", [])
            min_len = min(len(gt_pts), len(pred_pts))
            if min_len == 0:
                continue
            all_gt_y.extend([p["y"] for p in gt_pts[:min_len]])
            all_pred_y.extend([p["y"] for p in pred_pts[:min_len]])

        if all_gt_y:
            result.rmse = compute_rmse(np.array(all_gt_y), np.array(all_pred_y))
            result.r2 = compute_r2(np.array(all_gt_y), np.array(all_pred_y))
            result.passed = result.rmse <= self.rmse_threshold

        return result
