"""
training/synthetic_generator.py — Generate synthetic figure+data training pairs

Creates matplotlib figures from known datasets so we have perfectly labelled
(figure image, ground truth CSV) pairs for fine-tuning Gemma4.

Supports: line plots, bar charts, scatter plots, heatmaps.
"""

from __future__ import annotations

import csv
import logging
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SyntheticPair:
    """One synthetic training pair."""
    figure_path: Path
    csv_path: Path
    metadata_path: Path
    plot_type: str
    caption: str


class SyntheticGenerator:
    """Generate synthetic scientific figure + ground truth CSV pairs.

    Parameters
    ----------
    output_dir : Path
        Where to write generated files.
    seed : int
        Random seed for reproducibility.
    """

    PLOT_TYPES = ["line_plot", "bar_chart", "scatter_plot", "heatmap"]

    def __init__(self, output_dir: Path = Path("training_data/synthetic"), seed: int = 42) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self._pair_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n: int = 100) -> list[SyntheticPair]:
        """Generate ``n`` synthetic training pairs.

        Parameters
        ----------
        n : int
            Number of pairs to generate.

        Returns
        -------
        list[SyntheticPair]
        """
        pairs: list[SyntheticPair] = []
        for i in range(n):
            plot_type = random.choice(self.PLOT_TYPES)
            pair = self._generate_one(plot_type)
            pairs.append(pair)
            if (i + 1) % 50 == 0:
                logger.info("Generated %d / %d synthetic pairs", i + 1, n)
        logger.info("Synthetic generation complete: %d pairs in %s", n, self.output_dir)
        return pairs

    # ------------------------------------------------------------------
    # Per-type generators
    # ------------------------------------------------------------------

    def _generate_one(self, plot_type: str) -> SyntheticPair:
        """Dispatch to the appropriate plot-type generator."""
        self._pair_counter += 1
        idx = self._pair_counter

        if plot_type == "line_plot":
            return self._gen_line_plot(idx)
        elif plot_type == "bar_chart":
            return self._gen_bar_chart(idx)
        elif plot_type == "scatter_plot":
            return self._gen_scatter(idx)
        elif plot_type == "heatmap":
            return self._gen_heatmap(idx)
        else:
            return self._gen_line_plot(idx)

    def _gen_line_plot(self, idx: int) -> SyntheticPair:
        """Generate a multi-series line plot."""
        n_series = self.rng.integers(1, 5)
        n_points = self.rng.integers(5, 30)
        x = np.linspace(0, 10, n_points)
        x_label = random.choice(["Time (s)", "Concentration (µM)", "Temperature (°C)", "Wavelength (nm)"])
        y_label = random.choice(["Absorbance (AU)", "Fluorescence (a.u.)", "Activity (%)", "Signal (mV)"])

        fig, ax = plt.subplots(figsize=(6, 4))
        all_rows: list[dict] = []
        colors = plt.cm.tab10.colors

        for s in range(n_series):
            y = self.rng.normal(s + 1, 0.3, n_points).cumsum() * self.rng.uniform(0.1, 0.5)
            err = self.rng.uniform(0.05, 0.3, n_points)
            label = f"Series {chr(65 + s)}"
            ax.errorbar(x, y, yerr=err, label=label, color=colors[s % 10], capsize=3)
            for xi, yi, ei in zip(x, y, err):
                all_rows.append({"series": label, "x": round(xi, 4), "y": round(float(yi), 4), "err_y": round(float(ei), 4)})

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Synthetic Line Plot {idx}")
        ax.legend()
        plt.tight_layout()

        caption = f"Effect of {x_label.split('(')[0].strip()} on {y_label.split('(')[0].strip()} for {n_series} experimental conditions."
        return self._save_pair(idx, "line_plot", fig, all_rows, caption)

    def _gen_bar_chart(self, idx: int) -> SyntheticPair:
        """Generate a grouped bar chart."""
        categories = [f"Group {i}" for i in range(self.rng.integers(2, 7))]
        n_groups = self.rng.integers(1, 4)
        x = np.arange(len(categories))
        width = 0.8 / n_groups

        fig, ax = plt.subplots(figsize=(6, 4))
        all_rows: list[dict] = []
        colors = plt.cm.tab10.colors

        for g in range(n_groups):
            vals = self.rng.uniform(0.5, 5.0, len(categories))
            errs = self.rng.uniform(0.05, 0.5, len(categories))
            label = f"Condition {g + 1}"
            ax.bar(x + g * width, vals, width, yerr=errs, label=label, color=colors[g % 10], capsize=4)
            for cat, v, e in zip(categories, vals, errs):
                all_rows.append({"series": label, "category": cat, "value": round(float(v), 4), "err": round(float(e), 4)})

        ax.set_xticks(x + width * (n_groups - 1) / 2)
        ax.set_xticklabels(categories)
        ax.set_ylabel("Value (AU)")
        ax.set_title(f"Synthetic Bar Chart {idx}")
        ax.legend()
        plt.tight_layout()

        caption = f"Comparison of values across {len(categories)} groups under {n_groups} experimental conditions."
        return self._save_pair(idx, "bar_chart", fig, all_rows, caption)

    def _gen_scatter(self, idx: int) -> SyntheticPair:
        """Generate a scatter plot with optional trend line."""
        n_points = self.rng.integers(20, 80)
        x = self.rng.uniform(0, 10, n_points)
        slope = self.rng.uniform(-1, 1)
        y = slope * x + self.rng.normal(0, 1, n_points)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x, y, alpha=0.7, s=30)
        ax.set_xlabel("Variable X")
        ax.set_ylabel("Variable Y")
        ax.set_title(f"Synthetic Scatter {idx}")
        plt.tight_layout()

        all_rows = [{"x": round(float(xi), 4), "y": round(float(yi), 4)} for xi, yi in zip(x, y)]
        caption = f"Scatter plot showing the relationship between Variable X and Variable Y (n={n_points})."
        return self._save_pair(idx, "scatter_plot", fig, all_rows, caption)

    def _gen_heatmap(self, idx: int) -> SyntheticPair:
        """Generate a heatmap."""
        rows = self.rng.integers(4, 10)
        cols = self.rng.integers(4, 10)
        data = self.rng.uniform(0, 1, (rows, cols))

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Synthetic Heatmap {idx}")
        plt.tight_layout()

        all_rows = [
            {"row": r, "col": c, "value": round(float(data[r, c]), 4)}
            for r in range(rows)
            for c in range(cols)
        ]
        caption = f"Heatmap of measurement values across {rows} conditions and {cols} replicates."
        return self._save_pair(idx, "heatmap", fig, all_rows, caption)

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _save_pair(
        self,
        idx: int,
        plot_type: str,
        fig: plt.Figure,
        rows: list[dict],
        caption: str,
    ) -> SyntheticPair:
        """Save figure PNG, CSV, and metadata JSON."""
        stem = f"{plot_type}_{idx:05d}"
        fig_path = self.output_dir / f"{stem}.png"
        csv_path = self.output_dir / f"{stem}.csv"
        meta_path = self.output_dir / f"{stem}_meta.json"

        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        if rows:
            fieldnames = list(rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        import json
        meta_path.write_text(
            json.dumps({"plot_type": plot_type, "caption": caption, "n_rows": len(rows)}, indent=2),
            encoding="utf-8",
        )

        return SyntheticPair(
            figure_path=fig_path,
            csv_path=csv_path,
            metadata_path=meta_path,
            plot_type=plot_type,
            caption=caption,
        )
