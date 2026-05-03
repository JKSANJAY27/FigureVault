"""
pipeline/digitizer.py — Phase 6: OpenCV pixel-level digitization

Uses Gemma4's axis understanding (from Phase 5) combined with classical
computer vision to precisely digitize data points from plot images.

Workflow per figure:
  1. Detect the plot area (axes bounding box) using edge detection
  2. Locate axis tick marks and labels to calibrate pixel ↔ data coordinates
  3. Detect data series (by colour / marker)
  4. Extract (x, y) coordinates for each detected point/line sample
  5. Map pixel coordinates → data space using the axis calibration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AxisCalibration:
    """Pixel-to-data coordinate mapping for one axis."""
    pixel_min: float
    pixel_max: float
    data_min: float
    data_max: float
    is_log: bool = False

    def pixel_to_data(self, px: float) -> float:
        """Convert a pixel coordinate to a data value."""
        if self.pixel_max == self.pixel_min:
            return self.data_min
        t = (px - self.pixel_min) / (self.pixel_max - self.pixel_min)
        if self.is_log:
            import math
            return 10 ** (math.log10(self.data_min) + t * (math.log10(self.data_max) - math.log10(self.data_min)))
        return self.data_min + t * (self.data_max - self.data_min)


@dataclass
class DigitizedSeries:
    """Result of pixel-level digitization for one colour-identified series."""
    color_bgr: tuple[int, int, int] = (0, 0, 0)
    data_points: list[dict] = field(default_factory=list)
    method: str = "opencv_digitizer"


class PlotDigitizer:
    """Pixel-level plot digitizer using OpenCV.

    This class is invoked AFTER Phase 5 has provided:
      • The axis labels and scales (linear / log)
      • Approximate axis min/max values (from Gemma4's text reading)

    It then uses computer vision to precisely extract the data point
    coordinates, which are more accurate than LLM coordinate guesses.

    Parameters
    ----------
    image_path : str | Path
        Path to the figure PNG.
    x_cal : AxisCalibration
        Pixel-to-data calibration for the x-axis.
    y_cal : AxisCalibration
        Pixel-to-data calibration for the y-axis.
    """

    def __init__(
        self,
        image_path: str | Path,
        x_cal: Optional[AxisCalibration] = None,
        y_cal: Optional[AxisCalibration] = None,
    ) -> None:
        self.image_path = Path(image_path)
        self.x_cal = x_cal
        self.y_cal = y_cal
        self._img: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_image(self) -> np.ndarray:
        """Load and cache the image from disk."""
        if self._img is None:
            self._img = cv2.imread(str(self.image_path))
            if self._img is None:
                raise FileNotFoundError(f"Cannot open image: {self.image_path}")
        return self._img

    def detect_plot_area(self) -> Optional[tuple[int, int, int, int]]:
        """Detect the bounding box of the plot area (axes region).

        Uses Canny edge detection and contour finding to locate the largest
        rectangular region, which is typically the plot frame.

        Returns
        -------
        tuple (x, y, w, h) in pixels, or None if detection fails.
        """
        img = self.load_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # The plot frame is typically the largest contour with 4 corners
        best = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(best)
        logger.debug("Detected plot area: x=%d y=%d w=%d h=%d", x, y, w, h)
        return x, y, w, h

    def extract_series_by_color(
        self,
        hsv_ranges: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[DigitizedSeries]:
        """Extract data series by isolating distinct colours.

        Parameters
        ----------
        hsv_ranges : list of (lower_hsv, upper_hsv) numpy arrays
            Each pair defines an HSV colour range to isolate one series.

        Returns
        -------
        list[DigitizedSeries]
        """
        img = self.load_image()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        results: list[DigitizedSeries] = []

        for lower, upper in hsv_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            # Find pixel coordinates of the masked region
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            # Sort by x, group nearby points, take median y
            coords = sorted(zip(xs.tolist(), ys.tolist()))
            points = self._group_and_convert(coords)

            # Get representative BGR colour for this series
            mean_x = int(np.mean(xs))
            mean_y = int(np.mean(ys))
            color_bgr = tuple(int(c) for c in img[mean_y, mean_x])

            results.append(DigitizedSeries(
                color_bgr=color_bgr,
                data_points=points,
            ))

        return results

    def auto_digitize(self) -> list[DigitizedSeries]:
        """Attempt fully automatic digitization using colour clustering.

        Uses K-means to identify dominant colours in the plot area, then
        segments each colour as a separate data series.

        Returns
        -------
        list[DigitizedSeries]
            One series per detected dominant colour.
        """
        img = self.load_image()
        plot_box = self.detect_plot_area()
        if plot_box is None:
            logger.warning("Could not detect plot area — using full image")
            roi = img
        else:
            x, y, w, h = plot_box
            roi = img[y:y+h, x:x+w]

        # Reshape for K-means
        pixels = roi.reshape(-1, 3).astype(np.float32)

        # Remove near-white (background) and near-black (axes/text) pixels
        mask = np.all(pixels < 230, axis=1) & np.all(pixels > 30, axis=1)
        filtered = pixels[mask]

        if len(filtered) < 50:
            logger.warning("Not enough coloured pixels for K-means digitization")
            return []

        n_clusters = min(5, len(filtered))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            filtered, n_clusters, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS
        )

        series_list: list[DigitizedSeries] = []
        for k in range(n_clusters):
            cluster_pixels_idx = np.where(labels.flatten() == k)[0]
            if len(cluster_pixels_idx) < 20:
                continue

            # Map back to image coordinates
            all_fg_idx = np.where(mask)[0]
            selected = all_fg_idx[cluster_pixels_idx]
            h_roi, w_roi = roi.shape[:2]
            ys = (selected // w_roi).tolist()
            xs = (selected % w_roi).tolist()

            if plot_box:
                xs = [x + plot_box[0] for x in xs]
                ys = [y + plot_box[1] for y in ys]

            coords = sorted(zip(xs, ys))
            points = self._group_and_convert(coords)
            color_bgr = tuple(int(c) for c in centers[k])
            series_list.append(DigitizedSeries(color_bgr=color_bgr, data_points=points))

        logger.info("Auto-digitizer found %d series", len(series_list))
        return series_list

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _group_and_convert(
        self,
        coords: list[tuple[int, int]],
        x_tolerance: int = 5,
    ) -> list[dict]:
        """Group pixel coordinates into data points and convert via calibration.

        Parameters
        ----------
        coords : list of (px_x, px_y)
            Pixel coordinates sorted by x.
        x_tolerance : int
            Pixels within this x-distance are merged into one point.

        Returns
        -------
        list of {"x": float, "y": float, "err_x": None, "err_y": None}
        """
        if not coords:
            return []

        groups: list[list[tuple[int, int]]] = []
        current = [coords[0]]
        for pt in coords[1:]:
            if abs(pt[0] - current[-1][0]) <= x_tolerance:
                current.append(pt)
            else:
                groups.append(current)
                current = [pt]
        groups.append(current)

        points: list[dict] = []
        for group in groups:
            px_x = np.median([p[0] for p in group])
            px_y = np.median([p[1] for p in group])

            if self.x_cal and self.y_cal:
                data_x = self.x_cal.pixel_to_data(px_x)
                data_y = self.y_cal.pixel_to_data(px_y)
            else:
                data_x = float(px_x)
                data_y = float(px_y)

            points.append({"x": round(data_x, 6), "y": round(data_y, 6), "err_x": None, "err_y": None})

        return points
