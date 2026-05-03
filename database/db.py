"""
database/db.py — FigureVault Database Manager

Thin wrapper around SQLite (via Python's built-in sqlite3).
All heavy-lifting queries live here; pipeline modules stay DB-agnostic.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import pandas as pd

from config import DB_PATH, OUTPUT_DIR

logger = logging.getLogger(__name__)

# Path to the SQL schema file (sibling of this module)
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class DatabaseManager:
    """Manages all interactions with the FigureVault SQLite database.

    Usage
    -----
    db = DatabaseManager()
    db.init_db()
    paper_id = db.insert_paper(doi="10.1234/test", title="My Paper", ...)
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a sqlite3 connection and commits/rolls back."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema initialisation
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create all tables from schema.sql if they do not already exist."""
        schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(schema_sql)
        logger.info("Database initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Insert helpers
    # ------------------------------------------------------------------

    def insert_paper(
        self,
        pdf_path: str,
        doi: Optional[str] = None,
        title: Optional[str] = None,
        authors: Optional[list[str]] = None,
        journal: Optional[str] = None,
        year: Optional[int] = None,
    ) -> int:
        """Insert a paper record and return its auto-assigned id.

        If a record with the same pdf_path already exists, its id is returned
        without inserting a duplicate.
        """
        authors_json = json.dumps(authors or [])
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO papers (doi, title, authors, journal, year, pdf_path)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pdf_path) DO UPDATE SET
                    doi     = excluded.doi,
                    title   = excluded.title,
                    authors = excluded.authors,
                    journal = excluded.journal,
                    year    = excluded.year
                RETURNING id
                """,
                (doi, title, authors_json, journal, year, str(pdf_path)),
            )
            row = cur.fetchone()
            paper_id: int = row["id"]
        logger.debug("Upserted paper id=%d  pdf_path=%s", paper_id, pdf_path)
        return paper_id

    def insert_figure(
        self,
        paper_id: int,
        figure_number: Optional[str] = None,
        panel_label: Optional[str] = None,
        page_number: Optional[int] = None,
        figure_type: Optional[str] = None,
        caption: Optional[str] = None,
        image_path: Optional[str] = None,
        bounding_box: Optional[dict] = None,
        confidence_score: Optional[float] = None,
    ) -> int:
        """Insert a figure record and return its auto-assigned id."""
        bb_json = json.dumps(bounding_box) if bounding_box else None
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO figures
                    (paper_id, figure_number, panel_label, page_number, figure_type,
                     caption, image_path, bounding_box_json, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    paper_id, figure_number, panel_label, page_number, figure_type,
                    caption, str(image_path) if image_path else None,
                    bb_json, confidence_score,
                ),
            )
            figure_id: int = cur.fetchone()["id"]
        logger.debug("Inserted figure id=%d  paper_id=%d", figure_id, paper_id)
        return figure_id

    def insert_extracted_data(
        self,
        figure_id: int,
        series_name: Optional[str] = None,
        x_label: Optional[str] = None,
        x_unit: Optional[str] = None,
        y_label: Optional[str] = None,
        y_unit: Optional[str] = None,
        data_points: Optional[list[dict]] = None,
        axis_scale_x: str = "linear",
        axis_scale_y: str = "linear",
        statistical_annotations: Optional[list[dict]] = None,
        extraction_method: Optional[str] = None,
        confidence_score: Optional[float] = None,
    ) -> int:
        """Insert an extracted data series and return its auto-assigned id."""
        dp_json = json.dumps(data_points or [])
        sa_json = json.dumps(statistical_annotations or [])
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO extracted_data
                    (figure_id, series_name, x_label, x_unit, y_label, y_unit,
                     data_points_json, axis_scale_x, axis_scale_y,
                     statistical_annotations_json, extraction_method, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id
                """,
                (
                    figure_id, series_name, x_label, x_unit, y_label, y_unit,
                    dp_json, axis_scale_x, axis_scale_y,
                    sa_json, extraction_method, confidence_score,
                ),
            )
            data_id: int = cur.fetchone()["id"]
        logger.debug("Inserted extracted_data id=%d  figure_id=%d", data_id, figure_id)
        return data_id

    def log_error(
        self,
        error_type: str,
        error_message: str,
        figure_id: Optional[int] = None,
    ) -> None:
        """Record an extraction error."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO extraction_errors (figure_id, error_type, error_message) VALUES (?, ?, ?)",
                (figure_id, error_type, error_message),
            )
        logger.warning("Logged error [%s]: %s", error_type, error_message)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_paper_figures(self, paper_id: int) -> list[dict[str, Any]]:
        """Return all figures for a given paper, including their extracted data."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM figures WHERE paper_id = ? ORDER BY page_number, figure_number",
                (paper_id,),
            ).fetchall()
        figures = []
        for row in rows:
            fig = dict(row)
            fig["extracted_data"] = self._get_series_for_figure(fig["id"])
            figures.append(fig)
        return figures

    def _get_series_for_figure(self, figure_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM extracted_data WHERE figure_id = ?",
                (figure_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_by_doi(self, doi: str) -> Optional[dict[str, Any]]:
        """Return the paper record matching a DOI, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE doi = ?", (doi,)
            ).fetchone()
        if row is None:
            return None
        paper = dict(row)
        paper["figures"] = self.get_paper_figures(paper["id"])
        return paper

    def get_all_papers(self) -> list[dict]:
        """Return all paper records (without their figures)."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM papers ORDER BY processed_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_paper_csv(self, paper_id: int, output_dir: Optional[Path] = None) -> list[Path]:
        """Export all extracted data for a paper to per-figure CSV files.

        Returns a list of paths to the created CSV files.
        """
        output_dir = Path(output_dir or OUTPUT_DIR) / f"paper_{paper_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        figures = self.get_paper_figures(paper_id)
        csv_paths: list[Path] = []

        for fig in figures:
            for series in fig.get("extracted_data", []):
                try:
                    points = json.loads(series.get("data_points_json") or "[]")
                except json.JSONDecodeError:
                    points = []

                if not points:
                    continue

                df = pd.DataFrame(points)
                fig_label = fig.get("figure_number") or str(fig["id"])
                panel = fig.get("panel_label") or ""
                series_name = (series.get("series_name") or "series").replace(" ", "_")
                fname = f"fig{fig_label}{panel}_{series_name}.csv"
                csv_path = output_dir / fname
                df.to_csv(csv_path, index=False)
                csv_paths.append(csv_path)
                logger.info("Exported %s", csv_path)

        return csv_paths

    def export_paper_json(self, paper_id: int, output_dir: Optional[Path] = None) -> Path:
        """Export all data for a paper as a single structured JSON file."""
        output_dir = Path(output_dir or OUTPUT_DIR) / f"paper_{paper_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            paper_row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        if paper_row is None:
            raise ValueError(f"No paper with id={paper_id}")

        paper_dict = dict(paper_row)
        paper_dict["figures"] = self.get_paper_figures(paper_id)

        out_path = output_dir / "full_extraction.json"
        out_path.write_text(json.dumps(paper_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Exported JSON → %s", out_path)
        return out_path
