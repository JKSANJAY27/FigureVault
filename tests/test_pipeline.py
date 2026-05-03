"""
tests/test_pipeline.py — FigureVault smoke tests

Run with:
  python -m pytest tests/ -v
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    """Verify config constants are sensible."""

    def test_figure_types_nonempty(self):
        from config import FIGURE_TYPES
        assert len(FIGURE_TYPES) > 0

    def test_confidence_threshold_range(self):
        from config import CONFIDENCE_THRESHOLD
        assert 0 < CONFIDENCE_THRESHOLD < 1

    def test_image_dpi_positive(self):
        from config import IMAGE_DPI
        assert IMAGE_DPI > 0

    def test_db_path_is_path(self):
        from config import DB_PATH
        assert isinstance(DB_PATH, Path)


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestDatabaseManager:
    """Smoke tests for DatabaseManager CRUD operations."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a fresh in-memory-style database for each test."""
        from database.db import DatabaseManager
        db = DatabaseManager(db_path=tmp_path / "test.db")
        db.init_db()
        return db

    def test_init_creates_tables(self, db):
        conn = sqlite3.connect(db.db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert {"papers", "figures", "extracted_data", "extraction_errors"}.issubset(tables)

    def test_insert_paper_returns_id(self, db):
        pid = db.insert_paper(pdf_path="/fake/paper.pdf", doi="10.1234/test", title="Test Paper")
        assert isinstance(pid, int)
        assert pid > 0

    def test_insert_paper_upsert(self, db):
        pid1 = db.insert_paper(pdf_path="/fake/paper.pdf", title="Original")
        pid2 = db.insert_paper(pdf_path="/fake/paper.pdf", title="Updated")
        assert pid1 == pid2  # same row, upserted

    def test_insert_figure(self, db):
        pid = db.insert_paper(pdf_path="/fake/p.pdf")
        fid = db.insert_figure(paper_id=pid, figure_number="1", figure_type="line_plot")
        assert isinstance(fid, int)

    def test_insert_extracted_data(self, db):
        pid = db.insert_paper(pdf_path="/fake/p2.pdf")
        fid = db.insert_figure(paper_id=pid, figure_number="2")
        did = db.insert_extracted_data(
            figure_id=fid,
            series_name="Control",
            x_label="Time",
            x_unit="s",
            y_label="Absorbance",
            data_points=[{"x": 0, "y": 0.1}, {"x": 1, "y": 0.2}],
        )
        assert isinstance(did, int)

    def test_search_by_doi(self, db):
        db.insert_paper(pdf_path="/fake/doi.pdf", doi="10.9999/xyz")
        result = db.search_by_doi("10.9999/xyz")
        assert result is not None
        assert result["doi"] == "10.9999/xyz"

    def test_search_by_doi_not_found(self, db):
        result = db.search_by_doi("10.0000/not-exist")
        assert result is None

    def test_export_paper_csv(self, db, tmp_path):
        pid = db.insert_paper(pdf_path="/fake/export.pdf")
        fid = db.insert_figure(paper_id=pid, figure_number="3")
        db.insert_extracted_data(
            figure_id=fid,
            series_name="Series A",
            data_points=[{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 3.0}],
        )
        paths = db.export_paper_csv(paper_id=pid, output_dir=tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()


# ---------------------------------------------------------------------------
# OllamaClient tests (mocked — no real Ollama needed)
# ---------------------------------------------------------------------------

class TestOllamaClient:
    """Test OllamaClient with mocked HTTP responses."""

    def test_is_available_true(self):
        from models.ollama_client import OllamaClient
        client = OllamaClient(model="gemma4:4b")
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value.json.return_value = {
                "models": [{"name": "gemma4:4b"}]
            }
            mock_get.return_value.raise_for_status = MagicMock()
            assert client.is_available() is True

    def test_is_available_false_no_model(self):
        from models.ollama_client import OllamaClient
        client = OllamaClient(model="gemma4:4b")
        with patch.object(client._session, "get") as mock_get:
            mock_get.return_value.json.return_value = {"models": [{"name": "llama3:8b"}]}
            mock_get.return_value.raise_for_status = MagicMock()
            assert client.is_available() is False

    def test_query_text_returns_string(self):
        from models.ollama_client import OllamaClient
        client = OllamaClient()
        with patch.object(client._session, "post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "hello"}
            mock_post.return_value.raise_for_status = MagicMock()
            result = client.query_text("test prompt")
        assert result == "hello"

    def test_query_multimodal_encodes_image(self, tmp_path):
        from models.ollama_client import OllamaClient
        # Create a tiny 1-byte fake PNG
        fake_img = tmp_path / "fig.png"
        fake_img.write_bytes(b"\x89PNG\r\n\x1a\n")

        client = OllamaClient()
        with patch.object(client._session, "post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "data: []"}
            mock_post.return_value.raise_for_status = MagicMock()
            result = client.query_multimodal("describe", image_path=fake_img)

        # Verify images field was sent
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert "images" in payload
        assert len(payload["images"]) == 1
        assert isinstance(payload["images"][0], str)  # base64 string


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Unit tests for eval metrics."""

    def test_rmse_identical(self):
        from eval.metrics import compute_rmse
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known(self):
        from eval.metrics import compute_rmse
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.5, 0.5])
        rmse = compute_rmse(y_true, y_pred)
        assert 0 < rmse <= 1.0

    def test_r2_perfect(self):
        from eval.metrics import compute_r2
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert compute_r2(y, y) == pytest.approx(1.0)

    def test_r2_empty(self):
        from eval.metrics import compute_r2
        assert compute_r2(np.array([]), np.array([])) == 0.0

    def test_match_series_exact_name(self):
        from eval.metrics import match_series
        gt = [{"series_name": "Control", "data_points": []}]
        pred = [{"series_name": "Control", "data_points": []}]
        pairs = match_series(gt, pred)
        assert len(pairs) == 1
        assert pairs[0][0]["series_name"] == "Control"

    def test_match_series_positional_fallback(self):
        from eval.metrics import match_series
        gt = [{"series_name": "A"}, {"series_name": "B"}]
        pred = [{"series_name": "X"}, {"series_name": "Y"}]
        pairs = match_series(gt, pred)
        assert len(pairs) == 2


# ---------------------------------------------------------------------------
# Classifier tests (mocked Ollama)
# ---------------------------------------------------------------------------

class TestFigureClassifier:
    """Test classifier parsing logic without real Ollama."""

    def test_parse_valid_json(self):
        from pipeline.classifier import FigureClassifier
        clf = FigureClassifier(client=MagicMock())
        label, conf = clf._parse_response('{"label":"line_plot","confidence":0.9,"reasoning":"looks like a line"}')
        assert label == "line_plot"
        assert conf == pytest.approx(0.9)

    def test_parse_embedded_json(self):
        from pipeline.classifier import FigureClassifier
        clf = FigureClassifier(client=MagicMock())
        raw = 'Sure! Here is the result: {"label":"bar_chart","confidence":0.8,"reasoning":"bars"}'
        label, conf = clf._parse_response(raw)
        assert label == "bar_chart"

    def test_parse_invalid_falls_back(self):
        from pipeline.classifier import FigureClassifier
        clf = FigureClassifier(client=MagicMock())
        label, conf = clf._parse_response("I think this is a scatter plot")
        assert label in ["scatter_plot", "other"]


# ---------------------------------------------------------------------------
# Context builder tests
# ---------------------------------------------------------------------------

class TestContextBuilder:
    """Test context assembly."""

    def test_build_includes_caption(self):
        from pipeline.context_builder import ContextBuilder
        from pipeline.pdf_parser import PaperMetadata
        from pipeline.figure_extractor import FigureRecord

        meta = PaperMetadata(
            pdf_path="fake.pdf",
            title="Test Paper",
            page_texts=["Figure 1 shows a bar chart of cell viability."],
        )
        meta.page_count = 1
        fig = FigureRecord(page_number=1, figure_number="1", caption="Bar chart of viability.")
        ctx = ContextBuilder(meta).build(fig)
        assert "viability" in ctx.build_extraction_prompt()
