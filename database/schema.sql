-- =============================================================================
-- FigureVault SQLite Schema
-- =============================================================================
-- All JSON columns store UTF-8 JSON strings.
-- Timestamps are stored as ISO-8601 strings (YYYY-MM-DDTHH:MM:SS).
-- =============================================================================

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- papers
-- One row per processed PDF / publication.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS papers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doi             TEXT,                          -- e.g. "10.1038/s41586-023-00001-2"
    title           TEXT,
    authors         TEXT,                          -- JSON array of author strings
    journal         TEXT,
    year            INTEGER,
    pdf_path        TEXT    NOT NULL,              -- absolute path to source PDF
    processed_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(pdf_path)
);

-- ---------------------------------------------------------------------------
-- figures
-- One row per extracted figure (or sub-panel).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS figures (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id            INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    figure_number       TEXT,                      -- e.g. "3", "S2", "3b"
    panel_label         TEXT,                      -- e.g. "A", "B", "(i)"
    page_number         INTEGER,
    figure_type         TEXT,                      -- one of FIGURE_TYPES in config.py
    caption             TEXT,
    image_path          TEXT,                      -- absolute path to extracted PNG
    bounding_box_json   TEXT,                      -- {"x":..,"y":..,"w":..,"h":..}
    confidence_score    REAL,                      -- 0.0 – 1.0, from classifier
    processed_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_figures_paper_id ON figures(paper_id);
CREATE INDEX IF NOT EXISTS idx_figures_type     ON figures(figure_type);

-- ---------------------------------------------------------------------------
-- extracted_data
-- One row per data series within a figure.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS extracted_data (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    figure_id                       INTEGER NOT NULL REFERENCES figures(id) ON DELETE CASCADE,
    series_name                     TEXT,          -- e.g. "Control", "Treatment A"
    x_label                         TEXT,          -- e.g. "Time"
    x_unit                          TEXT,          -- e.g. "s", "min", "µM"
    y_label                         TEXT,          -- e.g. "Absorbance"
    y_unit                          TEXT,          -- e.g. "OD600", "a.u."
    data_points_json                TEXT,          -- [{"x":1.0,"y":2.3,"err_x":null,"err_y":0.1}, ...]
    axis_scale_x                    TEXT DEFAULT 'linear',   -- 'linear' | 'log'
    axis_scale_y                    TEXT DEFAULT 'linear',
    statistical_annotations_json    TEXT,          -- [{"marker":"*","x":3.0,"p_value":0.04}, ...]
    extraction_method               TEXT,          -- 'gemma4_multimodal' | 'opencv_digitizer' | 'hybrid'
    confidence_score                REAL           -- 0.0 – 1.0
);

CREATE INDEX IF NOT EXISTS idx_extracted_figure_id ON extracted_data(figure_id);

-- ---------------------------------------------------------------------------
-- extraction_errors
-- Records any error that occurred during extraction so we can retry/debug.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS extraction_errors (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    figure_id       INTEGER REFERENCES figures(id) ON DELETE SET NULL,
    error_type      TEXT,          -- e.g. 'ClassificationError', 'OllamaTimeout'
    error_message   TEXT,
    timestamp       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_errors_figure_id ON extraction_errors(figure_id);
