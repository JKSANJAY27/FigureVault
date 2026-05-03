"""
config.py — FigureVault Central Configuration

All tunable constants live here.  Import from anywhere in the project:
    from config import OLLAMA_MODEL, FIGURE_TYPES, ...
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (so relative paths work regardless of CWD)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# Ollama / Model settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# The multimodal Gemma 4 model tag as registered in Ollama.
# Run `ollama list` to confirm the exact tag on your machine.
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma4:4b")

# Timeout in seconds for a single Ollama API call
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# Number of retries on transient failure
OLLAMA_MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Figure classification taxonomy
# ---------------------------------------------------------------------------
FIGURE_TYPES: list[str] = [
    "line_plot",
    "bar_chart",
    "scatter_plot",
    "heatmap",
    "spectrum_nmr",
    "spectrum_ir",
    "spectrum_ms",
    "western_blot",
    "microscopy",
    "table",
    "diagram",
    "other",
]

# ---------------------------------------------------------------------------
# Pipeline settings
# ---------------------------------------------------------------------------
# Minimum confidence (0–1) to accept an extraction result
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))

# Hard cap on figures processed per PDF (prevents runaway jobs on massive papers)
MAX_FIGURES_PER_PAPER: int = int(os.getenv("MAX_FIGURES_PER_PAPER", "50"))

# DPI used when rendering PDF pages to images
IMAGE_DPI: int = int(os.getenv("IMAGE_DPI", "300"))

# Minimum bounding-box area (pixels²) for a region to be considered a figure
MIN_FIGURE_AREA_PX: int = 10_000

# ---------------------------------------------------------------------------
# PDFFigures2 (Java JAR)
# ---------------------------------------------------------------------------
PDFFIGURES2_JAR: Path = PROJECT_ROOT / "bin" / "pdffigures2.jar"
PDFFIGURES2_RELEASE_URL: str = (
    "https://github.com/allenai/pdffigures2/releases/download/v0.1.0/pdffigures2-assembly-0.1.0.jar"
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: Path = PROJECT_ROOT / os.getenv("DB_NAME", "figurevault.db")

# ---------------------------------------------------------------------------
# File-system layout
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
REPORTS_DIR: Path = OUTPUT_DIR / "reports"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Create directories at import time so callers never have to check
for _d in (OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR, PROJECT_ROOT / "bin"):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = LOGS_DIR / "figurevault.log"

# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
# Acceptable RMSE for a digitization to be counted as "correct"
BENCHMARK_RMSE_THRESHOLD: float = 0.05

# ---------------------------------------------------------------------------
# ChromaDB (vector store for figure embeddings)
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / "chroma_db"
CHROMA_COLLECTION_NAME: str = "figure_embeddings"
