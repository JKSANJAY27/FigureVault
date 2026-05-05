"""
main.py — FigureVault CLI entry point

Usage examples:
  python main.py process paper.pdf --output ./results
  python main.py process-batch ./papers/ --output ./results
  python main.py search --doi 10.1038/s41586-023-00001-2
  python main.py export --paper-id 3 --format csv
  python main.py benchmark --test-set ./test_data/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import DB_PATH, LOG_FILE, LOG_LEVEL, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Logging setup (must happen before any imports that use logging)
# ---------------------------------------------------------------------------
def _setup_logging(level: str = LOG_LEVEL) -> None:
    """Configure root logger to write to both stdout and a log file."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )


logger = logging.getLogger("figurevault.main")


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_process(args: argparse.Namespace) -> int:
    """Process a single PDF through the full FigureVault pipeline."""
    from database.db import DatabaseManager
    from models.ollama_client import OllamaClient
    from pipeline.pdf_parser import PDFParser
    from pipeline.figure_extractor import FigureExtractor
    from pipeline.classifier import FigureClassifier
    from pipeline.context_builder import ContextBuilder
    from pipeline.extractor import DataExtractor
    from pipeline.output_generator import OutputGenerator

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return 1

    out_dir = Path(args.output or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check Ollama availability
    client = OllamaClient()
    if not client.is_available():
        logger.error(
            "Ollama is not running or the model is not loaded. "
            "Start Ollama and pull the model with: ollama pull gemma4:e4b"
        )
        return 1

    logger.info("Starting pipeline for: %s", pdf_path)

    # Phase 1 — PDF parsing
    parser = PDFParser(pdf_path)
    paper_meta = parser.parse()

    # Phase 2 — Figure extraction
    extractor = FigureExtractor()
    figures = extractor.extract_all(pdf_path, output_dir=out_dir)
    logger.info("Extracted %d figures", len(figures))

    # Phase 3 — Classification
    classifier = FigureClassifier(client=client)
    classifier.classify_figure_records(figures)

    # Phase 4 — Context building
    ctx_builder = ContextBuilder(paper_meta)
    contexts = ctx_builder.build_all(figures)

    # Phase 5 — Data extraction
    data_extractor = DataExtractor(client=client)
    series_map = data_extractor.extract_batch(contexts)

    # Phase 7/8 — Output generation
    db = DatabaseManager()
    db.init_db()
    gen = OutputGenerator(
        paper_meta=paper_meta,
        figures=figures,
        series_map=series_map,
        db=db,
        output_dir=out_dir,
    )
    results = gen.generate_all()

    print("\n✅ Processing complete!")
    print(f"   CSV files  : {len(results['csv'])}")
    print(f"   JSON output: {results['json'][0]}")
    print(f"   Report     : {results['report'][0]}")
    return 0


def cmd_process_batch(args: argparse.Namespace) -> int:
    """Process all PDFs in a directory."""
    pdf_dir = Path(args.pdf_directory)
    if not pdf_dir.is_dir():
        logger.error("Directory not found: %s", pdf_dir)
        return 1

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return 0

    logger.info("Found %d PDF files — processing batch", len(pdf_files))
    errors = 0
    for i, pdf in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] {pdf.name}")
        # Re-use the single-file handler
        ns = argparse.Namespace(pdf_path=str(pdf), output=args.output)
        rc = cmd_process(ns)
        if rc != 0:
            errors += 1

    print(f"\n✅ Batch complete. {len(pdf_files) - errors}/{len(pdf_files)} succeeded.")
    return 0 if errors == 0 else 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search the database by DOI."""
    from database.db import DatabaseManager
    import json

    db = DatabaseManager()
    db.init_db()
    result = db.search_by_doi(args.doi)

    if result is None:
        print(f"No paper found with DOI: {args.doi}")
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export extracted data for a paper."""
    from database.db import DatabaseManager

    db = DatabaseManager()
    db.init_db()
    paper_id = int(args.paper_id)
    fmt = args.format.lower()
    out_dir = Path(args.output or OUTPUT_DIR)

    if fmt == "csv":
        paths = db.export_paper_csv(paper_id, output_dir=out_dir)
        print(f"Exported {len(paths)} CSV files to {out_dir}")
    elif fmt == "json":
        path = db.export_paper_json(paper_id, output_dir=out_dir)
        print(f"Exported JSON to {path}")
    elif fmt == "xlsx":
        import pandas as pd
        import json
        csv_paths = db.export_paper_csv(paper_id, output_dir=out_dir)
        if csv_paths:
            xlsx_path = out_dir / f"paper_{paper_id}_all.xlsx"
            with pd.ExcelWriter(xlsx_path) as writer:
                for csv_path in csv_paths:
                    df = pd.read_csv(csv_path)
                    sheet = csv_path.stem[:31]  # Excel sheet name max 31 chars
                    df.to_excel(writer, sheet_name=sheet, index=False)
            print(f"Exported XLSX to {xlsx_path}")
    else:
        print(f"Unknown format: {fmt}. Use csv, json, or xlsx.")
        return 1

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run benchmarking against a ground-truth test set."""
    from eval.benchmark import Benchmarker

    test_dir = Path(args.test_set)
    if not test_dir.is_dir():
        logger.error("Test set directory not found: %s", test_dir)
        return 1

    benchmarker = Benchmarker(test_dir=test_dir)
    results = benchmarker.run()
    benchmarker.print_report(results)
    return 0


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="figurevault",
        description="FigureVault — AI-powered scientific figure data extraction",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: %(default)s)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- process ----
    p_proc = sub.add_parser("process", help="Process a single PDF")
    p_proc.add_argument("pdf_path", help="Path to the input PDF")
    p_proc.add_argument("--output", "-o", default=None, help="Output directory")

    # ---- process-batch ----
    p_batch = sub.add_parser("process-batch", help="Process all PDFs in a directory")
    p_batch.add_argument("pdf_directory", help="Directory containing PDFs")
    p_batch.add_argument("--output", "-o", default=None, help="Output directory")

    # ---- search ----
    p_search = sub.add_parser("search", help="Search the database by DOI")
    p_search.add_argument("--doi", required=True, help="DOI string to look up")

    # ---- export ----
    p_export = sub.add_parser("export", help="Export data for a paper")
    p_export.add_argument("--paper-id", required=True, help="Paper ID in the database")
    p_export.add_argument(
        "--format", default="csv", choices=["csv", "json", "xlsx"],
        help="Output format (default: csv)",
    )
    p_export.add_argument("--output", "-o", default=None, help="Output directory")

    # ---- benchmark ----
    p_bench = sub.add_parser("benchmark", help="Benchmark against ground-truth test set")
    p_bench.add_argument("--test-set", required=True, help="Directory of test figure+data pairs")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)

    handlers = {
        "process": cmd_process,
        "process-batch": cmd_process_batch,
        "search": cmd_search,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
