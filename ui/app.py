"""
ui/app.py — FigureVault Streamlit Demo Interface

Launch with:
  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import OLLAMA_MODEL, OUTPUT_DIR
from database.db import DatabaseManager
from models.ollama_client import OllamaClient

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FigureVault",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=64)
    st.title("FigureVault")
    st.caption("AI-powered scientific figure data extraction")
    st.divider()

    st.subheader("⚙️ Settings")
    model_name = st.text_input("Ollama model", value=OLLAMA_MODEL)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.05)

    st.divider()
    # Ollama status indicator
    client = OllamaClient(model=model_name)
    if client.is_available():
        st.success("🟢 Ollama connected")
    else:
        st.error("🔴 Ollama not available")
        st.caption("Start Ollama and run: `ollama pull gemma4:e4b`")

    st.divider()
    st.subheader("📂 Recent Papers")
    db = DatabaseManager()
    db.init_db()
    papers = db.get_all_papers()
    if papers:
        for p in papers[:5]:
            st.write(f"• {p.get('title') or p.get('pdf_path','?')[:40]}")
    else:
        st.caption("No papers processed yet.")


# ---------------------------------------------------------------------------
# Main area — tabs
# ---------------------------------------------------------------------------
tab_process, tab_browse, tab_search, tab_about = st.tabs(
    ["🔬 Process PDF", "📊 Browse Results", "🔍 Search", "ℹ️ About"]
)

# ---- Tab 1: Process PDF ----
with tab_process:
    st.header("Extract Data from a Scientific Paper")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    col1, col2 = st.columns([2, 1])
    with col1:
        run_extraction = st.button("🚀 Run Extraction", type="primary", disabled=(uploaded is None))
    with col2:
        show_debug = st.checkbox("Show debug output")

    if run_extraction and uploaded is not None:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        with st.spinner("Running FigureVault pipeline…"):
            try:
                # Import pipeline lazily to avoid circular imports at startup
                from pipeline.pdf_parser import PDFParser
                from pipeline.figure_extractor import FigureExtractor
                from pipeline.classifier import FigureClassifier
                from pipeline.context_builder import ContextBuilder
                from pipeline.extractor import DataExtractor
                from pipeline.output_generator import OutputGenerator

                # Phase 1
                st.write("📄 **Phase 1:** Parsing PDF…")
                parser = PDFParser(tmp_path)
                meta = parser.parse()

                # Phase 2
                st.write("🖼️ **Phase 2:** Extracting figures…")
                fig_extractor = FigureExtractor(tmp_path, output_dir=OUTPUT_DIR / "ui_uploads")
                figures = fig_extractor.extract()
                st.write(f"   Found **{len(figures)}** figures")

                # Phase 3
                st.write("🏷️ **Phase 3:** Classifying figures…")
                clf = FigureClassifier(client=client, confidence_threshold=confidence_threshold)
                
                # Use a manual loop for classification to show progress
                progress_clf = st.progress(0, text="Classifying figures (0%)...")
                for i, fig in enumerate(figures):
                    clf.classify(fig)
                    progress_clf.progress((i + 1) / len(figures), text=f"Classified figure {i+1} of {len(figures)}...")

                # Phase 4 + 5
                st.write("🧠 **Phase 4–5:** Extracting data with Gemma4…")
                ctx_builder = ContextBuilder(meta)
                contexts = ctx_builder.build_all(figures)
                
                data_extractor = DataExtractor(client=client, confidence_threshold=confidence_threshold)
                series_map = {}
                
                # Use a manual loop for extraction to show progress
                progress_ext = st.progress(0, text="Extracting data from figures (0%)...")
                for i, ctx in enumerate(contexts):
                    fig_num = ctx.figure.figure_number
                    series_map[fig_num] = data_extractor.extract(ctx)
                    progress_ext.progress((i + 1) / len(contexts), text=f"Extracted data from figure {i+1} of {len(contexts)}...")

                # Phase 7/8
                st.write("💾 **Phase 7–8:** Generating outputs…")
                gen = OutputGenerator(
                    paper_meta=meta,
                    figures=figures,
                    series_map=series_map,
                    db=db,
                    output_dir=OUTPUT_DIR / "ui_uploads",
                )
                results = gen.generate_all()

                st.success(f"✅ Extraction complete! {len(results['csv'])} data series extracted.")

                # Display results
                st.divider()
                st.subheader("📊 Extracted Data")
                for fig in figures:
                    with st.expander(
                        f"Figure {fig.figure_number} — {getattr(fig, 'figure_type', '?')} "
                        f"(confidence: {fig.confidence:.0%})"
                    ):
                        if fig.image_path and fig.image_path.exists():
                            st.image(str(fig.image_path), caption=fig.caption or "", use_container_width=True)
                        series_list = series_map.get(fig.figure_number, [])
                        if series_list:
                            import pandas as pd
                            for s in series_list:
                                st.write(f"**{s.series_name or 'Series'}** — {s.y_label} vs {s.x_label}")
                                if s.data_points:
                                    df = pd.DataFrame(s.data_points)
                                    st.dataframe(df, use_container_width=True)
                                    csv_bytes = df.to_csv(index=False).encode()
                                    st.download_button(
                                        f"⬇️ Download {s.series_name or 'series'} CSV",
                                        csv_bytes,
                                        file_name=f"fig{fig.figure_number}_{s.series_name or 'series'}.csv",
                                    )
                        else:
                            st.info("No data series extracted for this figure.")

                if show_debug:
                    st.divider()
                    st.subheader("🐛 Debug: Full JSON")
                    if results["json"]:
                        st.json(json.loads(results["json"][0].read_text()))

            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                if show_debug:
                    st.exception(exc)

# ---- Tab 2: Browse Results ----
with tab_browse:
    st.header("Browse Extracted Papers")
    if not papers:
        st.info("No papers in the database yet. Process a PDF to get started.")
    else:
        import pandas as pd
        df_papers = pd.DataFrame(papers)
        display_cols = [c for c in ["id", "title", "doi", "journal", "year", "processed_at"] if c in df_papers.columns]
        st.dataframe(df_papers[display_cols], use_container_width=True)

        st.subheader("Inspect a Paper")
        paper_id = st.number_input("Paper ID", min_value=1, value=1, step=1)
        if st.button("Load paper"):
            figs = db.get_paper_figures(int(paper_id))
            if not figs:
                st.warning("No figures found for this paper ID.")
            else:
                for fig in figs:
                    with st.expander(f"Figure {fig['figure_number']} — {fig.get('figure_type','?')}"):
                        st.write(fig.get("caption", ""))
                        for s in fig.get("extracted_data", []):
                            pts = json.loads(s.get("data_points_json") or "[]")
                            if pts:
                                import pandas as pd
                                st.dataframe(pd.DataFrame(pts), use_container_width=True)

# ---- Tab 3: Search ----
with tab_search:
    st.header("Search by DOI")
    doi_input = st.text_input("Enter DOI", placeholder="10.1038/s41586-023-00001-2")
    if st.button("Search") and doi_input:
        result = db.search_by_doi(doi_input.strip())
        if result:
            st.success(f"Found: **{result.get('title','Unknown')}**")
            st.json(result)
        else:
            st.error(f"No paper found with DOI: {doi_input}")

# ---- Tab 4: About ----
with tab_about:
    st.header("About FigureVault")
    st.markdown("""
**FigureVault** is a local, offline AI system that reads scientific papers (PDFs)
and automatically extracts all quantitative data locked inside figures.

### How it works
1. **PDF Parser** — Extracts full text, metadata and page renders (PyMuPDF)
2. **Figure Extractor** — Finds figure images and captions (PDFFigures2 + PyMuPDF fallback)
3. **Classifier** — Labels each figure type using Gemma4 multimodal inference
4. **Context Builder** — Assembles caption + surrounding paper text
5. **Data Extractor** — Extracts structured JSON data using Gemma4
6. **Digitizer** — Pixel-level coordinate extraction using OpenCV
7. **Output Generator** — Produces CSV, JSON, SQLite with full provenance

### Technology
- **Core AI Model**: Gemma4 E4B via Ollama (local, private, no cloud)
- **Computer Vision**: OpenCV + scikit-image
- **Database**: SQLite with full provenance tracing
- **Output formats**: CSV, JSON, XLSX

### Gemma4 Impact Challenge
This project was built for the Gemma4 Impact Challenge. The key insight:
every number inside a published figure exists only as pixels. FigureVault
unlocks that data automatically using Gemma4's multimodal reasoning.
    """)
