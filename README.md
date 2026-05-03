# FigureVault — AI-powered Scientific Figure Data Extraction

> **Gemma4 Impact Challenge Submission**  
> Local, offline AI that unlocks quantitative data trapped inside scientific figures.

---

## The Problem

Every number inside a published scientific figure exists only as pixels.  
A meta-analyst aggregating data from 500 papers must manually click each data point in WebPlotDigitizer — **months of work**.

**FigureVault does it automatically**, understanding not just the pixels but the scientific context — because Gemma4 E4B reads the caption *and* sees the figure simultaneously.

---

## Architecture

```
PDF Input
    │
    ▼
[Phase 1] PDF Parser (PyMuPDF)
    │  → full text, metadata, page renders
    ▼
[Phase 2] Figure Extractor (PDFFigures2 + PyMuPDF fallback)
    │  → figure PNGs, captions, bounding boxes
    ▼
[Phase 3] Figure Classifier (Gemma4 E4B via Ollama)
    │  → line_plot | bar_chart | scatter | heatmap | spectrum | gel | microscopy | ...
    ▼
[Phase 4] Context Builder
    │  → figure image + caption + surrounding paper text
    ▼
[Phase 5] Data Extractor (Gemma4 E4B — the core model)
    │  → axis labels, units, scale, series names, (x,y) points, error bars, p-values
    ▼
[Phase 6] Pixel Digitizer (OpenCV)
    │  → precise pixel-level coordinate extraction
    ▼
[Phase 7] Provenance Packager
    │  → data + DOI + figure number + panel + caption + confidence score
    ▼
[Phase 8] Output Generator
       → CSV per figure | JSON with metadata | SQLite database | summary report
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Java 11+ (for PDFFigures2; optional — PyMuPDF fallback is available)

### Installation

```bash
git clone https://github.com/JKSANJAY27/FigureVault.git
cd FigureVault
bash setup.sh
```

Or manually:

```bash
pip install -r requirements.txt
ollama pull gemma4:4b
python -c "from database.db import DatabaseManager; DatabaseManager().init_db()"
```

### Process a Paper

```bash
# Single PDF
python main.py process path/to/paper.pdf --output results/

# Batch processing
python main.py process-batch papers/ --output results/

# Search by DOI
python main.py search --doi 10.1038/s41586-023-00001-2

# Export results
python main.py export --paper-id 1 --format csv
python main.py export --paper-id 1 --format xlsx
```

### Launch Demo UI

```bash
streamlit run ui/app.py
```

---

## Project Structure

```
figurevault/
├── main.py                    # CLI entry point
├── config.py                  # All configuration constants
├── requirements.txt
├── setup.sh                   # One-command environment setup
├── pipeline/
│   ├── pdf_parser.py          # Phase 1: PDF text + metadata
│   ├── figure_extractor.py    # Phase 2: Figure image extraction
│   ├── classifier.py          # Phase 3: Figure type classification
│   ├── context_builder.py     # Phase 4: Context assembly
│   ├── extractor.py           # Phase 5: Gemma4 data extraction
│   ├── digitizer.py           # Phase 6: OpenCV pixel digitization
│   └── output_generator.py    # Phase 7/8: CSV/JSON/SQLite output
├── models/
│   └── ollama_client.py       # Ollama API wrapper (multimodal)
├── training/
│   ├── data_collector.py      # Collect training pairs from open datasets
│   ├── synthetic_generator.py # Generate synthetic figure+CSV pairs
│   └── finetune.py            # Unsloth fine-tuning script
├── eval/
│   ├── benchmark.py           # Accuracy benchmarking
│   └── metrics.py             # RMSE, R², precision/recall
├── ui/
│   └── app.py                 # Streamlit demo
├── database/
│   ├── schema.sql             # SQLite schema
│   └── db.py                  # DatabaseManager class
└── tests/
    └── test_pipeline.py       # Smoke tests (pytest)
```

---

## Technology Stack

| Component | Tool | Why |
|-----------|------|-----|
| PDF parsing | PyMuPDF (fitz) | Fastest, most reliable |
| Figure extraction | PDFFigures2 (Java) + PyMuPDF fallback | Battle-tested on 1M+ papers |
| Core AI | Gemma4 E4B | Multimodal, fine-tunable, runs locally |
| Model serving | Ollama | Zero-config local API |
| Fine-tuning | Unsloth + FastVisionModel | Only framework supporting Gemma4 E4B multimodal |
| Computer vision | OpenCV + scikit-image | Axis detection, colour segmentation |
| Vector DB | ChromaDB | Figure embedding similarity search |
| Database | SQLite | Local, zero-config, full provenance |
| Backend | FastAPI | Serves the processing pipeline |
| Demo UI | Streamlit | Fastest to build, visually impressive |
| Output | pandas → CSV / JSON / Excel | Researchers need all three |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Fine-Tuning (Advanced)

Fine-tuning runs on Kaggle T4 GPU or Google Colab:

```bash
# Collect training data
python training/data_collector.py

# Generate synthetic pairs
python -c "from training.synthetic_generator import SyntheticGenerator; SyntheticGenerator().generate(500)"

# Fine-tune (requires GPU + Unsloth)
python training/finetune.py --epochs 3 --lora-rank 16
```

---

## Training Datasets

| Source | Type | Target pairs |
|--------|------|-------------|
| [eLife API](https://api.elifesciences.org) | Real papers + raw data | 1,000+ |
| [PLOS ONE](https://plos.org) | Real papers + raw data | 1,000+ |
| [Figshare](https://figshare.com) | Open datasets | 2,000+ |
| [Zenodo](https://zenodo.org) | Multi-domain | 500+ |
| Synthetic (matplotlib) | Perfectly labelled | Unlimited |
| [FigureSeer](https://nlp.stanford.edu/projects/figureseer/) | 60K labelled figures | Classification |
| [SciCap](https://github.com/tingyaohsu/SciCap) | 410K arXiv figures | Caption understanding |

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Citation

If you use FigureVault in your research, please cite:

```
@software{figurevault2025,
  title  = {FigureVault: AI-powered Scientific Figure Data Extraction},
  author = {Sanjay, J.K.},
  year   = {2025},
  url    = {https://github.com/JKSANJAY27/FigureVault}
}
```
