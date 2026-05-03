#!/usr/bin/env bash
# =============================================================================
# setup.sh — FigureVault one-command environment setup
# =============================================================================
# Usage: bash setup.sh
# =============================================================================

set -e
BOLD=$(tput bold 2>/dev/null || true)
RESET=$(tput sgr0 2>/dev/null || true)
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

header() { echo -e "\n${BOLD}${GREEN}==> $1${NC}${RESET}"; }
warn()   { echo -e "${YELLOW}[WARN] $1${NC}"; }
error()  { echo -e "${RED}[ERROR] $1${NC}"; }
ok()     { echo -e "${GREEN}[OK]   $1${NC}"; }

header "FigureVault Environment Setup"

# ---------------------------------------------------------------------------
# 1. Python version check (>= 3.10)
# ---------------------------------------------------------------------------
header "Checking Python version"
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    error "Python not found. Install Python >= 3.10 from https://www.python.org"
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    error "Python $PY_VERSION detected. FigureVault requires Python >= 3.10"
    exit 1
fi
ok "Python $PY_VERSION"

# ---------------------------------------------------------------------------
# 2. Create virtual environment (optional but recommended)
# ---------------------------------------------------------------------------
header "Setting up virtual environment"
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    ok "Created .venv/"
else
    ok ".venv/ already exists"
fi

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

# ---------------------------------------------------------------------------
# 3. Install Python dependencies
# ---------------------------------------------------------------------------
header "Installing Python dependencies"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
ok "Python packages installed"

# ---------------------------------------------------------------------------
# 4. Check Java (required for PDFFigures2)
# ---------------------------------------------------------------------------
header "Checking Java installation"
if command -v java &>/dev/null; then
    JAVA_VER=$(java -version 2>&1 | head -n 1)
    ok "Java found: $JAVA_VER"
else
    warn "Java not found — PDFFigures2 will be unavailable."
    echo "  Install OpenJDK 11+:"
    echo "    Ubuntu/Debian : sudo apt-get install -y openjdk-11-jdk"
    echo "    macOS (brew)  : brew install openjdk@11"
    echo "    Windows       : https://adoptium.net"
    echo "  FigureVault will fall back to the PyMuPDF heuristic extractor."
fi

# ---------------------------------------------------------------------------
# 5. Download PDFFigures2 JAR
# ---------------------------------------------------------------------------
header "Setting up PDFFigures2"
JAR_DIR="bin"
JAR_PATH="$JAR_DIR/pdffigures2.jar"
JAR_URL="https://github.com/allenai/pdffigures2/releases/download/v0.1.0/pdffigures2-assembly-0.1.0.jar"

mkdir -p "$JAR_DIR"
if [ -f "$JAR_PATH" ]; then
    ok "PDFFigures2 JAR already present"
else
    if command -v curl &>/dev/null; then
        echo "  Downloading PDFFigures2 (this may take a minute)..."
        curl -L --silent --show-error -o "$JAR_PATH" "$JAR_URL" && ok "Downloaded $JAR_PATH" || warn "Download failed — you can manually download from $JAR_URL"
    elif command -v wget &>/dev/null; then
        wget -q -O "$JAR_PATH" "$JAR_URL" && ok "Downloaded $JAR_PATH" || warn "Download failed"
    else
        warn "Neither curl nor wget found. Download PDFFigures2 manually:"
        echo "  $JAR_URL → $JAR_PATH"
    fi
fi

# ---------------------------------------------------------------------------
# 6. Create output directories
# ---------------------------------------------------------------------------
header "Creating directory structure"
mkdir -p outputs/figures outputs/reports logs bin chroma_db training_data/raw training_data/synthetic
ok "Directories created"

# ---------------------------------------------------------------------------
# 7. Initialise SQLite database from schema
# ---------------------------------------------------------------------------
header "Initialising database"
$PYTHON -c "
from database.db import DatabaseManager
db = DatabaseManager()
db.init_db()
print('  Database initialised at:', db.db_path)
" && ok "Database ready" || warn "Database init failed — check Python environment"

# ---------------------------------------------------------------------------
# 8. Verify Ollama + Gemma4
# ---------------------------------------------------------------------------
header "Checking Ollama"
if command -v ollama &>/dev/null; then
    ok "Ollama binary found"
    # Try listing models
    MODELS=$(ollama list 2>/dev/null || echo "")
    if echo "$MODELS" | grep -qi "gemma"; then
        ok "A Gemma model is available in Ollama"
    else
        warn "No Gemma model found. Pull one with:"
        echo "  ollama pull gemma4:4b   (recommended — requires ~4GB VRAM)"
        echo "  ollama pull gemma3:4b   (alternative tag)"
    fi
else
    warn "Ollama not found. Install from https://ollama.ai and then run:"
    echo "  ollama pull gemma4:4b"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}✅ FigureVault setup complete!${NC}${RESET}"
echo ""
echo "Quick start:"
echo "  python main.py process <your_paper.pdf>"
echo "  streamlit run ui/app.py"
echo ""
