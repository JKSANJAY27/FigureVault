@echo off
:: =============================================================================
:: setup.bat — FigureVault one-command environment setup for Windows
:: =============================================================================
:: Usage: setup.bat
:: =============================================================================

setlocal EnableDelayedExpansion
set "RED=[31m"
set "GREEN=[32m"
set "YELLOW=[33m"
set "BOLD=[1m"
set "RESET=[0m"

echo.
echo %BOLD%%GREEN%=^> FigureVault Environment Setup (Windows)%RESET%
echo.

:: ---------------------------------------------------------------------------
:: 1. Python version check (>= 3.10)
:: ---------------------------------------------------------------------------
echo %BOLD%%GREEN%=^> Checking Python version%RESET%
where python >nul 2>&1
if errorlevel 1 (
    echo %RED%[ERROR] Python not found. Install Python ^>= 3.10 from https://www.python.org%RESET%
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_VER=%%v
for /f "tokens=1 delims=." %%a in ("!PY_VER!") do set PY_MAJOR=%%a
for /f "tokens=2 delims=." %%b in ("!PY_VER!") do set PY_MINOR=%%b

if !PY_MAJOR! LSS 3 (
    echo %RED%[ERROR] Python !PY_VER! detected. FigureVault requires Python ^>= 3.10%RESET%
    exit /b 1
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 10 (
    echo %RED%[ERROR] Python !PY_VER! detected. FigureVault requires Python ^>= 3.10%RESET%
    exit /b 1
)
echo %GREEN%[OK]   Python !PY_VER!%RESET%

:: ---------------------------------------------------------------------------
:: 2. Create virtual environment
:: ---------------------------------------------------------------------------
echo.
echo %BOLD%%GREEN%=^> Setting up virtual environment%RESET%
if not exist ".venv" (
    python -m venv .venv
    echo %GREEN%[OK]   Created .venv\%RESET%
) else (
    echo %GREEN%[OK]   .venv\ already exists%RESET%
)
call .venv\Scripts\activate.bat

:: ---------------------------------------------------------------------------
:: 3. Install Python dependencies
:: ---------------------------------------------------------------------------
echo.
echo %BOLD%%GREEN%=^> Installing Python dependencies%RESET%
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo %RED%[ERROR] pip install failed. Check requirements.txt%RESET%
    exit /b 1
)
echo %GREEN%[OK]   Python packages installed%RESET%

:: ---------------------------------------------------------------------------
:: 4. Check Java
:: ---------------------------------------------------------------------------
echo.
echo %BOLD%%GREEN%=^> Checking Java installation%RESET%
where java >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%[WARN] Java not found — PDFFigures2 will be unavailable.%RESET%
    echo        Install OpenJDK 11+ from https://adoptium.net
    echo        Then run: build_pdffigures2.bat
    echo        FigureVault will fall back to the PyMuPDF heuristic extractor.
) else (
    for /f "tokens=*" %%j in ('java -version 2^>^&1 ^| findstr "version"') do echo %GREEN%[OK]   Java: %%j%RESET%
)

:: ---------------------------------------------------------------------------
:: 5. Check PDFFigures2 JAR
:: ---------------------------------------------------------------------------
echo.
echo %BOLD%%GREEN%=^> Checking PDFFigures2%RESET%
if not exist "bin" mkdir bin
if exist "bin\pdffigures2.jar" (
    echo %GREEN%[OK]   PDFFigures2 JAR found at bin\pdffigures2.jar%RESET%
) else (
    echo %YELLOW%[WARN] PDFFigures2 JAR not found.%RESET%
    echo        To build it, run: build_pdffigures2.bat
    echo        FigureVault will use the PyMuPDF fallback extractor until then.
)

:: ---------------------------------------------------------------------------
:: 6. Create output directories
:: ---------------------------------------------------------------------------
echo.
echo %BOLD%%GREEN%=^> Creating directory structure%RESET%
for %%d in (outputs\figures outputs\reports logs bin chroma_db training_data\raw training_data\synthetic) do (
    if not exist "%%d" mkdir "%%d"
)
echo %GREEN%[OK]   Directories created%RESET%

:: ---------------------------------------------------------------------------
:: 7. Initialise SQLite database
:: ---------------------------------------------------------------------------
echo.
echo %BOLD%%GREEN%=^> Initialising database%RESET%
python -c "from database.db import DatabaseManager; db = DatabaseManager(); db.init_db(); print('  DB at:', db.db_path)"
if errorlevel 1 (
    echo %YELLOW%[WARN] Database init failed — check Python environment%RESET%
) else (
    echo %GREEN%[OK]   Database ready%RESET%
)

:: ---------------------------------------------------------------------------
:: 8. Verify Ollama
:: ---------------------------------------------------------------------------
echo.
echo %BOLD%%GREEN%=^> Checking Ollama%RESET%
where ollama >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%[WARN] Ollama not found. Install from https://ollama.ai then run:%RESET%
    echo        ollama pull gemma4:e4b
) else (
    echo %GREEN%[OK]   Ollama binary found%RESET%
    ollama list 2>nul | findstr /i "gemma" >nul
    if errorlevel 1 (
        echo %YELLOW%[WARN] No Gemma model found. Pull one with:%RESET%
        echo        ollama pull gemma4:e4b
    ) else (
        echo %GREEN%[OK]   Gemma model available in Ollama%RESET%
    )
)

:: ---------------------------------------------------------------------------
:: Done
:: ---------------------------------------------------------------------------
echo.
echo %BOLD%%GREEN%✅ FigureVault setup complete!%RESET%
echo.
echo Quick start:
echo   python main.py process ^<your_paper.pdf^>
echo   streamlit run ui\app.py
echo.
endlocal
