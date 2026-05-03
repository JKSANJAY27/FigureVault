@echo off
:: =============================================================================
:: build_pdffigures2.bat — Clone and build the PDFFigures2 JAR from source
:: =============================================================================
:: Requires: Git, Java JDK 11+, sbt (Scala Build Tool)
::
:: After success, copies the assembled JAR to bin\pdffigures2.jar
:: =============================================================================

setlocal
echo.
echo ======================================================
echo  Building PDFFigures2 from allenai/pdffigures2
echo ======================================================
echo.

:: Check git
where git >nul 2>&1 || (
    echo [ERROR] git not found. Install Git from https://git-scm.com
    exit /b 1
)

:: Check sbt
where sbt >nul 2>&1 || (
    echo [ERROR] sbt not found.
    echo         Install the Scala Build Tool from https://www.scala-sbt.org/download.html
    echo         Then re-run this script.
    exit /b 1
)

if not exist "_build_tmp" mkdir _build_tmp

echo [1/3] Cloning allenai/pdffigures2...
git clone --depth 1 https://github.com/allenai/pdffigures2.git _build_tmp\pdffigures2
if errorlevel 1 (
    echo [ERROR] Clone failed. Check network connection.
    exit /b 1
)

echo [2/3] Building assembly JAR (this may take 5-10 minutes)...
pushd _build_tmp\pdffigures2
sbt assembly
if errorlevel 1 (
    echo [ERROR] sbt assembly failed. Check Java/sbt installation.
    popd
    exit /b 1
)
popd

echo [3/3] Copying JAR to bin\pdffigures2.jar...
if not exist bin mkdir bin
copy /Y "_build_tmp\pdffigures2\target\scala-*\pdffigures2-assembly-*.jar" "bin\pdffigures2.jar"
if errorlevel 1 (
    :: Try wildcard copy via PowerShell
    powershell -Command "Copy-Item (Get-ChildItem '_build_tmp\pdffigures2\target\scala-*\pdffigures2-assembly-*.jar' | Select-Object -First 1).FullName 'bin\pdffigures2.jar'"
)

if exist "bin\pdffigures2.jar" (
    echo.
    echo [OK] PDFFigures2 JAR built successfully: bin\pdffigures2.jar
    echo      You can now run: python main.py process ^<paper.pdf^>
) else (
    echo [ERROR] JAR not found after build. Check _build_tmp\pdffigures2\target\
    exit /b 1
)

:: Cleanup temp build dir (optional)
:: rmdir /s /q _build_tmp

endlocal
