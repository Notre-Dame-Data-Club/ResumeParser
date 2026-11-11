@echo off
REM Windows Resume Processing - Setup
REM

setlocal enabledelayedexpansion
pushd "%~dp0"

set "LOG_FILE=%CD%\setup.log"
set "SETUP_PIP_LOG=%CD%\setup-pip.log"
set "TMP_OUT=%TEMP%\sr_setup_out.txt"
set "CUDA_INDEX=https://download.pytorch.org/whl/cu126"
set TOTAL=4

cls
echo.
echo ============================================================
echo Windows Resume Processing - Setup
echo ============================================================
echo Log file: %LOG_FILE%
echo.
echo [%date% %time%] ==== setup started ====>>"%LOG_FILE%"

REM ============================================================
echo [1/%TOTAL%] Preflight checks
REM ============================================================
python --version >nul 2>&1
if errorlevel 1 (
    set "ERR=Python not found. Install Python 3.9 or newer from python.org, ticking 'Add Python to PATH'."
    goto :die
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if errorlevel 1 (
    set "ERR=Python 3.9 or newer is required. Install a newer Python from python.org."
    goto :die
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set "PYTHON_VER=%%v"
echo   %PYTHON_VER%
echo [%date% %time%] %PYTHON_VER%>>"%LOG_FILE%"

if not exist "pyproject.toml" (
    set "ERR=pyproject.toml not found. Run setup.bat from the project root folder."
    goto :die
)
if not exist "scripts\check_deps.py" (
    set "ERR=scripts\check_deps.py not found. The project files are incomplete."
    goto :die
)

if exist "configs\config.yaml" (
    echo   configs\config.yaml present - left as-is.
) else (
    if exist "configs\config.yaml.example" (
        copy /y "configs\config.yaml.example" "configs\config.yaml" >nul
        echo   configs\config.yaml created from the example.
    ) else (
        echo   WARNING: configs\config.yaml.example missing - config not created.
    )
)

if not exist "scripts\download_models.py" (
    echo   WARNING: scripts\download_models.py missing - step 4 will be skipped.
)

REM ============================================================
echo.
echo [2/%TOTAL%] Dependencies
REM ============================================================
python scripts\check_deps.py > "%TMP_OUT%" 2>&1
set DEPS_RC=!errorlevel!
type "%TMP_OUT%"
type "%TMP_OUT%">>"%LOG_FILE%"
del "%TMP_OUT%" >nul 2>&1

if !DEPS_RC! EQU 0 (
    echo   Nothing to install - skipping.
    goto :verify
)

set /a "NEED_TORCH=DEPS_RC & 2"

echo   Updating pip...
python -m pip install --upgrade pip >>"%LOG_FILE%" 2>&1

if !NEED_TORCH! NEQ 0 (
    echo   Installing torch + torchvision, CUDA 12.6  -  ~2.6 GB, 3-10 min
    echo   ----------------------------------------------------------
    python -m pip install --force-reinstall --log "%SETUP_PIP_LOG%" torch torchvision --index-url %CUDA_INDEX%
    if errorlevel 1 (
        set "ERR=torch CUDA install failed. See %SETUP_PIP_LOG%."
        goto :die
    )
    echo   ----------------------------------------------------------
)

echo   Installing project and remaining requirements...
python -m pip install -e . --log "%SETUP_PIP_LOG%" >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -Command "Get-Content -LiteralPath '%LOG_FILE%' -Tail 8"
    set "ERR=Dependency install failed. See %SETUP_PIP_LOG%."
    goto :die
)
echo   Done.

REM ============================================================
:verify
echo.
echo [3/%TOTAL%] Verifying environment
REM ============================================================
python scripts\check_deps.py --gpu > "%TMP_OUT%" 2>&1
set VERIFY_RC=!errorlevel!
type "%TMP_OUT%"
type "%TMP_OUT%">>"%LOG_FILE%"
del "%TMP_OUT%" >nul 2>&1
if !VERIFY_RC! NEQ 0 (
    set "ERR=Verification failed. See the report above and %LOG_FILE%."
    goto :die
)

python -c "from smartresume.backend.resume_analyzer import ResumeAnalyzer" >>"%LOG_FILE%" 2>&1
if errorlevel 1 (
    set "ERR=SmartResume failed to import. See %LOG_FILE%."
    goto :die
)
echo   SmartResume imports cleanly.

REM ============================================================
echo.
echo [4/%TOTAL%] Models - Gemma 4 E4B-it (~16 GB) + YOLOv10
REM ============================================================
if not exist "scripts\download_models.py" (
    echo   SKIPPED - scripts\download_models.py not found.
    echo [%date% %time%] model download skipped - script missing>>"%LOG_FILE%"
    goto :done
)
echo   Downloading from Hugging Face; already-downloaded files are reused.
echo   ----------------------------------------------------------
python scripts\download_models.py
if errorlevel 1 (
    set "ERR=Model download failed. Check your internet connection, then retry: python scripts\download_models.py"
    goto :die
)
echo   ----------------------------------------------------------
echo   Models ready.
echo [%date% %time%] models downloaded>>"%LOG_FILE%"

REM ============================================================
:done
echo.
echo ============================================================
echo Setup complete
echo ============================================================
echo.
echo Next steps:
echo   1. Interactive:   run_processor.bat
echo   2. Command line:  python process_resumes.py -o results.csv
echo.
echo Log file: %LOG_FILE%
echo.
echo [%date% %time%] ==== setup completed OK ====>>"%LOG_FILE%"
echo Press any key to close this window...
pause >nul
popd
endlocal
exit /b 0

REM ============================================================
:die
echo.
echo ============================================================
echo SETUP FAILED
echo ============================================================
echo   !ERR!
echo.
echo [%date% %time%] FAILED: !ERR!>>"%LOG_FILE%"
echo Press any key to close this window...
pause >nul
popd
endlocal
exit /b 1
