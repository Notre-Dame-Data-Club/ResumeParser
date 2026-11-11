@echo off
REM Resume Scraping Program
REM This script facilitates local use of the SmartResume library.
REM Simplified, translated (Thank you Mandarin minor), and documented by Gaspar
REM Provided by the Data Club of Notre Dame
REM Requires NVIDIA GPU and a Windows Device. Collab alternative is in the notebooks folder.

setlocal enabledelayedexpansion

REM -- CONFIG --
set PYTHON_CMD=python
set DEFAULT_OUTPUT_DIR=resume_results.csv
set DEFAULT_OUTPUT_SINGLE=resume_result.csv
set LOG_FILE=processing.log
set MODELS_SCRIPT=scripts/download_models.py
set CHECK_SCRIPT=scripts/check_requirements.py

REM -- CONSTANTS--
set CHOICE_PROCESS_DIR=1
set CHOICE_SINGLE_FILE=2
set CHOICE_DOWNLOAD=3
set CHOICE_CHECK_REQS=4
set CHOICE_HELP=5
set CHOICE_EXIT=6

call :main
exit /b %errorlevel%

:main

call :log "==================== Session started ===================="

echo.
echo ============================================================
echo English Smart Resume Scraping Program - Windows NVIDIA
echo ============================================================
echo.

REM INIT Python, CUDA and GPU Check
for /f "tokens=1,2,3" %%a in ('%PYTHON_CMD% -c "import sys, torch; cuda=torch.cuda.is_available(); gpu=torch.cuda.get_device_name(0) if cuda else \"CPU\"; print(sys.version.split()[0]); print(\"OK\" if cuda else \"CPU\"); print(gpu)"') do (
    set PYTHON_VER=%%a
    set GPU_AVAILABLE=%%b
    set GPU_STATUS=%%c
)

echo Current System Status:
echo - Python: %PYTHON_VER% [INSTALLED]
echo - GPU Device: %GPU_STATUS%
echo.

call :log "System status: Python=%PYTHON_VER% GPU=%GPU_STATUS%"

REM Check if process_resumes.py exists
if not exist "process_resumes.py" (
    echo.
    echo ERROR: process_resumes.py not found!
    echo Please ensure you are running this script from the correct directory.
    echo.
    pause
    exit /b 1
)

REM Menu display
echo Select operation:
echo %CHOICE_PROCESS_DIR%. Process all resumes in resumes/ directory
echo %CHOICE_SINGLE_FILE%. Process a single file
echo %CHOICE_DOWNLOAD%. Download models
echo %CHOICE_CHECK_REQS%. Check system requirements
echo %CHOICE_HELP%. Show help
echo %CHOICE_EXIT%. Exit
echo.

set /p choice="Enter choice (1-6): "
call :log "Menu choice: %choice%"

if "%choice%"=="%CHOICE_PROCESS_DIR%" (
    call :process_directory
    exit /b !errorlevel!
)
if "%choice%"=="%CHOICE_SINGLE_FILE%" (
    call :process_single
    exit /b !errorlevel!
)
if "%choice%"=="%CHOICE_DOWNLOAD%" (
    call :download_models
    exit /b !errorlevel!
)
if "%choice%"=="%CHOICE_CHECK_REQS%" (
    call :check_requirements
    exit /b !errorlevel!
)
if "%choice%"=="%CHOICE_HELP%" (
    call :show_help
    exit /b !errorlevel!
)
if "%choice%"=="%CHOICE_EXIT%" (
    call :log "Session ended: user exited"
    exit /b 0
)

call :log "Invalid menu choice: %choice%"
echo ERROR: Invalid choice. Enter 1-6.
pause
exit /b 1

REM -- 1. Process Directory --
:process_directory
echo.
echo Processing all PDFs in resumes/ directory...
echo Output will be saved to %DEFAULT_OUTPUT_DIR%
echo.

REM Output file name
set /p output_file="Enter output CSV filename [%DEFAULT_OUTPUT_DIR%]: "
if "!output_file!"=="" set output_file=%DEFAULT_OUTPUT_DIR%
set "OUTPUT_READY="
set "PREV_MTIME="

REM Optimization options
set EXTRA_FLAGS=

echo.
echo Performance Optimization (optional):
echo 1. Default mode - balanced speed/quality
echo 2. Fast mode - skip OCR for text-only PDFs
echo 3. Simpler mode - skip layout detection for simple resume formats.
echo 4. Max mode - force OCR on all pages
echo.
set /p perf_choice="Select mode (1-4) [1]: "
if "!perf_choice!"=="" set perf_choice=1

if "!perf_choice!"=="2" (
    set EXTRA_FLAGS=--pdf-raw-text
    echo [INFO] Fast mode enabled - OCR will be skipped
)
if "!perf_choice!"=="3" (
    set EXTRA_FLAGS=--no-layout-detection
    echo [INFO] Simpler mode - simple, single page resume optimized
)
if "!perf_choice!"=="4" (
    set EXTRA_FLAGS=--force-ocr
    echo [INFO] Better extraction enabled - OCR on all pages
)
set /p gpu_mem="Enter GPU memory limit in GB (leave blank for auto): "
set /p page_limit="Limit pages per resume (leave blank for no limit): "

REM Final build command
set CMD=%PYTHON_CMD% -u process_resumes.py -o "!output_file!" !EXTRA_FLAGS!
if not "!gpu_mem!"=="" (
    set CMD=!CMD! --gpu-memory !gpu_mem!
)
if not "!page_limit!"=="" (
    set CMD=!CMD! --page-limit !page_limit!
)

REM -- Runtime Logging -- 
echo.
echo ============================================================
echo Starting processing... (watch real-time progress below)
echo ============================================================
echo Progress will display as: [X/Y] percentage ^| speed ^| Last time ^| ETA
echo.


if exist "!output_file!" (
    for %%F in ("!output_file!") do set "PREV_MTIME=%%~tF"
)
!CMD!
set RC=!errorlevel!

if !RC! NEQ 0 (
    echo.
    echo ============================================================
    echo ERROR: Processing failed
    echo ============================================================
    echo.
    echo Check log file for details: %LOG_FILE%
    echo.
    pause
) else (
    REM Wait loop to compensate .csv creation delay
    for /l %%I in (1,1,5) do (
        if not defined OUTPUT_READY (
            if exist "!output_file!" (
                for %%F in ("!output_file!") do set "NEW_MTIME=%%~tF"
                if not defined PREV_MTIME (
                    set "OUTPUT_READY=1"
                ) else if /i not "!NEW_MTIME!"=="!PREV_MTIME!" (
                    set "OUTPUT_READY=1"
                )
            )
        )
        if not defined OUTPUT_READY timeout /t 1 /nobreak >nul 2>&1
    )

    if defined OUTPUT_READY (
        echo.
        echo ============================================================
        echo SUCCESS: Processing complete!
        echo ============================================================
        echo.
        echo Results saved to: !output_file!
        echo Log file: %LOG_FILE%
        echo.
        set /p view_results="View results in Excel now? (y/n): "
        if /i "!view_results!"=="y" (
            start excel "!output_file!"
        )
    ) else (
        echo.
        echo ============================================================
        echo WAITING FOR OUTPUT
        echo ============================================================
        echo.
        echo The processor exited without a newly written CSV for this run.
        echo No Excel prompt will be shown until a fresh output file exists.
        echo.
        echo Check log file for details: %LOG_FILE%
    )
)
pause
exit /b !RC!

REM -- 2. Process File --
:process_single
echo.
echo Processing a single resume file...
echo.

REM Single File Path
set /p input_file="Enter path to resume PDF: "
if "!input_file!"=="" (
    echo ERROR: No file path entered.
    pause
    exit /b 1
)
if not exist "!input_file!" (
    echo ERROR: File not found: !input_file!
    pause
    exit /b 1
)

REM Output file name
set /p output_file="Enter output CSV filename [%DEFAULT_OUTPUT_SINGLE%]: "
if "!output_file!"=="" set output_file=%DEFAULT_OUTPUT_SINGLE%
set "OUTPUT_READY="
set "PREV_MTIME="

REM Optimization options
set EXTRA_FLAGS=


echo.
echo Performance Optimization (optional):
echo 1. Default mode - balanced speed/quality
echo 2. Fast mode - skip OCR for text-only PDFs
echo 3. Simpler mode - skip layout detection for simple resume formats.
echo 4. Max mode - force OCR on all pages
echo.
set /p perf_choice="Select mode (1-4) [1]: "
if "!perf_choice!"=="" set perf_choice=1

if "!perf_choice!"=="2" (
    set EXTRA_FLAGS=--pdf-raw-text
    echo [INFO] Fast mode enabled - OCR will be skipped
)
if "!perf_choice!"=="3" (
    set EXTRA_FLAGS=--no-layout-detection
    echo [INFO] Simpler mode - simple, single page resume optimized
)
if "!perf_choice!"=="4" (
    set EXTRA_FLAGS=--force-ocr
    echo [INFO] Better extraction enabled - OCR on all pages
)

set /p gpu_mem="Enter GPU memory limit in GB (leave blank for auto): "
set /p page_limit="Limit pages for this resume (leave blank for no limit): "

REM Final build command
set CMD=%PYTHON_CMD% -u process_resumes.py -f "!input_file!" -o "!output_file!" !EXTRA_FLAGS!
if not "!gpu_mem!"=="" (
    set CMD=!CMD! --gpu-memory !gpu_mem!
)
if not "!page_limit!"=="" (
    set CMD=!CMD! --page-limit !page_limit!
)

REM -- Runtime Logging --
echo.
echo ============================================================
echo Starting processing... (watch real-time progress below)
echo ============================================================
echo.

if exist "!output_file!" (
    for %%F in ("!output_file!") do set "PREV_MTIME=%%~tF"
)
!CMD!
set RC=!errorlevel!

if !RC! NEQ 0 (
    echo.
    echo ============================================================
    echo ERROR: Processing failed
    echo ============================================================
    echo.
    echo Check log file for details: %LOG_FILE%
    echo.
    pause
) else (
     REM Wait loop to compensate .csv creation delay
    for /l %%I in (1,1,5) do (
        if not defined OUTPUT_READY (
            if exist "!output_file!" (
                for %%F in ("!output_file!") do set "NEW_MTIME=%%~tF"
                if not defined PREV_MTIME (
                    set "OUTPUT_READY=1"
                ) else if /i not "!NEW_MTIME!"=="!PREV_MTIME!" (
                    set "OUTPUT_READY=1"
                )
            )
        )
        if not defined OUTPUT_READY timeout /t 1 /nobreak >nul 2>&1
    )

    if defined OUTPUT_READY (
        echo.
        echo ============================================================
        echo SUCCESS: Processing complete!
        echo ============================================================
        echo.
        echo Results saved to: !output_file!
        echo Log file: %LOG_FILE%
        echo.
        set /p view_results="View results in Excel now? (y/n): "
        if /i "!view_results!"=="y" (
            start excel "!output_file!"
        )
    ) else (
        echo.
        echo ============================================================
        echo WAITING FOR OUTPUT
        echo ============================================================
        echo.
        echo The processor exited without a newly written CSV for this run.
        echo No Excel prompt will be shown until a fresh output file exists.
        echo.
        echo Check log file for details: %LOG_FILE%
    )
)
pause
exit /b !RC!

REM -- 3. Download Models --
:download_models
echo.
echo Downloading models from Hugging Face...
echo This may take several minutes depending on internet speed.
echo.

call :log "download_models: starting (script=%MODELS_SCRIPT%)"

%PYTHON_CMD% "%MODELS_SCRIPT%" --model_type llm --source huggingface
set RC=!errorlevel!

if !RC! NEQ 0 (
    call :log "download_models: FAILED with exit code !RC!"
    echo.
    echo ERROR: Model download failed!
    echo Check internet connection and try again.
) else (
    call :log "download_models: SUCCESS"
    echo.
    echo SUCCESS: Models downloaded successfully!
)
pause
exit /b !RC!

REM -- 3. Check Requirements --
:check_requirements
echo.
echo Checking system requirements...
echo.

call :log "check_requirements: starting"

if not exist "%CHECK_SCRIPT%" (
    call :log "check_requirements: FAILED - %CHECK_SCRIPT% not found"
    echo ERROR: %CHECK_SCRIPT% not found.
    echo Please ensure you are running this script from the repository root.
    pause
    exit /b 1
)

%PYTHON_CMD% "%CHECK_SCRIPT%"
set RC=!errorlevel!

if !RC! NEQ 0 (
    call :log "check_requirements: completed with missing packages (exit !RC!)"
) else (
    call :log "check_requirements: SUCCESS - all packages installed"
)

echo.
pause
exit /b !RC!

REM -- 5. FAQ --
:show_help
call :log "show_help: displayed"
cls
echo.
echo ============================================================
echo Help - Windows Resume Processing Script
echo ============================================================
echo.
echo REAL-TIME PROGRESS DISPLAY:
echo.
echo   During processing, you will see live progress updates:
echo   [5/15] 33.3%% ^| 2.45 files/sec ^| Last: 2.3s ^| ETA: 4m12s
echo.
echo   This shows: [processed/total] percentage | speed | time per file | estimated time remaining
echo.
echo COMMANDS:
echo.
echo Process directory:
echo   %PYTHON_CMD% process_resumes.py -d resumes/ -o results.csv
echo.
echo Process a single file:
echo   %PYTHON_CMD% process_resumes.py -f resumes\john_doe.pdf -o result.csv
echo.
echo Limit GPU memory:
echo   %PYTHON_CMD% process_resumes.py -d resumes/ --gpu-memory 8
echo.
echo PERFORMANCE TUNING (select during processing):
echo.
echo   Mode 1: Default — Balanced speed and quality
echo   Mode 2: Fast — Skip OCR for text-only PDFs (2-3x faster)
echo   Mode 3: Simpler — Skip layout detection (simpler PDFs only)
echo   Mode 4: Max — Force OCR on all pages (quality over speed)
echo.
echo   All modes batch the LLM prompts (--batch-size, default 8; several resumes share calls).
echo   Use --batch-size 1 if you hit GPU out-of-memory errors.
echo.
echo Advanced command-line:
echo   %PYTHON_CMD% process_resumes.py -d resumes/ --pdf-raw-text --page-limit 5 --gpu-memory 8
echo.
echo Save JSON details:
echo   %PYTHON_CMD% process_resumes.py -d resumes/ --save-json
echo.
echo VIEW LOGS:
echo   - %LOG_FILE% (text editor)
echo   - %DEFAULT_OUTPUT_DIR% (Excel)
echo.
echo TROUBLESHOOTING:
echo.
echo Q: How do I know if my GPU is being used?
echo A: Run 'nvidia-smi' in another terminal while processing
echo.
echo Q: Getting "out of memory" error?
echo A: Use --gpu-memory option or --page-limit to reduce memory usage
echo.
echo Q: No results extracted?
echo A: Check %LOG_FILE%, may need --force-ocr for scanned PDFs
echo.
echo Q: Very slow processing?
echo A: Use --pdf-raw-text for text-only PDFs or --no-layout-detection
echo.
echo For more help, see WINDOWS_SETUP.md
echo.
pause
exit /b 0

REM ========== LOG HELPER ==========
REM BAT-level logging is intentionally disabled so processing.log only
REM contains Python parsing logs and not wrapper/menu noise.
:log
goto :eof
