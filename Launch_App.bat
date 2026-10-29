@echo off
title AudioBench - Web Studio Dashboard
cd /d "%~dp0"

echo ========================================================
echo   AudioBench: Offline DSP Regression Benchmark & Studio
echo ========================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo Using local virtual environment .venv...
    ".venv\Scripts\python.exe" -m audiobench ui
) else (
    echo Using system Python...
    python -m audiobench ui
)

if errorlevel 1 (
    echo.
    echo Launch failed. Make sure dependencies are installed:
    echo   pip install -e ".[dev,ui]"
    pause
)
