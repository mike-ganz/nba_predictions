@echo off
REM Quick Error Check Batch Script for Windows
REM Run this after your simulations to check for issues

echo ========================================
echo NBA Simulation Error Check
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Make sure Python is installed and in your PATH.
    pause
    exit /b 1
)

REM Check if script exists
if not exist "quick_error_check.py" (
    echo ERROR: quick_error_check.py not found in current directory!
    pause
    exit /b 1
)

REM Run the quick error check
python quick_error_check.py

echo.
echo ========================================
echo Check complete! 
echo.
echo For detailed analysis, run:
echo   python error_analysis.py
echo ========================================
echo.
pause
