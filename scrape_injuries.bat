@echo off
REM Scrape NBA injury data from ESPN
REM This batch file can be scheduled to run daily

cd /d "%~dp0"
python scrape_injuries.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Injury data updated successfully!
    echo.
) else (
    echo.
    echo ERROR: Failed to scrape injury data
    echo.
    exit /b 1
)

