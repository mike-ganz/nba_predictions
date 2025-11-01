@echo off
REM Scrape NBA odds using Selenium (fully automated)

echo.
echo ======================================================================
echo NBA ODDS SCRAPER (Selenium - Fully Automated)
echo ======================================================================
echo.

python scrape_odds_selenium.py --output data/market/current_spreads.json

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo SUCCESS! Odds saved to data/market/current_spreads.json
    echo ======================================================================
    echo.
    echo Next steps:
    echo   1. Scrape injuries: .\scrape_injuries.bat
    echo   2. Generate predictions: python prepare_todays_games.py
    echo.
) else (
    echo.
    echo ======================================================================
    echo ERROR: Odds scraping failed!
    echo ======================================================================
    echo.
    echo Troubleshooting:
    echo   1. Make sure Chrome is installed
    echo   2. Check your internet connection
    echo   3. Try running with browser visible: python scrape_odds_selenium.py --show-browser
    echo.
)

pause

