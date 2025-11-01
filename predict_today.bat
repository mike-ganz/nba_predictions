@echo off
REM One-command prediction workflow (scrapes odds + injuries + generates predictions)

echo.
echo ======================================================================
echo NBA PREDICTIONS - ONE COMMAND WORKFLOW
echo ======================================================================
echo.
echo This script will:
echo   1. Scrape latest odds from oddschecker.com
echo   2. Scrape latest injuries from ESPN
echo   3. Prepare game data
echo   4. Generate predictions
echo.

python prepare_todays_games.py --scrape-all

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ======================================================================
    echo SUCCESS! Ready to view predictions
    echo ======================================================================
    echo.
    
    REM Get today's date in YYYY-MM-DD format
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
    set TODAY=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%
    
    echo Predictions file: predictions/predictions_%TODAY%.csv
    echo.
    echo To view predictions:
    echo   python show_predictions.py predictions/predictions_%TODAY%.csv
    echo.
) else (
    echo.
    echo ======================================================================
    echo ERROR: Prediction workflow failed!
    echo ======================================================================
    echo.
    echo Check the error messages above for details.
    echo.
)

pause

