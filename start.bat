@echo off
echo ========================================
echo Parking Lot Monitor - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        echo Make sure Python 3.10+ is installed
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
if not exist "venv\Lib\site-packages\ultralytics\" (
    echo Installing dependencies...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check for YOLO model
if not exist "yolo11n.pt" (
    echo Downloading YOLO model...
    python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
)

echo.
echo ========================================
echo Starting Parking Lot Monitor...
echo ========================================
echo.

REM Run the application
python parking_lot_monitor.py

REM Keep window open on error
if errorlevel 1 (
    echo.
    echo ========================================
    echo Application exited with error
    echo ========================================
    pause
)
