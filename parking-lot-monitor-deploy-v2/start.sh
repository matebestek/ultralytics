#!/bin/bash

echo "========================================"
echo "Parking Lot Monitor - Quick Start"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        echo "Make sure Python 3.10+ is installed"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
if [ ! -d "venv/lib/python*/site-packages/ultralytics" ]; then
    echo "Installing dependencies..."
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Check for YOLO model
if [ ! -f "yolo11n.pt" ]; then
    echo "Downloading YOLO model..."
    python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
fi

echo ""
echo "========================================"
echo "Starting Parking Lot Monitor..."
echo "========================================"
echo ""

# Run the application
python parking_lot_monitor.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "========================================"
    echo "Application exited with error"
    echo "========================================"
    read -p "Press Enter to continue..."
fi
