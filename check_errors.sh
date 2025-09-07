#!/bin/bash
# Quick Error Check Shell Script for Linux/Mac
# Run this after your simulations to check for issues

echo "========================================"
echo "NBA Simulation Error Check"
echo "========================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python not found! Make sure Python is installed and in your PATH."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check if script exists
if [ ! -f "quick_error_check.py" ]; then
    echo "ERROR: quick_error_check.py not found in current directory!"
    exit 1
fi

# Run the quick error check
$PYTHON_CMD quick_error_check.py

echo
echo "========================================"
echo "Check complete!"
echo
echo "For detailed analysis, run:"
echo "  $PYTHON_CMD error_analysis.py"
echo "========================================"
echo
