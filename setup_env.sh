#!/bin/bash

# Configuration
VENV_DIR=".venv"

echo "Creating Python virtual environment in $VENV_DIR..."

# Check if python3 or python is available
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python is not installed or not in your PATH."
    exit 1
fi

# Create virtual environment
$PYTHON_CMD -m venv $VENV_DIR

# Check if creation was successful
if [ ! -d "$VENV_DIR" ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

echo "Virtual environment created."
echo "Activating virtual environment..."

# Activate the virtual environment
# Handle different OS platforms (Windows via Git Bash vs Linux/macOS)
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Could not find activation script. Exiting."
    exit 1
fi

echo "Virtual environment activated."
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing core dependencies (PyTorch)..."
# Using standard PyTorch index. Depending on your GPU/CUDA, you might need a specific index URL
pip install torch torchvision torchaudio

echo "Installing other dependencies (numpy, YOLO, OpenCV, etc.)..."
pip install numpy pandas matplotlib opencv-python ultralytics tqdm psutil

# If requirements.txt exists, install from it as well
if [ -f "requirements.txt" ]; then
    echo "Found requirements.txt. Installing additional requirements..."
    pip install -r requirements.txt
fi

echo "================================================="
echo "Environment setup complete!"
echo "To activate the environment later, run:"
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    echo "    source $VENV_DIR/Scripts/activate"
else
    echo "    source $VENV_DIR/bin/activate"
fi
echo "================================================="
