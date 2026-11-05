#!/bin/bash
echo "============================================================"
echo " IBM AICTE — Interview Trainer Agent"
echo " Setup Script (Linux / macOS)"
echo "============================================================"

echo ""
echo "[1/4] Checking Python version..."
python3 --version || { echo "ERROR: Python 3 not found."; exit 1; }

echo ""
echo "[2/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo ""
echo "[3/4] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "[4/4] Setting up .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env created. Fill in your IBM Cloud credentials!"
else
    echo ".env already exists."
fi

echo ""
echo "============================================================"
echo " Setup complete!"
echo " 1. Edit .env with your IBM watsonx.ai credentials"
echo " 2. Run: bash start.sh"
echo " 3. Open: http://localhost:8000"
echo "============================================================"
