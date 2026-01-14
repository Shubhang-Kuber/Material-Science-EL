#!/bin/bash

echo "========================================"
echo "Steel Plate Fault Detector - Setup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed"
    echo "Please install Python 3.9+ first"
    exit 1
fi

echo "[1/4] Creating Python virtual environment..."
cd backend
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to create virtual environment"
    exit 1
fi

echo "[2/4] Activating virtual environment..."
source venv/bin/activate

echo "[3/4] Installing Python dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install Python dependencies"
    exit 1
fi

echo "[4/4] Creating sample model (untrained)..."
python train_model.py

echo ""
echo "========================================"
echo "Backend setup complete!"
echo "========================================"
echo ""
echo "To start the backend server:"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "========================================"
echo "Now setting up Frontend..."
echo "========================================"
echo ""

cd ../frontend

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "[WARNING] Node.js is not installed"
    echo "Please install Node.js 16+ from https://nodejs.org/"
    echo "Then run: cd frontend && npm install && npm start"
    exit 0
fi

echo "Installing Node.js dependencies..."
npm install
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install Node.js dependencies"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To run the application:"
echo ""
echo "1. Start the Backend (Terminal 1):"
echo "   cd steel-fault-detector/backend"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "2. Start the Frontend (Terminal 2):"
echo "   cd steel-fault-detector/frontend"
echo "   npm start"
echo ""
echo "The app will open at http://localhost:3000"
echo "========================================"
