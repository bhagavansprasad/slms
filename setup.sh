#!/bin/bash
# Quick setup script for SLM project
# Run with: bash setup.sh

echo "=================================================="
echo "🚀 SLM Project Setup"
echo "=================================================="

# Check Python version
echo ""
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1)
if [ $? -eq 0 ]; then
    echo "✅ Found: $python_version"
else
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment (optional but recommended)
echo ""
read -p "🤔 Create virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
    
    echo "✅ Virtual environment created and activated"
    echo "   To activate later, run: source venv/bin/activate"
else
    echo "⏭️  Skipping virtual environment creation"
fi

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Create necessary directories
echo ""
echo "📁 Creating project directories..."
mkdir -p models
mkdir -p outputs
mkdir -p data
mkdir -p notebooks

echo "✅ Directories created:"
echo "   - models/    (for saved model checkpoints)"
echo "   - outputs/   (for training curves and generated text)"
echo "   - data/      (for custom training data)"
echo "   - notebooks/ (for Jupyter notebooks)"

# Check if PyTorch can use GPU
echo ""
echo "🔍 Checking for GPU support..."
python3 -c "import torch; print('✅ GPU available:', torch.cuda.is_available()); print('   Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

# Summary
echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "📝 Next steps:"
echo "   1. Review configurations in config_cpu.py"
echo "   2. Run your first training:"
echo "      python train.py"
echo ""
echo "📚 Documentation:"
echo "   - README.md        (main guide)"
echo "   - README_CPU.md    (CPU training guide)"
echo ""
echo "🎓 For teaching demos:"
echo "   - Use CONFIG_TINY_CPU (10-20 min training)"
echo "   - Check README_CPU.md for timing estimates"
echo ""
echo "=================================================="