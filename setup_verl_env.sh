#!/bin/bash
# VeRL Environment Setup Script
# This script helps reproduce the verl conda environment

set -e

echo "=========================================="
echo "VeRL Environment Setup"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo "❌ Error: environment.yml not found in current directory"
    echo "Please run this script from the arm2 directory"
    exit 1
fi

# Check if verl directory exists
if [ ! -d "verl" ]; then
    echo "❌ Error: verl directory not found"
    exit 1
fi

echo ""
echo "Step 1: Creating conda environment from environment.yml..."
if conda env list | grep -q "^verl "; then
    echo "⚠️  Environment 'verl' already exists"
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n verl -y
        conda env create -f environment.yml
    else
        echo "Skipping environment creation. Activating existing environment..."
    fi
else
    conda env create -f environment.yml
fi

echo ""
echo "Step 2: Activating environment and installing verl..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

echo ""
echo "Step 3: Installing verl package..."
cd verl
pip install -e .

echo ""
echo "Step 3.5: Installing flash-attn (may take a while)..."
pip install flash-attn==2.7.4.post1 --no-build-isolation || {
    echo "⚠️  flash-attn installation failed, trying alternative method..."
    pip install flash-attn --no-build-isolation
}

echo ""
echo "Step 4: Verifying installation..."
python -c "import verl; print('✅ verl imported successfully')" || echo "⚠️  verl import failed"
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || echo "⚠️  PyTorch not found"
python -c "import vllm; print(f'✅ vLLM {vllm.__version__}')" || echo "⚠️  vLLM not found"
python -c "import ray; print(f'✅ Ray {ray.__version__}')" || echo "⚠️  Ray not found"

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate verl"
echo ""
