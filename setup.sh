#!/bin/bash

# Check if a mode argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {conda|uv}"
    exit 1
fi

MODE=$1

# Function to set up using conda
conda_setup() {
    echo "Setting up using conda..."
    conda remove --name rppg-toolbox --all -y || exit 1
    conda create -n rppg-toolbox python=3.8 -y || exit 1
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate rppg-toolbox || exit 1
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt || exit 1
    cd tools/mamba || exit 1
    python setup.py install || exit 1
}

# Function to set up using uv
uv_setup() {
    rm -rf .venv || exit 1
    uv venv --python 3.8 || exit 1
    source .venv/bin/activate || exit 1
    uv pip install setuptools wheel || exit 1
    uv pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121 || exit 1
    uv pip install -r requirements.txt || exit 1
    cd tools/mamba && python setup.py install || exit 1
    # Explicitly install PyQt5 to use interactive plotting and avoid non-interactive backends
    # See this relevant issue for more details: https://github.com/astral-sh/uv/issues/6893
    uv pip install PyQt5
}

# Execute the appropriate setup based on the mode
case $MODE in
    conda)
        conda_setup
        ;;
    uv)
        uv_setup
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 {conda|uv}"
        exit 1
        ;;
esac
