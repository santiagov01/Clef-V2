#!/usr/bin/env bash
# Usage: ./setup_env.sh
#        ENV_NAME=my_env ./setup_env.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-train_pseudo_env}"
PYTHON="${PYTHON:-python3}"

${PYTHON} -m venv "${ENV_NAME}"
# shellcheck disable=SC1090
source "${ENV_NAME}/bin/activate"

pip install --upgrade pip setuptools wheel

# PyTorch — change the URL for a different CUDA version:
#   CUDA 12.1 -> https://download.pytorch.org/whl/cu121
#   CPU only  -> https://download.pytorch.org/whl/cpu
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

pip install \
    numpy pandas matplotlib scikit-learn joblib tqdm \
    librosa soundfile audioread \
    Pillow optuna psutil colorlog

pip freeze > requirements_binary.txt

echo "Done. Activate with: source ${ENV_NAME}/bin/activate"
