#!/usr/bin/env bash
# Usage: ./train_binary_species.sh
#        ENV_NAME=my_env ./train_binary_species.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-train_pseudo_env}"

# Activate the environment
# shellcheck disable=SC1090
source "${ENV_NAME}/bin/activate"

# Run the training script

python train_binary_species.py \
    --data_dir data/labeled \
    --output outputs_binary \
    --architectures efficientnet_b0 efficientnet_b3 regnet_y_400mf \
    --trials 20 --epochs_min 5 --epochs_max 30

# Train only specific species:
#   python train_binary_species.py \
#       --data_dir data/labeled \
#       --output outputs_binary \
#       --species_filter species_A species_B \
#       --trials 15