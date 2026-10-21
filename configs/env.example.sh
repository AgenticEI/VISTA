# Copy to configs/env.local.sh and supply paths on your own machine.
# This file is sourced by configs/env.sh. Never commit env.local.sh.

# Host Python: install the core repository in editable mode here.
# export RPENT_PYTHON="/path/to/host-env/bin/python"
# export CODEX_BIN="/path/to/codex"
# Simulator Python: install the selected simulator extra in this environment.
export RPENT_ENV_PYTHON="/path/to/simulator-env/bin/python"
# VLA and SAM3 may require separate environments.
export RPENT_VLA_PYTHON="/path/to/vla-env/bin/python"
export RPENT_SAM3_PYTHON="/path/to/sam3-env/bin/python"

# Set credentials in your shell or this ignored local file.
# export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"

# LIBERO-Pro / Pi0.5.
export LIBERO_TYPE=pro
export PI05_CHECKPOINT_PATH="/path/to/pi05-checkpoint"
# export LIBEROPRO_CONFIG_PATH="/path/to/liberopro/config"

# SAM3 perception service for LIBERO.
export SAM3_CHECKPOINT_PATH="/path/to/sam3.pt"
export SAM3_BPE_PATH="/path/to/bpe_simple_vocab_16e6.txt.gz"
