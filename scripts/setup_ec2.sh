#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-scripts/ec2_env.sh}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  echo "Missing env file: $ENV_FILE"
  exit 1
fi

if command -v dnf >/dev/null 2>&1; then
  PKG=dnf
else
  PKG=yum
fi

sudo "$PKG" -y update
sudo "$PKG" -y install git curl gcc gcc-c++ make cmake unzip tar gzip

# Install Lustre client
if [[ "$PKG" == "dnf" ]]; then
  sudo "$PKG" -y install lustre-client
else
  sudo amazon-linux-extras install -y lustre2
fi

# Miniforge
if [[ ! -d "$HOME/miniforge3" ]]; then
  curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
fi

# Conda env
source "$HOME/miniforge3/bin/activate"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-detanet}"
if ! conda env list | grep -q "^${CONDA_ENV_NAME}\\s"; then
  mamba create -n "$CONDA_ENV_NAME" python=3.10 -y
fi
mamba activate "$CONDA_ENV_NAME"

# Core deps
mamba install -y -c conda-forge rdkit ase h5py
pip install --upgrade pip
pip install torch==2.2.2+cpu -f https://download.pytorch.org/whl/cpu
pip install torch_geometric -f https://data.pyg.org/whl/torch-2.2.2+cpu.html
pip install mace-torch==0.3.14 nequip==0.6.2 e3nn==0.4.4 tqdm>=4.66

# DeepMD (force rebuild if wheel ABI mismatches)
DP_ENABLE_PYTORCH=1 pip install --no-binary deepmd-kit deepmd-kit==3.1.2

echo "Setup complete. Activate with: source $HOME/miniforge3/bin/activate && mamba activate $CONDA_ENV_NAME"
