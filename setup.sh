#!/usr/bin/env bash
#
# Setup script for visionary
# Creates a virtual environment, installs dependencies, and optionally enables GPU support.

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$1"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$1"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$1"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$1"; }

# -------------------------------------------------------------------
# 1. Detect OS
# -------------------------------------------------------------------
OS="$(uname -s)"
case "$OS" in
    Linux)  info "Detected OS: Linux" ;;
    Darwin) info "Detected OS: macOS" ;;
    *)      error "Unsupported OS: $OS"; exit 1 ;;
esac

# -------------------------------------------------------------------
# 2. Install uv if not present
# -------------------------------------------------------------------
if command -v uv &>/dev/null; then
    ok "uv is already installed ($(uv --version))"
else
    info "Installing uv..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        error "Failed to install uv"
        exit 1
    fi
    # Source the env so uv is available in this session
    export PATH="$HOME/.local/bin:$PATH"
    ok "uv installed ($(uv --version))"
fi

# -------------------------------------------------------------------
# 3. Sync dependencies (creates .venv and installs from pyproject.toml)
# -------------------------------------------------------------------
info "Running uv sync..."
if ! uv sync; then
    error "uv sync failed"
    exit 1
fi
ok "Dependencies installed"

info "Installing visionary package..."
if ! uv pip install --python .venv/bin/python .; then
    error "uv pip install . failed"
    exit 1
fi
ok "visionary package installed"

# -------------------------------------------------------------------
# 4. Detect NVIDIA GPU and install JAX CUDA support
# -------------------------------------------------------------------
GPU_INSTALLED=false

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
    if [ -n "$GPU_NAME" ]; then
        info "NVIDIA GPU detected: $GPU_NAME"
        info "Installing jax[cuda12] for GPU acceleration..."
        if uv pip install --python .venv/bin/python 'jax[cuda12]'; then
            ok "JAX CUDA 12 support installed"
            GPU_INSTALLED=true
        else
            warn "Failed to install jax[cuda12] — falling back to CPU-only JAX"
        fi
    fi
else
    info "No NVIDIA GPU detected — using CPU-only JAX"
fi

# -------------------------------------------------------------------
# 5. Summary
# -------------------------------------------------------------------
echo ""
echo "==========================================="
echo "  visionary setup complete"
echo "==========================================="
echo "  Python:  $(uv run python --version)"
echo "  venv:    .venv/"
if $GPU_INSTALLED; then
    echo "  JAX:     GPU (CUDA 12) — $GPU_NAME"
else
    echo "  JAX:     CPU only"
fi
echo ""
echo "  Run:     uv run python -m visionary ..."
echo "==========================================="
