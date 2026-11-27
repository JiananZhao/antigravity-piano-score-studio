#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

INSTALL_TOOLS=0
DOWNLOAD_SALAMANDER=0
DOWNLOAD_MUSCRIPTOR=0
CPU_ONLY=0
CUDA_INDEX="cu124"

for arg in "$@"; do
  case "$arg" in
    --install-all)
      INSTALL_TOOLS=1
      DOWNLOAD_SALAMANDER=1
      DOWNLOAD_MUSCRIPTOR=1
      ;;
    --install-tools)
      INSTALL_TOOLS=1
      ;;
    --download-salamander)
      DOWNLOAD_SALAMANDER=1
      ;;
    --download-muscriptor)
      DOWNLOAD_MUSCRIPTOR=1
      ;;
    --cpu-only)
      CPU_ONLY=1
      ;;
    --cuda-index=*)
      CUDA_INDEX="${arg#*=}"
      ;;
    -h|--help)
      echo "Usage: ./install.sh [--install-all] [--install-tools] [--download-salamander] [--download-muscriptor] [--cpu-only] [--cuda-index=cu124]"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: ./install.sh [--install-all] [--install-tools] [--download-salamander] [--download-muscriptor] [--cpu-only] [--cuda-index=cu124]" >&2
      exit 2
      ;;
  esac
done

echo "=== Piano Score Studio Plugin Bootstrap (macOS / Linux) ==="
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PY="$ROOT/.venv/bin/python"
"$VENV_PY" -m pip install --upgrade pip

# Auto-detect GPU and prefer CUDA PyTorch on Linux (macOS uses MPS natively)
HOST_OS="$(uname -s)"
if [ "$CPU_ONLY" -eq 0 ]; then
  if [ "$HOST_OS" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [ -n "$GPU_NAME" ]; then
      echo "NVIDIA GPU detected: $GPU_NAME"
      HAS_CUDA="$("$VENV_PY" -c "import torch; print(int(torch.cuda.is_available()))" 2>/dev/null || echo 0)"
      if [ "$HAS_CUDA" != "1" ]; then
        CUDA_OK=0
        for idx in "$CUDA_INDEX" "cu121"; do
          echo "Installing CUDA PyTorch ($idx) for GPU acceleration..."
          if "$VENV_PY" -m pip install --upgrade "torch>=2.2" --index-url "https://download.pytorch.org/whl/$idx"; then
            CUDA_OK=1
            break
          fi
          echo "Warning: CUDA PyTorch install failed for $idx; trying fallback..."
        done
        if [ "$CUDA_OK" -eq 0 ]; then
          echo "Warning: Could not install CUDA PyTorch automatically; falling back to default PyPI torch."
        fi
      else
        echo "CUDA-enabled PyTorch is already installed in .venv."
      fi
    fi
  elif [ "$HOST_OS" = "Darwin" ]; then
    echo "macOS detected: standard PyTorch wheel supports Apple Silicon MPS GPU acceleration automatically."
  fi
else
  echo "--cpu-only specified; skipping CUDA PyTorch auto-detection."
fi

"$VENV_PY" -m pip install -r requirements.txt

if [ ! -f "config.json" ]; then
  cp config.example.json config.json
  echo "Created config.json from config.example.json"
fi

if [ "$INSTALL_TOOLS" -eq 1 ]; then
  OS_NAME="$(uname -s)"
  if [ "$OS_NAME" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
      if ! command -v mscore >/dev/null 2>&1 && [ ! -d "/Applications/MuseScore 4.app" ]; then
        echo "Installing MuseScore 4 via Homebrew..."
        brew install --cask musescore || echo "Warning: brew install --cask musescore failed; please install from https://musescore.org"
      fi
      if ! command -v sfizz_render >/dev/null 2>&1; then
        echo "Installing sfizz via Homebrew..."
        brew install sfizz || echo "Warning: brew install sfizz failed; please install from https://sfztools.github.io/sfizz/"
      fi
    else
      echo "Warning: Homebrew not found. Install MuseScore 4 (https://musescore.org) and sfizz (https://sfztools.github.io/sfizz/) manually."
    fi
  else
    if ! command -v mscore >/dev/null 2>&1 && ! command -v musescore >/dev/null 2>&1 && ! command -v musescore4 >/dev/null 2>&1; then
      if command -v apt-get >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        echo "Installing musescore via apt-get..."
        sudo apt-get update && sudo apt-get install -y musescore3 || true
      else
        echo "Note: MuseScore 4 not detected. Install via your package manager or AppImage from https://musescore.org"
      fi
    fi
  fi
fi

SETUP_SCRIPT="$ROOT/skills/piano-score-studio/scripts/setup_assets.py"
SETUP_ARGS=(--write-config --check)
if [ "$INSTALL_TOOLS" -eq 1 ]; then SETUP_ARGS+=(--download-sfizz); fi
if [ "$DOWNLOAD_SALAMANDER" -eq 1 ]; then SETUP_ARGS+=(--download-salamander); fi
if [ "$DOWNLOAD_MUSCRIPTOR" -eq 1 ]; then SETUP_ARGS+=(--download-muscriptor); fi

"$VENV_PY" "$SETUP_SCRIPT" "${SETUP_ARGS[@]}"

if [ "$INSTALL_TOOLS" -eq 0 ] && [ "$DOWNLOAD_SALAMANDER" -eq 0 ] && [ "$DOWNLOAD_MUSCRIPTOR" -eq 0 ]; then
  echo ""
  echo "Tip: Run './install.sh --install-all' to auto-install MuseScore/sfizz, Salamander Grand Piano V3 (~714 MB), and MuScriptor (if HF logged in)."
fi