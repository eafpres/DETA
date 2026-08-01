#!/usr/bin/env bash
set -euo pipefail
#
# User configuration
#
REPO_URL="https://github.com/eafpres/DETA.git"
BASE_DIR="$HOME"
REPO_DIR="$HOME/DETA"
DETA_DIR="$REPO_DIR"
VENV_DIR="$HOME/venvs/deta_infer_jetson"
TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
TORCHVISION_TAG="v0.20.0"
#
# Normalize this script if copied from Windows
#
sed -i 's/\r$//' "$0"
#
# Install system packages
#
sudo apt update
sudo apt install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  wget \
  pkg-config \
  python3-dev \
  python3-pip \
  python3-venv \
  libopenblas-dev \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libffi-dev \
  liblzma-dev \
  tk-dev \
  libjpeg-dev \
  libpng-dev \
  ninja-build \
  libgl1 \
  libglib2.0-0 \
  libgtk2.0-0 \
  libgtk-3-0 \
  libsm6 \
  libxext6 \
  libxrender1 \
  libfontconfig1
#
# Verify required Python version
#
PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "$PYTHON_VERSION" != "3.10" ]; then
  echo "expected Python 3.10, found Python $PYTHON_VERSION"
  exit 1
fi
#
# Create clean virtual environment
#
deactivate 2>/dev/null || true
mkdir -p "$(dirname "$VENV_DIR")"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
python -m pip install --force-reinstall "setuptools<81" wheel packaging
python -m pip install "numpy==1.26.1"
#
# Install cuSPARSELt required by NVIDIA PyTorch 24.06+
#
CUSPARSELT_VERSION="0.5.2.1"
CUSPARSELT_ARCHIVE="libcusparse_lt-linux-sbsa-${CUSPARSELT_VERSION}-archive"
CUSPARSELT_TMP="$BASE_DIR/tmp_cusparselt"
rm -rf "$CUSPARSELT_TMP"
mkdir -p "$CUSPARSELT_TMP"
cd "$CUSPARSELT_TMP"
wget \
  "https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/${CUSPARSELT_ARCHIVE}.tar.xz"
tar -xf "${CUSPARSELT_ARCHIVE}.tar.xz"
sudo cp -a \
  "$CUSPARSELT_ARCHIVE/include/." \
  /usr/local/cuda/include/
sudo cp -a \
  "$CUSPARSELT_ARCHIVE/lib/." \
  /usr/local/cuda/lib64/
sudo ldconfig
cd "$BASE_DIR"
rm -rf "$CUSPARSELT_TMP"
#
# Install NVIDIA Jetson PyTorch wheel
#
python -m pip uninstall -y torch torchvision torchaudio || true
python -m pip cache purge || true
python -m pip install --no-cache-dir "$TORCH_WHEEL"
#
# Install runtime Python dependencies except torchvision
#
python -m pip install --no-deps opencv-python
python -m pip install \
  Pillow==9.5.0 \
  matplotlib \
  scipy \
  tqdm \
  cython \
  pycocotools \
  pandas \
  pyyaml \
  requests \
  einops \
  huggingface_hub \
  safetensors
python -m pip install --no-deps "timm==1.0.28"
#
# Build torchvision from source for Jetson/aarch64
#
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
cd "$BASE_DIR"
rm -rf vision
git clone --branch "$TORCHVISION_TAG" --depth 1 https://github.com/pytorch/vision.git
cd "$BASE_DIR/vision"
export BUILD_VERSION="${TORCHVISION_TAG#v}"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS=2
python -m pip install -v . --no-build-isolation --no-deps
cd "$BASE_DIR"
rm -rf "$BASE_DIR/vision"
python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
if not torch.__version__.startswith("2.5.0a0"):
  raise SystemExit("wrong torch installed")
if not torchvision.__version__.startswith("0.20.0"):
  raise SystemExit("wrong torchvision installed")
PY
#
# Patch torchvision SSDlite to match the trained SSD checkpoint architecture
#
python - <<'PY'
from pathlib import Path
import torchvision.models.detection.ssdlite as ssdlite
path = Path(ssdlite.__file__)
text = path.read_text()
old = "reduce_tail = weights_backbone is None"
new = "reduce_tail = False"
if old in text:
  backup = path.with_suffix(path.suffix + ".bak")
  backup.write_text(text)
  path.write_text(text.replace(old, new, 1))
  print(f"patched torchvision ssdlite: {path}")
  print(f"backup written: {backup}")
elif new in text:
  print(f"torchvision ssdlite already patched: {path}")
else:
  raise SystemExit(f"expected reduce_tail assignment not found in {path}")
PY
#
# Clone or update DETA fork
#
cd "$BASE_DIR"
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" pull --ff-only
elif [ -e "$REPO_DIR" ]; then
  if [ ! -d "$REPO_DIR" ]; then
    echo "repository path exists but is not a directory: $REPO_DIR"
    exit 1
  fi
  REPO_STAGE="$(mktemp -d "$BASE_DIR/.deta_clone.XXXXXX")"
  cleanup_repo_stage() {
    if [ -n "${REPO_STAGE:-}" ] && [ -d "$REPO_STAGE" ]; then
      rm -rf "$REPO_STAGE"
    fi
  }
  trap cleanup_repo_stage EXIT
  git clone "$REPO_URL" "$REPO_STAGE/repo"
  cp -a -n "$REPO_STAGE/repo/." "$REPO_DIR/"
  rm -rf "$REPO_STAGE"
  REPO_STAGE=""
  echo "adopted existing directory as DETA repository: $REPO_DIR"
  echo "existing files were preserved"
else
  git clone "$REPO_URL" "$REPO_DIR"
fi
if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "failed to initialize DETA repository: $REPO_DIR"
  exit 1
fi
if [ ! -d "$DETA_DIR/models/ops" ]; then
  echo "DETA models/ops directory not found: $DETA_DIR/models/ops"
  exit 1
fi
if [ ! -f "$DETA_DIR/tools/infer_folder_remote_Jetson.py" ]; then
  echo "DETA inference script not found:"
  echo "  $DETA_DIR/tools/infer_folder_remote_Jetson.py"
  exit 1
fi
#
# Compile DETA CUDA extension
#
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cd "$DETA_DIR/models/ops"
bash make.sh
#
# Create launcher
#
cat > "$HOME/run_deta_infer.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PATH="/usr/local/cuda/bin:\$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}"
source "$VENV_DIR/bin/activate"
cd "$DETA_DIR"
python tools/infer_folder_remote_Jetson.py "\$@"
EOF
chmod +x "$HOME/run_deta_infer.sh"
#
# Verify installation
#
cd "$DETA_DIR"
python - <<'PY'
import platform
import torch
import torchvision
import MultiScaleDeformableAttention
print("machine:", platform.machine())
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
  raise SystemExit("PyTorch cannot access CUDA")
print("CUDA device:", torch.cuda.get_device_name(0))
x = torch.ones((1024, 1024), device="cuda")
y = x @ x
torch.cuda.synchronize()
print("CUDA tensor test:", y.device)
print("DETA op import OK")
PY
#
# Done
#
echo "installed DETA Jetson environment:"
echo "  venv: $VENV_DIR"
echo "  repo: $REPO_DIR"
echo "  DETA working directory: $DETA_DIR"
echo "  launcher: $HOME/run_deta_infer.sh"