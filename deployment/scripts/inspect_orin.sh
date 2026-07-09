#!/usr/bin/env bash
# =============================================================================
# inspect_orin.sh
#
# Read-only environment inspection for NVIDIA Jetson Orin.
# Collects hardware, OS, JetPack, CUDA, TensorRT, Python information.
#
# Usage:
#   bash deployment/scripts/inspect_orin.sh
#   bash deployment/scripts/inspect_orin.sh 2>&1 | tee orin_env.txt
#
# SAFE: Does NOT install packages, modify system, or build TensorRT.
# =============================================================================

set -euo pipefail
# Don't abort on individual check failures — continue and report
set +e

SEP="=================================================================="
WARN="[WARN]"
OK="[OK]"
SKIP="[SKIP]"

echo "$SEP"
echo " NVIDIA Orin Environment Inspection"
echo " Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "$SEP"

# =============================================================================
echo ""
echo "=== HARDWARE ==="
# =============================================================================

echo "--- Device model ---"
if [ -f /proc/device-tree/model ]; then
    cat /proc/device-tree/model && echo ""
    echo "$OK /proc/device-tree/model"
else
    echo "$WARN /proc/device-tree/model not found"
fi

echo ""
echo "--- Architecture ---"
uname -m && echo "$OK uname -m" || echo "$WARN uname -m failed"

echo ""
echo "--- CPU info ---"
if [ -f /proc/cpuinfo ]; then
    grep -m 4 "Hardware\|Processor\|model name\|cpu MHz" /proc/cpuinfo || true
    CPU_CORES=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "unknown")
    echo "  Logical cores: $CPU_CORES"
    echo "$OK /proc/cpuinfo"
else
    echo "$WARN /proc/cpuinfo not found"
fi

echo ""
echo "--- Memory ---"
if [ -f /proc/meminfo ]; then
    grep -E "^MemTotal:|^MemAvailable:|^SwapTotal:" /proc/meminfo
    echo "$OK /proc/meminfo"
else
    echo "$WARN /proc/meminfo not found"
fi

echo ""
echo "--- Disk space ---"
df -h / 2>/dev/null && echo "$OK df -h /" || echo "$WARN df -h failed"

# =============================================================================
echo ""
echo "=== OS ==="
# =============================================================================

echo "--- OS release ---"
if [ -f /etc/os-release ]; then
    cat /etc/os-release
    echo "$OK /etc/os-release"
else
    echo "$WARN /etc/os-release not found"
fi

echo ""
echo "--- Kernel ---"
uname -r && echo "$OK uname -r" || echo "$WARN uname -r failed"

# =============================================================================
echo ""
echo "=== JETSON / L4T ==="
# =============================================================================

echo "--- nv_tegra_release ---"
if [ -f /etc/nv_tegra_release ]; then
    cat /etc/nv_tegra_release
    echo "$OK /etc/nv_tegra_release"
else
    echo "$WARN /etc/nv_tegra_release not found (may not be Jetson or different L4T layout)"
fi

echo ""
echo "--- Tegra chip ID ---"
if [ -f /sys/module/tegra_fuse/parameters/tegra_chip_id ]; then
    cat /sys/module/tegra_fuse/parameters/tegra_chip_id && echo ""
    echo "$OK tegra_chip_id"
else
    echo "$SKIP tegra_chip_id sysfs not present"
fi

# =============================================================================
echo ""
echo "=== JETPACK ==="
# =============================================================================

echo "--- nvidia-jetpack package ---"
if command -v dpkg >/dev/null 2>&1; then
    dpkg -l nvidia-jetpack 2>/dev/null | grep -v "^[|+]" || echo "$SKIP nvidia-jetpack not installed via dpkg"
else
    echo "$SKIP dpkg not available"
fi

echo ""
echo "--- L4T version via apt ---"
if command -v apt-cache >/dev/null 2>&1; then
    apt-cache show nvidia-l4t-core 2>/dev/null | grep -E "^Version:|^Package:" | head -4 || \
        echo "$SKIP nvidia-l4t-core package not found"
else
    echo "$SKIP apt-cache not available"
fi

# =============================================================================
echo ""
echo "=== CUDA ==="
# =============================================================================

echo "--- nvcc version ---"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
    echo "$OK nvcc found at: $(which nvcc)"
else
    echo "$WARN nvcc not in PATH"
    for p in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [ -f "$p" ]; then
            echo "  Found at: $p"
            "$p" --version
            break
        fi
    done
fi

echo ""
echo "--- CUDA libraries ---"
CUDA_LIB_PATHS="/usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu"
for p in $CUDA_LIB_PATHS; do
    if [ -d "$p" ]; then
        echo "  $p:"
        ls "$p"/libcuda*.so* 2>/dev/null | head -4 || echo "    (no libcuda)"
        ls "$p"/libcudart*.so* 2>/dev/null | head -4 || echo "    (no libcudart)"
    fi
done

echo ""
echo "--- CUDA_HOME / LD_LIBRARY_PATH ---"
echo "  CUDA_HOME=${CUDA_HOME:-<not set>}"
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<not set>}"

# =============================================================================
echo ""
echo "=== TENSORRT ==="
# =============================================================================

echo "--- TensorRT dpkg packages ---"
if command -v dpkg >/dev/null 2>&1; then
    dpkg -l | grep -i tensorrt | grep -v "^un" || echo "$SKIP no TensorRT dpkg packages found"
else
    echo "$SKIP dpkg not available"
fi

echo ""
echo "--- libnvinfer version ---"
NVINFER_FOUND=false
for p in /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/local/lib; do
    if ls "$p"/libnvinfer.so* 2>/dev/null | head -2; then
        echo "$OK libnvinfer found in $p"
        NVINFER_FOUND=true
        break
    fi
done
if [ "$NVINFER_FOUND" = false ]; then
    echo "$WARN libnvinfer.so not found in common paths"
    find /usr -name "libnvinfer.so*" 2>/dev/null | head -5 || true
fi

echo ""
echo "--- libnvinfer-dev headers ---"
if [ -f /usr/include/NvInfer.h ]; then
    head -5 /usr/include/NvInfer.h | grep -E "define NV_TENSORRT" || true
    echo "$OK NvInfer.h found"
else
    echo "$SKIP NvInfer.h not found"
fi

# =============================================================================
echo ""
echo "=== TRTEXEC ==="
# =============================================================================

echo "--- trtexec path ---"
TRTEXEC_PATH=""
if command -v trtexec >/dev/null 2>&1; then
    TRTEXEC_PATH=$(which trtexec)
    echo "$OK trtexec in PATH: $TRTEXEC_PATH"
else
    for p in /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec /opt/tensorrt/bin/trtexec; do
        if [ -f "$p" ]; then
            TRTEXEC_PATH="$p"
            echo "$OK trtexec found at: $p"
            break
        fi
    done
    if [ -z "$TRTEXEC_PATH" ]; then
        echo "$WARN trtexec not found"
    fi
fi

echo ""
echo "--- trtexec version ---"
if [ -n "$TRTEXEC_PATH" ]; then
    "$TRTEXEC_PATH" --help 2>&1 | head -3 || true
else
    echo "$SKIP trtexec not available"
fi

# =============================================================================
echo ""
echo "=== PYTHON ==="
# =============================================================================

echo "--- python3 version ---"
if command -v python3 >/dev/null 2>&1; then
    python3 --version && echo "$OK python3 found at: $(which python3)"
else
    echo "$WARN python3 not in PATH"
fi

echo ""
echo "--- pip packages (relevant) ---"
if command -v python3 >/dev/null 2>&1; then
    python3 -m pip list 2>/dev/null | grep -iE "onnx|tensorrt|numpy|opencv|torch" | head -20 || \
        echo "$SKIP pip list failed or no matching packages"
else
    echo "$SKIP python3 not available"
fi

echo ""
echo "--- Python environment check ---"
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PYEOF'
import sys
print(f"  Python: {sys.version}")
try:
    import torch
    print(f"  torch={torch.__version__}  CUDA available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version (torch): {torch.version.cuda}")
except ImportError:
    print("  torch: not installed")
try:
    import onnxruntime as ort
    print(f"  onnxruntime={ort.__version__}  providers={ort.get_available_providers()}")
except ImportError:
    print("  onnxruntime: not installed")
try:
    import tensorrt as trt
    print(f"  tensorrt={trt.__version__}")
except ImportError:
    print("  tensorrt python binding: not installed")
try:
    import cv2
    print(f"  opencv-python={cv2.__version__}")
except ImportError:
    print("  opencv-python: not installed")
try:
    import numpy as np
    print(f"  numpy={np.__version__}")
except ImportError:
    print("  numpy: not installed")
PYEOF
fi

# =============================================================================
echo ""
echo "=== NVIDIA TOOLS ==="
# =============================================================================

echo "--- tegrastats ---"
if command -v tegrastats >/dev/null 2>&1; then
    echo "$OK tegrastats found at: $(which tegrastats)"
else
    echo "$WARN tegrastats not in PATH"
fi

echo ""
echo "--- nvpmodel ---"
if command -v nvpmodel >/dev/null 2>&1; then
    echo "$OK nvpmodel found at: $(which nvpmodel)"
    nvpmodel -q 2>/dev/null || echo "  (nvpmodel -q failed, may need sudo)"
else
    echo "$WARN nvpmodel not in PATH"
fi

echo ""
echo "--- jetson_clocks ---"
if command -v jetson_clocks >/dev/null 2>&1; then
    echo "$OK jetson_clocks found at: $(which jetson_clocks)"
else
    echo "$WARN jetson_clocks not in PATH"
fi

echo ""
echo "--- jtop (optional) ---"
if command -v jtop >/dev/null 2>&1; then
    echo "$OK jtop found at: $(which jtop)"
else
    echo "$SKIP jtop not installed (optional monitoring tool)"
fi

# =============================================================================
echo ""
echo "$SEP"
echo " Inspection complete."
echo " Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo " Save output: bash inspect_orin.sh 2>&1 | tee orin_env.txt"
echo "$SEP"
