---
title: macOS Build and Development
description: Install dependencies, build the CPU backend, and run the Python sensing frontend on macOS.
---

This page covers local development and demonstration use on macOS. Ubuntu 24.04 remains the primary platform for OpenISAC USRP experiments. macOS is suitable for CPU builds, documentation, offline analysis, and sensing frontends, but it does not support CUDA and cannot replace real-time hardware validation on Linux.

Before you begin:

- Real-time tuning helpers such as `isolate_cpus.bash` are Linux-only.
- Keep only one UHD package source; do not mix Homebrew and MacPorts UHD installations in the same shell environment.
- Validate UHD, USRP, real-time scheduling, USB, and network-throughput behavior on the Linux hardware host.

## 1. Install Prerequisites

Install Xcode Command Line Tools:

```bash
xcode-select --install
```

Install the Homebrew dependencies:

```bash
brew install cmake pkgconf uhd boost fftw yaml-cpp libomp ffmpeg
```

Confirm that the shell uses Homebrew UHD:

```bash
which uhd_find_devices
uhd_config_info --version
```

Normally, `uhd_find_devices` resolves under `/opt/homebrew/bin`, and `uhd_config_info --version` reports `UHD 4.9.0.1`. If MacPorts UHD is also installed, remove it or ensure Homebrew appears first in `PATH`.

## 2. Install AFF3CT

OpenISAC requires AFF3CT. Because no Homebrew formula is currently available, install it from source into a user-local prefix:

```bash
brew install nlohmann-json
git clone https://github.com/aff3ct/aff3ct.git
cd aff3ct
git submodule update --init --recursive
export AFF3CT_PREFIX="$HOME/.local/openisac-aff3ct"
mkdir -p build
cd build
cmake .. \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$AFF3CT_PREFIX" \
  -DAFF3CT_COMPILE_EXE=OFF \
  -DAFF3CT_COMPILE_SHARED_LIB=ON \
  -DSPU_STACKTRACE=OFF \
  -DSPU_STACKTRACE_SEGFAULT=OFF \
  -DCMAKE_CXX_FLAGS="-funroll-loops -march=native -faligned-new"
cmake --build . -j4
cmake --install .
```

## 3. Build OpenISAC

Clone the repository and explicitly select AppleClang and the AFF3CT installation:

```bash
cd ~
git clone https://github.com/zhouzhiwen2000/OpenISAC.git OpenISAC
cd OpenISAC
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DAFF3CT_ROOT="$AFF3CT_PREFIX"
cmake --build build -j4
```

Successful builds produce the CPU binaries:

- `build/BS`
- `build/UE`

## 4. Prepare Runtime Configuration

BS and UE load YAML from the current working directory. For B210 templates:

```bash
cd build
cp ../config/BS_B210.yaml BS.yaml
cp ../config/UE_B210.yaml UE.yaml
```

Use the corresponding `X310` templates when running an X310.

## 5. Run the CPU Backend

Run the required programs from `build/`:

```bash
./BS
./UE
```

OpenISAC skips `mlockall()` on macOS; this is expected. For a startup check without hardware, the program should report a clear YAML, socket, or device-initialization error rather than crash.

## 6. Run the Python Sensing Frontend

Use the provided conda environment:

```bash
conda env create -f environment.openisac-plot.yml
conda activate openisac-plot
python scripts/plot_sensing_fast.py
```

On Apple Silicon, the environment installs `mlx` conditionally for `Darwin + arm64`. The fast sensing frontend probes MLX/Metal and automatically falls back to CPU when it is unavailable.
