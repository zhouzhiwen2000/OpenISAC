# macOS Build Guide

This guide is for local development and demo use on macOS.

Scope and limitations:

- Ubuntu 24.04 remains the primary backend platform documented in the main README.
- macOS builds are CPU-only. CUDA targets are not available.
- Real-time tuning helpers such as `isolate_cpus.bash` are Linux-specific.
- Use one package manager for UHD. Do not mix Homebrew and MacPorts UHD installs in the same shell environment.

## 1. Install prerequisites

Install Xcode Command Line Tools first:

```bash
xcode-select --install
```

Install Homebrew dependencies:

```bash
brew install cmake pkgconf uhd boost fftw yaml-cpp libomp ffmpeg
```

Check that the Homebrew UHD toolchain is the one visible in your shell:

```bash
which uhd_find_devices
uhd_config_info --version
```

Expected result:

- `uhd_find_devices` should resolve under `/opt/homebrew/bin`
- `uhd_config_info --version` should report `UHD 4.9.0.1`

If you also have MacPorts UHD installed, remove it or make sure Homebrew comes first in `PATH`.

## 2. Install AFF3CT

This repository requires AFF3CT and does not currently provide a Homebrew formula for it.
Install it into a user-local prefix instead of a system directory.

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

Clone the repository and configure with AppleClang explicitly:

```bash
cd ~
git clone https://github.com/zhouzhiwen2000/OpenISAC.git
cd UHD_OFDM
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DAFF3CT_ROOT="$AFF3CT_PREFIX"
cmake --build build -j4
```

Expected CPU binaries:

- `build/OFDMModulator`
- `build/OFDMDemodulator`

## 4. Prepare runtime YAML files

The executables load YAML from the current working directory.

```bash
cd build
cp ../config/Modulator_B210.yaml Modulator.yaml
cp ../config/Demodulator_B210.yaml Demodulator.yaml
```

Use the `X310` templates instead if needed.

## 5. Run on macOS

Example:

```bash
cd build
./OFDMModulator
./OFDMDemodulator
```

Notes:

- On macOS the code skips `mlockall()`. That is expected.
- If you only want to validate startup without hardware, running from `build/` should stop at missing YAML or socket/device setup rather than crash.

## 6. Python frontend on macOS

For the fast sensing viewers:

```bash
conda env create -f environment.openisac-plot.yml
conda activate openisac-plot
python scripts/plot_sensing_fast.py
```

Apple Silicon note:

- `mlx` is listed as a conditional dependency for `Darwin + arm64`
- The fast viewers probe `MLX` safely and fall back to CPU if Metal is unavailable
