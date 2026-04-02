# macOS 构建教程

本教程面向 macOS 上的本地开发和演示环境。

适用范围与限制：

- 主 README 仍然以 Ubuntu 24.04 作为后端主平台。
- macOS 仅支持 CPU 版构建，不支持 CUDA 目标。
- `isolate_cpus.bash` 等实时调优脚本是 Linux 专用的。
- UHD 请只保留一套来源。不要在同一个 shell 环境里混用 Homebrew 和 MacPorts 的 UHD。

## 1. 安装前置依赖

先安装 Xcode Command Line Tools：

```bash
xcode-select --install
```

安装 Homebrew 依赖：

```bash
brew install cmake pkgconf uhd boost fftw yaml-cpp libomp ffmpeg
```

检查当前 shell 是否命中了 Homebrew 的 UHD：

```bash
which uhd_find_devices
uhd_config_info --version
```

期望结果：

- `uhd_find_devices` 位于 `/opt/homebrew/bin`
- `uhd_config_info --version` 显示 `UHD 4.9.0.1`

如果你系统里还装了 MacPorts 版 UHD，请卸载它，或确保 Homebrew 在 `PATH` 里优先。

## 2. 安装 AFF3CT

这个仓库依赖 AFF3CT，目前没有现成的 Homebrew 公式，因此建议源码安装到用户目录，而不是系统目录。

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

## 3. 构建 OpenISAC

克隆仓库，并显式指定 AppleClang：

```bash
cd ~
git clone https://github.com/zhouzhiwen2000/UHD_OFDM.git
cd UHD_OFDM
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DAFF3CT_ROOT="$AFF3CT_PREFIX"
cmake --build build -j4
```

成功后会生成 CPU 版二进制：

- `build/OFDMModulator`
- `build/OFDMDemodulator`

## 4. 准备运行 YAML

可执行文件会从当前工作目录读取 YAML。

```bash
cd build
cp ../config/Modulator_B210.yaml Modulator.yaml
cp ../config/Demodulator_B210.yaml Demodulator.yaml
```

如果你使用的是 X310，请改用对应的 `X310` 模板。

## 5. 在 macOS 上运行

示例：

```bash
cd build
./OFDMModulator
./OFDMDemodulator
```

说明：

- macOS 上会跳过 `mlockall()`，这是预期行为。
- 如果只是做无硬件启动验证，从 `build/` 启动时，程序应当停在 YAML 缺失或设备/端口初始化阶段，而不是异常崩溃。

## 6. macOS 上的 Python 前端

快速感知前端建议使用仓库内的 conda 环境文件：

```bash
conda env create -f environment.openisac-plot.yml
conda activate openisac-plot
python scripts/plot_sensing_fast.py
```

Apple Silicon 说明：

- `mlx` 已作为 `Darwin + arm64` 条件依赖写入环境文件
- 快速前端会安全探测 `MLX`，若 Metal 不可用则自动回退到 CPU
