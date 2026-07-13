---
title: macOS 构建与开发
description: 在 macOS 上安装依赖、构建 CPU 版本并运行 Python 感知前端。
---

本页适用于在 macOS 上搭建本地开发或演示环境。OpenISAC 的 USRP 实验仍以 Ubuntu 24.04 为主要运行平台；macOS 适合编译 CPU 版本、编辑文档、运行离线分析和感知前端，但不支持 CUDA，也不能替代 Linux 上的实时硬件验证。

使用前请注意：

- `isolate_cpus.bash` 等实时调优脚本仅支持 Linux。
- UHD 只保留一套安装来源，不要在同一个 shell 环境中混用 Homebrew 和 MacPorts 版本。
- 涉及 UHD、USRP、实时调度、USB 传输或网络吞吐的功能，最终应在实际运行 UHD 的 Linux 主机上验证。

## 1. 安装基础依赖

先安装 Xcode Command Line Tools：

```bash
xcode-select --install
```

再安装 Homebrew 依赖：

```bash
brew install cmake pkgconf uhd boost fftw yaml-cpp libomp ffmpeg
```

确认当前 shell 使用的是 Homebrew 提供的 UHD：

```bash
which uhd_find_devices
uhd_config_info --version
```

正常情况下，`uhd_find_devices` 位于 `/opt/homebrew/bin`，`uhd_config_info --version` 显示 `UHD 4.9.0.1`。如果系统中还安装了 MacPorts 版 UHD，请将其卸载，或确保 Homebrew 路径在 `PATH` 中优先。

## 2. 安装 AFF3CT

OpenISAC 依赖 AFF3CT。由于目前没有对应的 Homebrew 公式，建议从源码安装到用户目录：

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

克隆仓库，并显式使用 AppleClang 和刚才安装的 AFF3CT：

```bash
cd ~
git clone https://github.com/zhouzhiwen2000/OpenISAC.git OpenISAC
cd OpenISAC
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DAFF3CT_ROOT="$AFF3CT_PREFIX"
cmake --build build -j4
```

构建成功后会生成 CPU 版本：

- `build/BS`
- `build/UE`

## 4. 准备运行配置

BS 和 UE 会从当前工作目录读取 YAML。以 B210 模板为例：

```bash
cd build
cp ../config/BS_B210.yaml BS.yaml
cp ../config/UE_B210.yaml UE.yaml
```

使用 X310 时，请改用对应的 `X310` 模板。

## 5. 运行 CPU 后端

在 `build/` 目录运行所需程序：

```bash
./BS
./UE
```

macOS 会跳过 `mlockall()`，这是预期行为。如果只是检查无硬件启动流程，程序应在 YAML、端口或设备初始化阶段给出明确错误，而不是异常崩溃。

## 6. 运行 Python 感知前端

建议使用仓库提供的 conda 环境：

```bash
conda env create -f environment.openisac-plot.yml
conda activate openisac-plot
python scripts/plot_sensing_fast.py
```

在 Apple Silicon 上，环境文件会按 `Darwin + arm64` 条件安装 `mlx`。快速感知前端会检测 MLX/Metal 是否可用；不可用时自动回退到 CPU。
