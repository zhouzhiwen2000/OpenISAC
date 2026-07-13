---
title: 软件安装
description: 编译和运行 OpenISAC 所需的软件依赖。
---

## 后端 (C++)

### 操作系统
*   **Ubuntu 24.04 LTS**
    *   下载: [http://www.ubuntu.com/download/desktop](http://www.ubuntu.com/download/desktop)
*   **macOS (Apple Silicon，仅建议用于本地开发 / 演示)**
    *   教程: [独立 macOS 构建教程](../../tools-workflows/macos-development/)

### 依赖项和安装

### 1. UHD (USRP 硬件驱动程序)
按照 Ettus 官方指南安装 UHD 工具链 (请遵循 Ubuntu 24.04 的教程):
*   [在 Linux 上构建和安装 USRP 开源工具链](https://kb.ettus.com/Building_and_Installing_the_USRP_Open-Source_Toolchain_(UHD_and_GNU_Radio)_on_Linux#Update_and_Install_dependencies)

> **注意:** 此代码已在 UHD v4.9.0.1 上测试。您可以使用 `git checkout v4.9.0.1` 检出此版本。

### 2. 安装 Aff3ct
本项目使用 Aff3ct 库进行前向纠错 (FEC)。从源码安装：

```bash
sudo apt-get install nlohmann-json3-dev
git clone https://github.com/aff3ct/aff3ct.git
cd aff3ct
git submodule update --init --recursive
mkdir build
cd build
cmake .. -G"Unix Makefiles" -DCMAKE_CXX_COMPILER="g++" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_CXX_FLAGS="-funroll-loops -march=native" -DAFF3CT_COMPILE_EXE="OFF" -DAFF3CT_COMPILE_SHARED_LIB="ON" -DSPU_STACKTRACE="OFF" -DSPU_STACKTRACE_SEGFAULT="OFF" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -faligned-new"
make -j$(nproc)
sudo make install
```

### 3. 克隆仓库
克隆 OpenISAC 仓库：

```bash
cd ~
git clone https://github.com/zhouzhiwen2000/OpenISAC.git
```

### 4. 配置
系统使用 YAML 文件管理参数。

*   **配置文件名**: `BS` 读取 `BS.yaml`，`UE` 读取 `UE.yaml`。
*   **首次运行**: 模板 YAML 统一放在 `config/` 目录。请将 `config/BS_X310.yaml` / `config/BS_B210.yaml` 或
    `config/UE_X310.yaml` / `config/UE_B210.yaml` 复制为 `BS.yaml` / `UE.yaml`，再按需修改。
    B210 TDD 双工预设可直接复制 `config/BS_B210_Duplex.yaml` 和 `config/UE_B210_Duplex.yaml`。

## 前端 (Python)

建议使用 **Python 3.13** 的 `conda` 或 `venv` 环境。

**Miniconda 安装教程:**
*   **Windows:** [Miniconda Windows 安装教程](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-powershell)
*   **Linux:** [Miniconda Linux 安装教程](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2)

**新建 conda 环境:**
```bash
conda create -n OpenISAC python=3.13
conda activate OpenISAC
```

### 安装依赖项

```bash
pip install -r requirements.txt
```

**注意:** 视频流演示需要 `ffmpeg`。
*   **Ubuntu:** `sudo apt install ffmpeg`
*   **Windows:** 从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载并添加到 PATH，或将可执行文件放置在工作目录中。

### 启用 GPU 加速 (可选)

如果有 Nvidia GPU，请安装 `cupy-cuda12x` 以启用 GPU 加速：

> **注意:** 安装 CuPy 之前，请务必先安装 CUDA Toolkit。

```bash
pip install cupy-cuda12x
```

### 启用 Intel 集成显卡加速 (可选)

如果您的设备配备 Intel 集成显卡（如 Intel UHD Graphics、Intel Iris Xe 等），可以通过 `dpctl` 和 `dpnp` 启用 GPU 加速。这对于没有 Nvidia 独立显卡的笔记本电脑和台式机特别有用。

#### 1. 安装 Intel 显卡官方驱动

首先确保您的系统已安装最新的 Intel 显卡驱动：
*   **Windows:** 从 Intel 官方下载中心下载并安装最新驱动: [https://www.intel.com/content/www/us/en/download-center/home.html](https://www.intel.com/content/www/us/en/download-center/home.html)
*   **Ubuntu:** 安装 Intel compute-runtime:
    ```bash
    sudo apt install intel-opencl-icd libze-intel-gpu1 libze1 intel-media-va-driver-non-free
    ```

#### 2. 安装 Python 依赖

```bash
pip install dpctl dpnp
```

#### 3. 验证安装

运行以下命令检查 Intel GPU 是否被正确识别：

```bash
python -c "import dpctl; print(dpctl.get_devices())"
```

如果安装成功，您应该能看到类似以下的输出：
```
[<dpctl.SyclDevice [backend_type.level_zero, device_type.gpu, Intel(R) UHD Graphics] at 0x...>]
```

#### 4. 使用说明

OpenISAC 前端会自动检测可用的 GPU 后端。优先级顺序为：
1. Nvidia GPU (CUDA)
2. Intel iGPU (dpnp)
3. CPU (回退选项)

无需修改代码，系统会自动选择最佳可用后端。
