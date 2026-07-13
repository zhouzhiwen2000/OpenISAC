---
title: Installation
description: Software dependencies needed to build and operate OpenISAC.
---

## Backend (C++)

### Operating System
*   **Ubuntu 24.04 LTS**
    *   Download: [http://www.ubuntu.com/download/desktop](http://www.ubuntu.com/download/desktop)
*   **macOS (Apple Silicon, local development / demo only)**
    *   Guide: [Separate macOS build guide](https://github.com/zhouzhiwen2000/OpenISAC/blob/main/docs/macos_build.md)

### Dependencies & Installation

### 1. UHD (USRP Hardware Driver)
Install the UHD toolchain by following the official Ettus guide (Please follow the tutorial for Ubuntu 24.04):
*   [Building and Installing the USRP Open-Source Toolchain on Linux](https://kb.ettus.com/Building_and_Installing_the_USRP_Open-Source_Toolchain_(UHD_and_GNU_Radio)_on_Linux#Update_and_Install_dependencies)

> **Note:** This code has been tested on UHD v4.9.0.1. You can checkout this version using `git checkout v4.9.0.1`.

### 2. Install Aff3ct
This project uses the Aff3ct library for Forward Error Correction (FEC). Install it from source:

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

### 3. Clone Repository
Clone the OpenISAC repository:

```bash
cd ~
git clone https://github.com/zhouzhiwen2000/OpenISAC.git
```

### 4. Configuration
The system uses YAML files for runtime configuration.

*   **Config filenames**:
    `BS` reads `BS.yaml`, and `UE` reads `UE.yaml` from the working directory.
*   **First run**:
    Template YAML files live in `config/`. Copy `config/BS_X310.yaml` / `config/BS_B210.yaml` or
    `config/UE_X310.yaml` / `config/UE_B210.yaml` to `BS.yaml` / `UE.yaml`, then edit them in place.
    For the B210 TDD duplex preset, copy `config/BS_B210_Duplex.yaml` and `config/UE_B210_Duplex.yaml`.

## Frontend (Python)

It is recommended to use a `conda` or `venv` environment with **Python 3.13**.

**Miniconda Installation Guide:**
*   **Windows:** [Miniconda Installation for Windows](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-powershell)
*   **Linux:** [Miniconda Installation for Linux](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2)

**Create a new conda environment:**
```bash
conda create -n OpenISAC python=3.13
conda activate OpenISAC
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** `ffmpeg` is required for video streaming demonstrations.
*   **Ubuntu:** `sudo apt install ffmpeg`
*   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or place executable in the working directory.

### Enable GPU Acceleration (Optional)

If an Nvidia GPU is available, install `cupy-cuda12x` to enable GPU acceleration:

> **Note:** Please ensure that the CUDA Toolkit is installed before installing CuPy.

```bash
pip install cupy-cuda12x
```

### Enable Intel Integrated GPU Acceleration (Optional)

If your device has an Intel integrated GPU (e.g., Intel UHD Graphics, Intel Iris Xe, etc.), you can enable GPU acceleration via `dpctl` and `dpnp`. This is especially useful for laptops and desktops without a dedicated Nvidia GPU.

#### 1. Install Intel GPU Official Driver

First, ensure your system has the latest Intel GPU driver installed:
*   **Windows:** Download and install the latest driver from Intel Download Center: [https://www.intel.com/content/www/us/en/download-center/home.html](https://www.intel.com/content/www/us/en/download-center/home.html)
*   **Ubuntu:** Install Intel compute-runtime:
    ```bash
    sudo apt install intel-opencl-icd libze-intel-gpu1 libze1 intel-media-va-driver-non-free
    ```

#### 2. Install Python Dependencies

```bash
pip install dpctl dpnp
```

#### 3. Verify Installation

Run the following command to check if the Intel GPU is correctly recognized:

```bash
python -c "import dpctl; print(dpctl.get_devices())"
```

If the installation is successful, you should see output similar to:
```
[<dpctl.SyclDevice [backend_type.level_zero, device_type.gpu, Intel(R) UHD Graphics] at 0x...>]
```

#### 4. Usage Notes

The OpenISAC frontend automatically detects available GPU backends. The priority order is:
1. Nvidia GPU (CUDA)
2. Intel iGPU (dpnp)
3. CPU (fallback)

No code modifications are required; the system will automatically select the best available backend.
