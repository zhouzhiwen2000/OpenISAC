<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/logo.svg">
    <img src="docs/images/logo.svg" alt="OpenISAC Logo" width="400">
  </picture>
</p>

# OpenISAC

[English Version](README.md) | [更新日志](CHANGELOG.md)

OpenISAC 是一个面向实时实验的 OFDM 通信感知一体化（ISAC）平台，重点服务于学术研究和 PHY 层快速迭代。

它试图填补“纯仿真代码”和“完整标准协议栈”之间的空档：既尽量保持代码简洁、容易修改，又能基于 USRP 直接开展空口实验。

如果你的目标是用一个足够轻量、易读、可快速改造的系统，把算法思路尽快跑到 OTA 实验里，这个仓库适合你；如果你需要 Wi-Fi/5G NR 互操作性或生产级协议栈，它并不适合。

## 项目亮点

- 支持实时 OFDM 通信，以及单站和双站感知
- 支持双站实验所需的空口同步机制
- 后端采用 C++ 实时链路，前端提供 Python 可视化与工具脚本
- 基于 YAML 的运行时配置，仓库内提供 X310/B210 样例，也可扩展到其他 UHD 支持的 USRP
- 自带 CPU 隔离、绘图分析、网页配置与进程控制工具

## 一眼看懂

| 组件 | 主要入口 | 作用 |
| :--- | :--- | :--- |
| BS 后端 | `OFDMModulator`、`config/Modulator_*.yaml` | 发送 OFDM 帧、接收业务 UDP，并输出单站感知结果 |
| UE 后端 | `OFDMDemodulator`、`config/Demodulator_*.yaml` | 接收解调 OFDM 帧、输出业务数据，并运行双站感知 |
| 前端工具 | `scripts/plot_*.py`、`scripts/config_web_editor.py` | 显示感知/信道结果，并编辑运行时配置 |

## 快速导航

- 环境准备与安装: [硬件准备](#硬件准备)、[软件安装](#软件安装)
- 先跑起来: [典型使用示例](#典型使用示例)
- 运行参数说明: [OFDM 调制器 (OFDMModulator)](#ofdm-调制器-ofdmmodulator)、[OFDM 解调器 (OFDMDemodulator)](#ofdm-解调器-ofdmdemodulator)
- 网页控制台: [网页配置控制台](#7-网页配置控制台)
- 最近更新: [更新日志](CHANGELOG.md)

## 仓库结构

| 路径 | 说明 |
| :--- | :--- |
| `src/`、`include/` | 核心 C++ PHY、感知、线程与运行时逻辑 |
| `config/` | 不同角色的 YAML 样例配置，内含 X310/B210 示例，也可作为其他 USRP 的起点 |
| `scripts/` | Python 前端、网页配置控制台、Linux 性能调优脚本 |
| `capture/` | 离线感知结果绘图工具 |
| `docs/` | 项目静态站点，以及架构/信号处理说明页 |

## 常见工作流

| 目标 | 后端程序 | 常用配置 | 常用前端 |
| :--- | :--- | :--- | :--- |
| 运行 BS 端 | `OFDMModulator` | `config/Modulator_X310.yaml` 或 `config/Modulator_B210.yaml` | `plot_sensing_fast.py` |
| 运行 UE 端 | `OFDMDemodulator` | `config/Demodulator_X310.yaml` 或 `config/Demodulator_B210.yaml` | `plot_bi_sensing_fast.py` |
| 用浏览器调参数 | `scripts/config_web_editor.py` | 读取 `build/Modulator.yaml` 和 `build/Demodulator.yaml` | 浏览器访问 `http://<host>:8765` |

## 首次 OTA 运行前检查

- 准备两台后端节点：BS 端需要一路 TX 和一路 RX 天线链路，UE 端需要一路 RX 链路。
- 先选一份最接近你硬件的 YAML 模板。仓库自带 X310/B210 示例，但项目并不限于这两种 USRP。
- 运行时 YAML 需要放在 `build/` 目录，因为两个二进制程序都会从当前工作目录读取 `Modulator.yaml` 或 `Demodulator.yaml`。
- 如果前端跑在另一台机器上，请把运行时 YAML 中的 `default_ip` 改成那台机器的 IP。
- 如果你关心实时稳定性，请先执行 `scripts/set_performance.bash`，然后先用 `sudo ../scripts/isolate_cpus.bash` 做 CPU 隔离，再用 `sudo ../scripts/isolate_cpus.bash run ...` 启动程序。

## 它是什么，又不是什么？

### OpenISAC 是什么？

- 一个面向通信感知一体化研究的极简 OFDM PHY
- 适合原型验证、学术实验与快速算法迭代
- 更强调代码可读性、易改性和实验效率，而不是全栈完备性

### OpenISAC 不是什么？

- 不是标准兼容实现，不以 Wi-Fi 或 5G NR 兼容为目标
- 不是 openwifi、OpenAirInterface 这类全栈系统的替代品
- 不是生产级通信协议栈

### 何时使用它

- 快速验证新的 OFDM/ISAC 算法
- 将同步、感知或 PHY 思路尽快跑到空口实验
- 不要求互操作性的科研环境

### 何时不使用它

- 需要构建兼容 Wi-Fi/NR 的系统
- 需要完整 MAC/协议栈、互操作性或面向认证的行为

## 引用

如果您觉得这个仓库有用，请引用我们的论文：

> Z. Zhou, C. Zhang, X. Xu, and Y. Zeng, "OpenISAC: An Open-Source Real-Time Experimentation Platform for OFDM-ISAC with Over-the-Air Synchronization," *arXiv preprint* arXiv:2601.03535, Jan. 2026.
>
> [[arXiv](https://arxiv.org/pdf/2601.03535)]

## 作者

- 周智文 (zhiwen_zhou@seu.edu.cn)
- 张超越 (chaoyue_zhang@seu.edu.cn)
- 徐晓莉 (Member, IEEE) (xiaolixu@seu.edu.cn)
- 曾勇 (Fellow, IEEE) (yong_zeng@seu.edu.cn)

## 所属机构

<img src="docs/images/SEUlogo.png" height="80" alt="SEU Logo" style="border:none; box-shadow:none;"> &nbsp;&nbsp; <img src="docs/images/PML.png" height="80" alt="PML Logo" style="border:none; box-shadow:none;">

**东南大学移动通信国家重点实验室 & 紫金山实验室 曾勇课题组**

## 社区

- [加入我们的 QQ 群](https://qm.qq.com/q/NIQRNGb0kY)
- [Bilibili 频道 (曾勇课题组)](https://space.bilibili.com/627920129)
- 微信公众号:

  <img src="docs/images/WeChat.jpg" width="150" alt="WeChat QR Code">

## 硬件准备
 
### 后端 (C++)
 
要设置完整的系统，您需要以下硬件：
 
*   **USRP 设备**: 2 台 (例如 USRP X310, B210 等)
*   **计算机**: 2 台 (推荐使用高主频CPU)
*   **天线**: 3 根
*   **OCXO/GPSDO**: 2 个 (两台 USRP 都需要)
 
#### 连接设置
 
该系统由两个主要节点组成：
 
1.  **基站 (BS) 节点**
    *   **硬件**: 1x 计算机, 1x USRP。
    *   **天线**: 连接 2 根天线到此 USRP (1 发送, 1 接收)。
    *   **时钟**: 将 OCXO 或 GPSDO 连接到 USRP 的 REFIN 端口。
    *   **功能**: 发送 OFDM 信号并接收雷达回波。
 
2.  **用户 (UE) 节点**
    *   **硬件**: 1x 计算机, 1x USRP。
    *   **天线**: 连接 1 根天线到此 USRP 的 RX 端口。
    *   **时钟**: 将 OCXO 或 GPSDO 连接到 USRP 的 REFIN 端口。
    *   **高精度 DAC (可选)**: 使用高精度 DAC 来微调 OCXO。
    *   **功能**: 接收 OFDM 信号用于通信和双站感知。
 
#### 接口要求
为了支持高带宽和采样率，请确保计算机和 USRP 之间的连接使用：
*   **>= 10 Gigabit Ethernet (10GbE)** (X 系列)
*   **USB 3.0** (B 系列)
 
### 前端 (Python)
*   **计算机**: 1x 计算机 (Windows 或 Linux)。
    *   可以是后端计算机之一，也可以是单独的机器。
*   **CPU**: 如果没有 GPU，建议使用高性能 CPU (i7 10700 或更高)。
*   **GPU**: 建议使用 Nvidia GPU 进行加速。


## 软件安装
 
### 后端 (C++)
 
#### 操作系统
*   **Ubuntu 24.04 LTS**
    *   下载: [http://www.ubuntu.com/download/desktop](http://www.ubuntu.com/download/desktop)
*   **macOS (Apple Silicon，仅建议用于本地开发 / 演示)**
    *   教程: [独立 macOS 构建教程](https://github.com/zhouzhiwen2000/UHD_OFDM/blob/OpenISAC_MultiChannel/docs/macos_build_zh.md)
 
#### 依赖项和安装
 
#### 1. UHD (USRP 硬件驱动程序)
按照 Ettus 官方指南安装 UHD 工具链 (请遵循 Ubuntu 24.04 的教程):
*   [在 Linux 上构建和安装 USRP 开源工具链](https://kb.ettus.com/Building_and_Installing_the_USRP_Open-Source_Toolchain_(UHD_and_GNU_Radio)_on_Linux#Update_and_Install_dependencies)
 
> **注意:** 此代码已在 UHD v4.9.0.1 上测试。您可以使用 `git checkout v4.9.0.1` 检出此版本。
 
#### 2. 安装 Aff3ct
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
 
#### 3. 克隆仓库
克隆 OpenISAC 仓库：
 
```bash
cd ~
git clone https://github.com/zhouzhiwen2000/OpenISAC.git
```
 
#### 4. 构建 OpenISAC
使用 CMake 构建项目：
 
```bash
# 安装 libyaml-cpp-dev
sudo apt-get install libyaml-cpp-dev

cd OpenISAC
mkdir build
cd build
cmake ..
make -j$(nproc)
```
 
#### 5. 系统性能调优
运行提供的脚本以优化您的系统设置，以满足实时处理需求：
 
```bash
cd ~/OpenISAC
chmod +x scripts/set_performance.bash
./scripts/set_performance.bash
```
 
> **注意:** 如果您需要启用 `RT_RUNTIME_SHARE` 功能，则需要在 BIOS 设置中关闭 `secure_boot`。
 
当 UHD 生成新线程时，它可能会尝试提高线程的调度优先级。如果设置新优先级失败，UHD 软件将向控制台打印警告，如下所示：
 
```text
[WARNING] [UHD] Failed to set desired affinity for thread
```
 
为了解决这个问题，需要给予非特权 (非 root) 用户更改调度优先级的特殊权限。这可以通过创建一个组 `usrp`，将您的用户添加到该组，然后将行 `@usrp - rtprio 99` 追加到文件 `/etc/security/limits.conf` 来启用。
 
```bash
sudo groupadd usrp
sudo usermod -aG usrp $USER
```
 
然后将以下行添加到文件 `/etc/security/limits.conf` 的末尾：
 
```text
@usrp - rtprio  99
```
 
您必须注销并重新登录帐户才能使设置生效。在大多数 Linux 发行版中，组和组成员列表可以在 `/etc/group` 文件中找到。
 
#### 6. CPU 隔离和执行
为了确保稳定的实时性能，可使用 `scripts/isolate_cpus.bash` 将系统服务 (`system.slice`、`user.slice`、`init.scope`) 限制到指定 CPU，并将其余 CPU 保留给应用程序。

以下命令都需要 root 权限（`sudo`）：

```bash
cd ~/OpenISAC
chmod +x scripts/isolate_cpus.bash
sudo ./scripts/isolate_cpus.bash --help
```

**默认隔离策略**

```bash
sudo ./scripts/isolate_cpus.bash
```

- 默认给应用预留前 8 个核心（`0-7`）。
- 系统服务会被限制到其余核心。
- 如果总核心数 `<= 8`，则无法有效隔离，应用与系统都会使用全部核心。

**自定义应用 CPU 集合**

```bash
sudo ./scripts/isolate_cpus.bash 4          # 应用使用 0-3
sudo ./scripts/isolate_cpus.bash 8-15       # 应用使用 8-15
sudo ./scripts/isolate_cpus.bash 0,2,4,6    # 应用使用显式核心列表
```

脚本会将应用核心配置保存到 `/tmp/isolate_cpus_app.conf`。

**CPU 绑核优先级（当核心数量紧张时）**

- 首先给主线程预留一个专用核心。
- 其次优先保证 TX/RX 实时线程。
- 最后再分配调制/解调线程和感知/信号处理线程，因为这些计算线程通常对应更大的缓冲区，可以吸收一定的调度抖动。

在网页 CPU 绑核编辑器里，通常意味着优先保证 `main thread affinity`，其次是 `_tx_proc` / `rx_proc` 和各通道 RX loop，最后再考虑调制、解调和感知处理相关线程。

**在预留核心上运行应用**

```bash
cd build
sudo ../scripts/isolate_cpus.bash run ./OFDMModulator
```

- `run` 会优先读取 `/tmp/isolate_cpus_app.conf` 中保存的应用核心配置。
- 若没有保存配置，`run` 会回退到默认应用核心集合。

> **注意:** 完成隔离后，请始终使用 `sudo ../scripts/isolate_cpus.bash run ...` 启动应用程序。直接执行或手动使用 `taskset` 可能因 slice 亲和性限制而失败。

**重置配置（可选）**

```bash
sudo ./scripts/isolate_cpus.bash reset
```

该命令会移除隔离设置，并将系统 slice 恢复为可使用全部 CPU。

#### 7. 配置
系统使用 YAML 文件管理参数。

*   **配置文件名**: `OFDMModulator` 读取 `Modulator.yaml`，`OFDMDemodulator` 读取 `Demodulator.yaml`。
*   **首次运行**: 模板 YAML 统一放在 `config/` 目录。请将 `config/Modulator_X310.yaml` / `config/Modulator_B210.yaml` 或
    `config/Demodulator_X310.yaml` / `config/Demodulator_B210.yaml` 复制为 `Modulator.yaml` / `Demodulator.yaml`，再按需修改。
 
### 前端 (Python)

建议使用 **Python 3.13** 的 `conda` 或 `venv` 环境。

**Miniconda 安装教程:**
*   **Windows:** [Miniconda Windows 安装教程](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-powershell)
*   **Linux:** [Miniconda Linux 安装教程](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2)

**新建 conda 环境:**
```bash
conda create -n OpenISAC python=3.13
conda activate OpenISAC
```
 
#### 安装依赖项
 
```bash
pip install -r requirements.txt
```

**注意:** 视频流演示需要 `ffmpeg`。
*   **Ubuntu:** `sudo apt install ffmpeg`
*   **Windows:** 从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载并添加到 PATH，或将可执行文件放置在工作目录中。
 
#### 启用 GPU 加速 (可选)
 
如果有 Nvidia GPU，请安装 `cupy-cuda12x` 以启用 GPU 加速：

> **注意:** 安装 CuPy 之前，请务必先安装 CUDA Toolkit。
 
```bash
pip install cupy-cuda12x
```

#### 启用 Intel 集成显卡加速 (可选)

如果您的设备配备 Intel 集成显卡（如 Intel UHD Graphics、Intel Iris Xe 等），可以通过 `dpctl` 和 `dpnp` 启用 GPU 加速。这对于没有 Nvidia 独立显卡的笔记本电脑和台式机特别有用。

##### 1. 安装 Intel 显卡官方驱动

首先确保您的系统已安装最新的 Intel 显卡驱动：
*   **Windows:** 从 Intel 官方下载中心下载并安装最新驱动: [https://www.intel.com/content/www/us/en/download-center/home.html](https://www.intel.com/content/www/us/en/download-center/home.html)
*   **Ubuntu:** 安装 Intel compute-runtime:
    ```bash
    sudo apt install intel-opencl-icd libze-intel-gpu1 libze1 intel-media-va-driver-non-free
    ```

##### 2. 安装 Python 依赖

```bash
pip install dpctl dpnp
```

##### 3. 验证安装

运行以下命令检查 Intel GPU 是否被正确识别：

```bash
python -c "import dpctl; print(dpctl.get_devices())"
```

如果安装成功，您应该能看到类似以下的输出：
```
[<dpctl.SyclDevice [backend_type.level_zero, device_type.gpu, Intel(R) UHD Graphics] at 0x...>]
```

##### 4. 使用说明

OpenISAC 前端会自动检测可用的 GPU 后端。优先级顺序为：
1. Nvidia GPU (CUDA)
2. Intel iGPU (dpnp)
3. CPU (回退选项)

无需修改代码，系统会自动选择最佳可用后端。

## 典型使用示例

### 1. 启动 BS (基站)
```bash
sudo -s
cd build
# 对于 X310:
cp ../config/Modulator_X310.yaml Modulator.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./OFDMModulator

# 对于 B210:
cp ../config/Modulator_B210.yaml Modulator.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./OFDMModulator
```
*如果您使用单独的计算机作为前端，请在 `Modulator.yaml` 中将 `default_ip` 设置为前端 IP。*

### 2. 启动 UE (用户端)
```bash
sudo -s
cd build
# 对于 X310:
cp ../config/Demodulator_X310.yaml Demodulator.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./OFDMDemodulator

# 对于 B210:
cp ../config/Demodulator_B210.yaml Demodulator.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./OFDMDemodulator
```
*如果您使用单独的计算机作为前端，请在 `Demodulator.yaml` 中将 `default_ip` 设置为前端 IP。*

### 3. 将视频流传输到 BS
```bash
ffmpeg -re -stream_loop -1 -fflags +genpts -i video.mp4 -an -c:v libx264 -x264-params keyint=5:min-keyint=1 -b:v 3000k -minrate 3000k -maxrate 3000k -bufsize 1M -f rtp -sdp_file video.sdp "rtp://<your IP of the BS>:50000"
```
*如果您在BS本地传输视频流，BS 的 IP 可以设置为 127.0.0.1。*

如果您希望连同音频一起传输，可以改用 RTP MPEG-TS：
```bash
ffmpeg -re -stream_loop -1 -fflags +genpts -i video.mp4 -c:v libx264 -x264-params keyint=5:min-keyint=1 -b:v 30000k -minrate 30000k -maxrate 30000k -bufsize 1M -c:a aac -b:a 128k -ar 48000 -ac 2 -f rtp_mpegts "rtp://<your IP of the BS>:50000"
```

### 4. 从 UE 播放视频
将 `video.sdp` 复制到视频接收端，将 `m=video 50000 RTP/AVP 96` 修改为 `m=video 50001 RTP/AVP 96`。
```bash
ffplay -protocol_whitelist file,rtp,udp -i video1.sdp
```
*注意：此命令应在前端运行。*

对于带音频的 RTP MPEG-TS 流，则可以不依赖 SDP 文件，直接播放：
```bash
ffplay rtp://0.0.0.0:50001
```

### 5. 运行单站感知前端
```bash
python3 ./scripts/plot_sensing_fast.py
```
这是当前维护的统一前端，会自动选择 CUDA、MLX、Intel GPU 或 CPU 后端。

### 6. 运行双站感知前端
```bash
python3 ./scripts/plot_bi_sensing_fast.py
```
这是当前维护的统一前端，会自动选择 CUDA、MLX、Intel GPU 或 CPU 后端。

### 7. 网页配置控制台
如果希望通过浏览器远程修改配置并控制进程，可运行：
```bash
python3 scripts/config_web_editor.py --host 0.0.0.0 --port 8765
```

随后在浏览器打开 `http://<your-host>:8765`。

功能：
* 调制端和解调端使用不同 tab 分开管理，并额外提供 `Resource Planner` 和 `Sensing Resource Map` 两个规划 tab，分别对应 `data_resource_blocks` 与 `sensing_mask_blocks`。
* 以“参数 / 值”表单方式编辑 `build/Modulator.yaml` 和 `build/Demodulator.yaml`，而不是原始 YAML 文本框。
* 为 `cpu_cores` 提供专门的 CPU 绑定编辑器，可按线程名填写 CPU，并自动生成对应注释。
* 保存当前表单后，可在 `build/` 目录中启动/停止调制与解调进程。
* 提供启动相关选项，例如是否启用 CPU 隔离、以及是否覆盖默认的 isolate CPU 列表。
* 每个 tab 都提供 CPU/CUDA 预设命令，也支持自定义启动命令。
* 可以在较大的时频资源网格画布上直接绘制 `data_resource_blocks` 的 payload / sensing-pilot 矩形块，或绘制 `sensing_mask_blocks` 的紧凑感知矩形块；绘制结果会吸附到整数 RE 格点边界，并可分别应用到发射端或接收端 YAML。
* 内置 `Guard Band Grid` 预设，规则与 `scripts/plot_const.py` 一致：默认仅保留 `1..489` 和 `535..N-1` 这两段子载波，然后再继续套用 sync / pilot 的剔除规则。

说明：
* 默认命令分别是 `./OFDMModulator` 和 `./OFDMDemodulator`；如果需要 CUDA 版本，可在下拉框里切换。
* 编辑器当前直接面向 `build/` 目录中的运行时 YAML，因为二进制程序会从各自工作目录读取 `Modulator.yaml` / `Demodulator.yaml`。
* `Resource Planner` 用来编辑 `data_resource_blocks`：它决定哪些 RE 承载 payload，哪些 RE 作为 `sensing_pilot` 保留给感知参考。
* `Sensing Resource Map` 用来编辑 `sensing_mask_blocks`：它决定 `sensing_output_mode=compact_mask` 时哪些 RE 会被送到感知输出。
* 两个 planner 都可以分别应用到发射端或接收端。实验时 TX 和 RX 可以暂时不同，但正常收发时 `data_resource_blocks` 仍应保持一致。
* 当 CPU 核心不足时，建议先给 `main thread affinity` 预留一个专用核心，然后优先保证 TX/RX 线程，最后再保证调制/解调线程和感知/信号处理线程；这些计算线程通常对应更大的缓冲区，对瞬时抖动更耐受。
* 若开启 `Enable runtime CPU isolation`，控制台会根据当前 `cpu_cores` 计算默认 isolate CPU 列表，并在启动前调用 `scripts/isolate_cpus.bash`。
* 若再开启 `Override CPU isolation list`，右侧文本框会先用默认 isolate 列表初始化，然后允许按本次启动需要手工修改。
* 若关闭 `Enable runtime CPU isolation`，控制台仍会通过特权运行路径启动选中的命令，但不会调用 `scripts/isolate_cpus.bash`。
* 运行面板还提供可选的 sudo 密码输入框，以及 `Reset CPU isolation` 操作。
* 该控制台可以执行网页中输入的启动命令，因此只应绑定到可信网络，或者保持默认 `127.0.0.1`。

## 参数说明

### OFDM 调制器 (OFDMModulator)

`OFDMModulator` (BS 节点) 使用 `Modulator.yaml` 配置。
可使用 `config/Modulator_X310.yaml` 或 `config/Modulator_B210.yaml` 作为模板。

`Modulator.yaml` 参数说明：

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `fft_size` | `int` | `1024` | OFDM FFT 点数。 |
| `cp_length` | `int` | `128` | 循环前缀长度（采样点）。 |
| `sync_pos` | `int` | `1` | 同步符号在帧内的位置索引。 |
| `sample_rate` | `float` / Hz | `50000000` | 基带采样率。 |
| `bandwidth` | `float` / Hz | `50000000` | 模拟带宽。通常与 `sample_rate` 保持一致。 |
| `center_freq` | `float` / Hz | `2400000000` | 射频中心频率。 |
| `tx_gain` | `float` / dB | `30` | 发射增益。 |
| `tx_channel` | `int` | `0` | TX 通道索引。 |
| `zc_root` | `int` | `29` | Zadoff-Chu 根序号。 |
| `num_symbols` | `int` | `100` | 每帧 OFDM 符号数。 |
| `sensing_output_mode` | `string` | `dense` | 感知 UDP 输出模式。`dense` 保持旧版基于 STRD 的全缓冲区输出；`compact_mask` 切换为按帧提取紧凑感知 RE。 |
| `cuda_mod_pipeline_slots` | `int` | `2` | CUDA 调制流水线 slot 数。小于 `1` 时会钳制到 `1`。 |
| `pilot_positions` | `int[]` | `[571,...,451]` | 导频子载波索引列表。 |
| `data_resource_blocks` | `object[]` | 缺省 | 可选的通信资源映射，用来回答“哪些 RE 用来放业务数据”。省略该键时保持旧行为：除同步和导频外的所有 RE 都承载 payload。设为 `[]` 表示完全不发送 payload。每个块是一个矩形，使用 `symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`，并可选 `kind`。`kind: payload` 表示这些 RE 承载真实业务数据；`kind: sensing_pilot` 表示这些 RE 不承载 payload，而是发送确定性的感知参考序列，便于感知侧把这些 RE 当作已知参考。该感知参考序列使用一个不同于帧同步符号的备选 Zadoff-Chu 根生成，避免把 `sensing_pilot` 误判成专用同步符号。未被 `payload` 块选中的其余非同步、非导频 RE 会发送预生成 QPSK。 |
| `sensing_mask_blocks` | `object[]` | 缺省 | 可选的紧凑感知资源映射，用来回答“compact 感知时哪些 RE 要导出”。仅在 `sensing_output_mode=compact_mask` 时生效；`dense` 模式下会忽略。每个块也是矩形，坐标使用绝对帧符号索引和原始 FFT bin 索引。这里允许覆盖同步 / 导频 RE，重叠块会自动并集，输出顺序固定为“先符号、后子载波”。如果每个被选中的符号都使用相同的子载波集合，且这些符号在环形帧轴上等间隔，那么运行时 `MTI` 和本地 Delay-Doppler 处理也可以开启。 |
| `device_args` | `string` | `""` | 通用 USRP 参数（TX/RX 兜底）。 |
| `tx_device_args` | `string` | `""` | TX 专用 USRP 参数。 |
| `rx_device_args` | `string` | `""` | 感知 RX 默认 USRP 参数。 |
| `clock_source` | `string` | `internal/external/gpsdo` | 全局时钟源。 |
| `time_source` | `string` | `""` | 全局时间/PPS 源；空字符串表示跟随 `clock_source`。 |
| `tx_clock_source` | `string` | `""` | TX 时钟源覆盖项。 |
| `tx_time_source` | `string` | `""` | TX 时间源覆盖项。 |
| `rx_clock_source` | `string` | `""` | 感知 RX 默认时钟源覆盖项。 |
| `rx_time_source` | `string` | `""` | 感知 RX 默认时间源覆盖项。 |
| `wire_format_tx` | `string` | `sc16` | TX 链路数据格式，常用 `sc16` 或 `sc8`。 |
| `wire_format_rx` | `string` | `sc16` | RX 链路数据格式，常用 `sc16` 或 `sc8`。 |
| `udp_input_ip` | `string` / IPv4 | `0.0.0.0` | 调制器接收业务 UDP 的绑定地址。 |
| `udp_input_port` | `int` | `50000` | 调制器接收业务 UDP 的端口。 |
| `mono_sensing_ip` | `string` / IPv4 | `127.0.0.1` | 单站感知输出目标 IP。 |
| `mono_sensing_port` | `int` | `8888` | 单站感知输出目标端口。 |
| `sensing_rx_channel_count` | `int` | `1` | 感知 RX 通道数量（`0` 表示关闭感知 RX）。 |
| `sensing_rx_channels` | `object[]` | `[]` | 感知 RX 每通道详细配置，字段见下表。 |
| `default_ip` | `string` / IPv4 | `127.0.0.1` | 默认目标 IP；未单独配置的输出 IP 使用该值。 |
| `control_port` | `int` | `9999` | 控制命令 UDP 端口（心跳/MTI 等）。 |
| `measurement_enable` | `bool` | `false` | 启用 CPU 版内部测量模式。启用后，`OFDMModulator` 不再监听 `udp_input_*`，而是内部生成确定性的 PRBS 载荷；`OFDMDemodulator` 会把测量载荷转入 BER/BLER/EVM 统计。CUDA 二进制忽略该模式。 |
| `measurement_mode` | `string` | `internal_prbs` | 测量模式选择。目前仅支持 `internal_prbs`；非法值会在配置归一化阶段自动关闭测量模式。 |
| `measurement_run_id` | `string` | `""` | 写入测量 CSV 汇总的运行 ID。 |
| `measurement_output_dir` | `string` | `""` | CPU 测量汇总文件输出目录。 |
| `measurement_payload_bytes` | `int` | `1024` | 每个内部测量载荷的字节数。若小于内部测量头长度，会自动钳制到最小合法值。 |
| `measurement_prbs_seed` | `int` | `0x5A` | 用于生成确定性 PRBS 载荷内容的基础种子。 |
| `measurement_packets_per_point` | `int` | `1` | 每个在线 `MRST` epoch 要发送的测量载荷数。小于 `1` 时会钳制到 `1`。 |
| `profiling_modules` | `string` | `""` | 性能统计模块列表，逗号分隔。常用值包括 `modulation`、`latency`、`data_ingest`、`sensing_proc`、`sensing_process`；`all` 表示全部。调制器端到端时延统计只有在同时包含 `modulation` 和 `latency` 时才启用。 |
| `cpu_cores` | `int[]` | `[0,1,2,3,4,5]` | 允许使用的 CPU 核列表。建议按 TX 线程、调制线程、数据输入线程、每个已启用感知通道的 RX/感知线程，以及主线程来预留。若核心数量有限，应先给主线程保留一个专用核心，其次优先 TX 和感知 RX 线程，最后再考虑调制、数据输入和感知处理线程，因为后者有更深的缓冲区。 |

快速理解：
* `data_resource_blocks` 决定“通信数据放在哪里”。
* `sensing_mask_blocks` 决定“compact 感知时哪些 RE 要导出”。
* 前者影响 payload 映射，后者只影响感知输出，两者不是互相替代的关系。

若启用 `data_resource_blocks`，请把相同的矩形块和 `kind` 同步写入 `Demodulator.yaml`。如果与 `sync_pos` 或 `pilot_positions` 重叠，内置的同步 / 导频 RE 仍然优先。优先级始终是“同步符号 > 导频 > sensing_pilot > payload/预生成 QPSK”。

当 `sensing_output_mode=compact_mask` 时，感知会变成“每个 OFDM 帧发送一个紧凑 UDP 包”，其中只包含 `sensing_mask_blocks` 选中的 RE。此时 `STRD` 会被忽略，因为采样图样已经由 mask 本身决定。若这个 mask 是“规则”的，也就是每个被选中的符号都使用相同的子载波集合，且这些符号在环形帧轴上等间隔，那么运行时 `MTI` 和本地 Delay-Doppler 处理也可以开启：`SKIP=1` 保持输出紧凑原始 RE，`SKIP=0` 切回基于该规则采样生成的 dense Delay-Doppler 输出。配置归一化还会按需要自动扩展 `range_fft_size` 和 `doppler_fft_size`，确保它们能覆盖所选子载波数和符号数。紧凑感知 UDP 载荷格式为 `CompactSensingFrameHeader { magic/version, mask_hash, re_count, frame_start_symbol_index }`，后面跟着固定顺序的 `re_count` 个原始 `complex<float>` 数据。当前 `plot_sensing*.py` 还不能处理非“规则”的 compact 载荷。

`sensing_rx_channels` 子项字段：

| 字段 | 类型 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `usrp_channel` | `int` | `0` | 对应 USRP RX 通道号。 |
| `device_args` | `string` | `""` | 该通道专用 USRP 参数。 |
| `clock_source` | `string` | `""` | 该通道专用时钟源覆盖。 |
| `time_source` | `string` | `""` | 该通道专用时间源覆盖。 |
| `wire_format_rx` | `string` | `""` | 该通道专用 RX 数据格式覆盖。 |
| `rx_gain` | `float` | `30` | 该通道 RX 增益。 |
| `alignment` | `int` | `63` | 该通道对齐偏移（采样点）。 |
| `rx_antenna` | `string` | `""` | 该通道天线口，如 `TX/RX`、`RX1`。 |
| `enable_system_delay_estimation` | `bool` | `false` | 若为 `true`，该通道会在启动时执行一次基于 ZC 相关的系统时延估计，之后每隔 434 个帧再执行一次；同时继续消耗帧数据，但保持常规感知处理链停用。 |

说明：
* 当 `sensing_rx_channels` 为空且 `sensing_rx_channel_count > 0` 时，程序按通道号 `0..N-1` 自动补齐默认项。
* 若两者数量不一致，程序会按 `sensing_rx_channel_count` 对通道列表做裁剪或扩展。
* 当某通道设置 `enable_system_delay_estimation=true` 时，该通道会在启动附近执行一次系统时延估计，之后每隔 434 个帧重复执行一次，同时继续消耗帧数据；常规感知处理和感知 UDP 输出保持停用。
* `device_args`、`wire_format_*`、每通道天线口和输出 IP 等字段通常与硬件平台和部署环境强相关；样本 YAML 只是起点，不应把不同机器/射频平台的值机械互换。

### OFDM 解调器 (OFDMDemodulator)

`OFDMDemodulator` (UE 节点) 使用 `Demodulator.yaml` 配置。
可使用 `config/Demodulator_X310.yaml` 或 `config/Demodulator_B210.yaml` 作为模板。

`Demodulator.yaml` 参数说明：

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `fft_size` | `int` | `1024` | OFDM FFT 点数。 |
| `cp_length` | `int` | `128` | 循环前缀长度（采样点）。 |
| `sync_pos` | `int` | `1` | 同步符号在帧内的位置索引。 |
| `sample_rate` | `float` / Hz | `50000000` | 基带采样率。 |
| `bandwidth` | `float` / Hz | `50000000` | 模拟带宽。通常与 `sample_rate` 保持一致。 |
| `center_freq` | `float` / Hz | `2400000000` | 射频中心频率。 |
| `rx_gain` | `float` / dB | `50` | 接收增益。 |
| `rx_agc_enable` | `bool` | `false` | 启用硬件 RX AGC。跟踪阶段 AGC 基于滤波后的 `delay_spectrum` 主峰调节 USRP 的 RX 增益，并使用与对齐/频偏调整相同的时间戳门控丢弃旧帧 AGC 动作；若同步符号接近 ADC 满幅，还会强制降增益。 |
| `rx_agc_low_threshold_db` | `float` / dB | `11.0` | 跟踪阶段 AGC 窗口下界。只有当滤波后的 `delay_spectrum` 主峰低于该阈值时，才提高 RX 增益。 |
| `rx_agc_high_threshold_db` | `float` / dB | `13.0` | 跟踪阶段 AGC 窗口上界。只有当滤波后的 `delay_spectrum` 主峰高于该阈值时，才降低 RX 增益。 |
| `rx_agc_max_step_db` | `float` / dB | `3.0` | 单次 AGC 更新允许的最大 RX 增益步进。饱和保护强制降增益时也使用这个步进上限。 |
| `rx_agc_update_frames` | `int` | `4` | 跟踪阶段两次 AGC 更新之间最少处理的帧数。小于 `1` 时会钳制到 `1`。 |
| `rx_channel` | `int` | `0` | RX 通道索引。 |
| `zc_root` | `int` | `29` | Zadoff-Chu 根序号。 |
| `num_symbols` | `int` | `100` | 每帧 OFDM 符号数。 |
| `sensing_symbol_num` | `int` | `100` | 参与感知处理的符号数。 |
| `sensing_output_mode` | `string` | `dense` | 双站感知 UDP 输出模式。`dense` 保持旧版基于 STRD 的全缓冲区输出；`compact_mask` 切换为按帧提取紧凑感知 RE。 |
| `cuda_demod_pipeline_slots` | `int` | `3` | CUDA 解调流水线 slot 数。小于 `1` 时会钳制到 `1`。 |
| `frame_queue_size` | `int` | `8` | 解调器 RX 帧队列容量。小于 `1` 时会钳制到 `1`。 |
| `sync_queue_size` | `int` | `8` | 同步搜索批队列容量。小于 `1` 时会钳制到 `1`。 |
| `reset_hold_s` | `float` / s | `0.5` | 在强制回到同步搜索前，坏的 delay 条件必须持续累积的时间。内部会按 `samples_per_frame / sample_rate` 换算成帧数阈值。小于 `0` 时会钳制到 `0.5`。 |
| `range_fft_size` | `int` | `1024` | 距离向 FFT 点数。 |
| `doppler_fft_size` | `int` | `100` | 多普勒向 FFT 点数。 |
| `pilot_positions` | `int[]` | `[571,...,451]` | 导频子载波索引列表。 |
| `data_resource_blocks` | `object[]` | 缺省 | 接收侧的通信资源映射，用来回答“哪些 RE 应该被当作 payload 来解调”。省略该键时保持旧行为：除同步和导频外的所有 RE 都参与 payload 提取。设为 `[]` 表示完全不提取 payload LLR。应与发射端使用相同的矩形块和 `kind`。其中 `kind: payload` 的块会产生 payload LLR，`kind: sensing_pilot` 的块则会被当作已知参考 RE，不参与 payload 提取。该已知参考序列与发射端保持一致，也使用不同于帧同步符号的备选 Zadoff-Chu 根。 |
| `sensing_mask_blocks` | `object[]` | 缺省 | 接收侧的紧凑感知资源映射，用来回答“在 `compact_mask` 模式下，双站感知要导出哪些 RE”。坐标系和行为与发射端一致：使用绝对帧符号索引和原始 FFT bin 索引，允许覆盖同步 / 导频 RE，重叠块自动并集，输出顺序固定为“先符号、后子载波”。如果 mask 满足规则采样条件，同样可以开启运行时 `MTI` 和本地 Delay-Doppler 处理。 |
| `device_args` | `string` | `""` | USRP 参数。 |
| `clock_source` | `string` | `internal/external/gpsdo` | 时钟源。 |
| `wire_format_rx` | `string` | `sc16` | RX 链路数据格式，常用 `sc16` 或 `sc8`。 |
| `software_sync` | `bool` | `true` | 启用软件同步跟踪。 |
| `predictive_delay` | `bool` | `true` | 启用基于 CFO 的预测性时延补偿，用于初始对齐和跟踪阶段的时延修正。仅当采样时钟和载波频率来自同一个参考源，且信号链路在 USRP 外没有二级变频时才应启用。 |
| `hardware_sync` | `bool` | `false` | 启用硬件同步。 |
| `hardware_sync_tty` | `string` | `/dev/ttyUSB0` | 硬件同步控制串口设备。 |
| `ocxo_pi_switch_abs_error_ppm` | `float` | `0.0002` | 当 `error_ppm` 绝对值持续低于该阈值时切换到慢速 OCXO PI 阶段。 |
| `akf_enable` | `bool` | `true` | 启用硬件同步 `error_ppm` 的自适应卡尔曼滤波（AKF）。 |
| `akf_bootstrap_frames` | `int` | `64` | AKF 进入正常 KF 更新前的冷启动帧数。 |
| `akf_innovation_window` | `int` | `64` | 用于新息自相关/最小二乘自适应的窗口长度。 |
| `akf_max_lag` | `int` | `4` | 最小二乘拟合使用的新息自相关最大滞后阶。 |
| `akf_adapt_interval` | `int` | `64` | 自适应更新 `Q/R` 的帧间隔。 |
| `akf_gate_sigma` | `float` | `3.0` | 新息门限系数（sigma）。 |
| `akf_tikhonov_lambda` | `float` | `1e-3` | 最小二乘自适应的 Tikhonov 正则项权重。 |
| `akf_update_smooth` | `float` | `0.2` | `Q/R` 更新的指数平滑系数。 |
| `akf_q_wf_min` | `float` | `1e-10` | 白频率噪声系数下界。 |
| `akf_q_wf_max` | `float` | `1e2` | 白频率噪声系数上界。 |
| `akf_q_rw_min` | `float` | `1e-12` | 随机游走频率噪声系数下界。 |
| `akf_q_rw_max` | `float` | `1e1` | 随机游走频率噪声系数上界。 |
| `akf_r_min` | `float` | `1e-8` | 观测噪声方差 `R` 下界。 |
| `akf_r_max` | `float` | `1e3` | 观测噪声方差 `R` 上界。 |
| `ppm_adjust_factor` | `float` | `0.05` | 频偏补偿调节系数。 |
| `desired_peak_pos` | `int` | `20` | 时延峰目标位置（用于对齐策略）。 |
| `enable_bi_sensing` | `bool` | `true` | 启用双站感知处理链和 UDP 输出；设为 `false` 时 `OFDMDemodulator` 与 `CUDAOFDMDemodulator` 均不会启动双站感知通道。 |
| `bi_sensing_ip` | `string` / IPv4 | `127.0.0.1` | 双站感知输出目标 IP。 |
| `bi_sensing_port` | `int` | `8889` | 双站感知输出目标端口。 |
| `channel_ip` | `string` / IPv4 | `127.0.0.1` | 信道估计输出 IP。 |
| `channel_port` | `int` | `12348` | 信道估计输出端口。 |
| `pdf_ip` | `string` / IPv4 | `127.0.0.1` | PDP/PDF 输出 IP。 |
| `pdf_port` | `int` | `12349` | PDP/PDF 输出端口。 |
| `constellation_ip` | `string` / IPv4 | `127.0.0.1` | 星座图输出 IP。 |
| `constellation_port` | `int` | `12346` | 星座图输出端口。 |
| `vofa_debug_ip` | `string` / IPv4 | `127.0.0.1` | VOFA+ 调试输出 IP。 |
| `vofa_debug_port` | `int` | `12347` | VOFA+ 调试输出端口。 |
| `udp_output_ip` | `string` / IPv4 | `127.0.0.1` | 解码后业务数据输出 IP。 |
| `udp_output_port` | `int` | `50001` | 解码后业务数据输出端口。 |
| `default_ip` | `string` / IPv4 | `127.0.0.1` | 默认输出 IP；其他输出 IP 留空时使用该值。 |
| `control_port` | `int` | `9999` | 控制命令 UDP 端口。 |
| `measurement_enable` | `bool` | `false` | 启用 CPU 版内部测量模式。启用后，测量载荷不会再转发到 `udp_output_*`，而是直接用于 BER/BLER/EVM 统计。CUDA 二进制忽略该模式。 |
| `measurement_mode` | `string` | `internal_prbs` | 测量模式选择。目前仅支持 `internal_prbs`；非法值会在配置归一化阶段自动关闭测量模式。 |
| `measurement_run_id` | `string` | `""` | 写入测量 CSV 汇总的运行 ID。 |
| `measurement_output_dir` | `string` | `""` | CPU 测量汇总文件输出目录。 |
| `measurement_payload_bytes` | `int` | `1024` | 每个测量载荷期望的字节数。若小于内部测量头长度，会自动钳制到最小合法值。 |
| `measurement_prbs_seed` | `int` | `0x5A` | 用于重建确定性 PRBS 测量载荷的基础种子。 |
| `measurement_packets_per_point` | `int` | `1` | 每个在线 `MRST` epoch 期望统计的测量载荷数。小于 `1` 时会钳制到 `1`。 |
| `profiling_modules` | `string` | `""` | 性能统计模块列表，逗号分隔。常用值包括 `demodulation`、`agc`、`align`、`snr`；`all` 表示全部。`agc` 控制 AGC 日志，`align` 控制运行时 `ALGN:` 日志，`snr` 会周期性打印当前解调路径的 `_snr_db / _noise_var / _llr_scale`。 |
| `cpu_cores` | `int[]` | `[0,1,2,3,4,5]` | 允许使用的 CPU 核列表。若核心数量有限，应先给主线程保留一个专用核心，其次优先 `rx_proc`，最后再考虑 `process_proc`、`sensing_process_proc` 和 `bit_processing_proc`，因为这些后级处理线程通常有更大的缓冲区，能更好吸收短时调度抖动。 |

说明：
* `data_resource_blocks` 通常应与发射端完全一致，包括 `kind`。
* 如果资源块与 `sync_pos` 或 `pilot_positions` 重叠，内置的同步 / 导频 RE 仍然优先。优先级是“同步符号 > 导频 > sensing_pilot > payload/预生成 QPSK”。
* 当 `sensing_output_mode=compact_mask` 时，双站感知同样会变成“每个 OFDM 帧发送一个紧凑 UDP 包”，只包含 `sensing_mask_blocks` 选中的 RE；此时 `STRD` 会被忽略，因为 mask 已经定义了采样图样。
* 紧凑载荷格式与发射端一致：`CompactSensingFrameHeader` 后面跟固定顺序的原始 `complex<float>` 数据。
* RX AGC 分为两个阶段。`SYNC_SEARCH` 阶段会先把增益恢复到配置的 `rx_gain`，然后进行粗搜索扫描（每 10 个帧增加 `1 dB`，达到最大增益后回绕到最小增益）；锁定后进入跟踪阶段，使用 `rx_agc_low_threshold_db` / `rx_agc_high_threshold_db` 定义的窗口来细调增益。
* 解调器还会检查同步符号时域样本的 I/Q 分量是否接近或达到 ADC 满幅。如果满幅点数过多，会立即强制降低增益，并在短时间内禁止再次升增益，避免在噪声或削顶附近来回摆动。
* 硬 reset 会清空时延/频偏跟踪状态、刷新队列、重置跟踪 AGC 状态，并回到 `SYNC_SEARCH`。`reset_hold_s` 决定坏的 delay 条件需要持续多久才会触发这一动作。
