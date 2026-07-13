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
| BS 后端 | `BS`、`config/BS_*.yaml` | 发送 OFDM 帧、接收业务 UDP，并输出单站感知结果 |
| UE 后端 | `UE`、`config/UE_*.yaml` | 接收解调 OFDM 帧、输出业务数据，并运行双站感知 |
| 前端工具 | `scripts/plot_*.py`、`scripts/config_web_editor.py` | 显示感知/信道结果，并编辑运行时配置 |

## 快速导航

- 环境准备与安装: [硬件准备](#硬件准备)、[软件安装](#软件安装)
- 先跑起来: [典型使用示例](#典型使用示例)
- 运行参数说明: [BS](#bs)、[UE](#ue)
- 网页控制台: [网页配置控制台](#8-网页配置控制台)
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
| 运行 BS 端 | `BS` | `config/BS_X310.yaml` 或 `config/BS_B210.yaml` | `plot_sensing_fast.py` |
| 运行 UE 端 | `UE` | `config/UE_X310.yaml` 或 `config/UE_B210.yaml` | `plot_bi_sensing_fast.py` |
| 用浏览器调参数 | `scripts/config_web_editor.py` | 读取 `build/BS.yaml` 和 `build/UE.yaml` | 浏览器访问 `http://<host>:8765` |

## 首次 OTA 运行前检查

- 准备两台后端节点。下行-only 模式下，BS 端需要一路 TX 和一路感知 RX 天线链路，UE 端需要一路 RX 链路。若开启双工/上行，UE 端还必须具备一路 TX 天线/RF 链路，BS 端也必须具备用于接收上行的一路 RX 天线/RF 链路；FDD 模式还要求射频设备支持配置的上行载波，并具备足够的收发隔离。
- 先选一份最接近你硬件的 YAML 模板。仓库自带 X310/B210 示例，但项目并不限于这两种 USRP。
- 运行时 YAML 需要放在 `build/` 目录，因为两个二进制程序都会从当前工作目录读取 `BS.yaml` 或 `UE.yaml`。
- 如果前端跑在另一台机器上，请在 Python 前端中用 `--host <后端IP>` 或 Backend IP 输入框指向后端。`default_out_ip` 只用于目的输出 IP，不用于 UDP/ZMQ 监听地址。
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
*   **天线**: 下行-only 运行需要 3 根；若开启双工/上行且 UE TX 使用独立天线口，通常需要 4 根
*   **OCXO/GPSDO**: 2 个 (两台 USRP 都需要)
 
#### 连接设置
 
该系统由两个主要节点组成：
 
1.  **基站 (BS) 节点**
    *   **硬件**: 1x 计算机, 1x USRP。
    *   **天线**: 连接 2 根天线到此 USRP (1 路下行 TX，1 路感知/上行 RX)。
    *   **时钟**: 将 OCXO 或 GPSDO 连接到 USRP 的 REFIN 端口。
    *   **功能**: 发送 OFDM 信号并接收雷达回波。
 
2.  **用户 (UE) 节点**
    *   **硬件**: 1x 计算机, 1x USRP。
    *   **天线**: 下行-only 运行时连接 1 根天线到此 USRP 的 RX 端口。若开启双工/上行，还需要连接 UE TX 天线/RF 链路；FDD 模式还要求设备支持配置的上行载波并具备足够收发隔离。
    *   **时钟**: 将 OCXO 或 GPSDO 连接到 USRP 的 REFIN 端口。
    *   **高精度 DAC (可选)**: 使用高精度 DAC 来微调 OCXO。
    *   **功能**: 接收 OFDM 信号用于通信和双站感知；开启双工/上行后还会发送 UE->BS 上行业务。
 
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
    *   教程: [独立 macOS 构建教程](https://github.com/zhouzhiwen2000/OpenISAC/blob/main/docs/macos_build_zh.md)
 
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
# 安装 libyaml-cpp-dev 与 ZeroMQ（libzmq + cppzmq 头文件）
sudo apt-get install libyaml-cpp-dev libzmq3-dev cppzmq-dev

cd OpenISAC
mkdir build
cd build
cmake ..
make -j$(nproc)
```

> 后端↔前端通信（感知数据流 + 控制/参数通道）现已基于 **ZeroMQ**。后端按配置的监听 IP/端口为感知/调试数据流绑定 PUB socket、为控制通道绑定 ROUTER socket；示例 YAML 使用 `0.0.0.0` 作为感知/调试/控制监听 IP。Python 前端用 SUB/DEALER socket 连接。让某个前端连接远端后端时，使用其 `--host <ip>` 参数或 Backend IP 输入框（默认 `127.0.0.1`）。
 
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
sudo ../scripts/isolate_cpus.bash run ./BS
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

*   **配置文件名**: `BS` 读取 `BS.yaml`，`UE` 读取 `UE.yaml`。
*   **首次运行**: 模板 YAML 统一放在 `config/` 目录。请将 `config/BS_X310.yaml` / `config/BS_B210.yaml` 或
    `config/UE_X310.yaml` / `config/UE_B210.yaml` 复制为 `BS.yaml` / `UE.yaml`，再按需修改。
    B210 TDD 双工预设可直接复制 `config/BS_B210_Duplex.yaml` 和 `config/UE_B210_Duplex.yaml`。
 
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
cp ../config/BS_X310.yaml BS.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./BS

# 对于 B210:
cp ../config/BS_B210.yaml BS.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./BS

# 对于 B210 双工:
cp ../config/BS_B210_Duplex.yaml BS.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./BS
```
*如果您使用单独的计算机作为前端，请在单站感知前端中用 `--host` 或 Backend IP 输入框指向 BS 后端 IP。*

### 2. 启动 UE (用户端)
```bash
sudo -s
cd build
# 对于 X310:
cp ../config/UE_X310.yaml UE.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./UE

# 对于 B210:
cp ../config/UE_B210.yaml UE.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./UE

# 对于 B210 双工:
cp ../config/UE_B210_Duplex.yaml UE.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./UE
```
*如果您使用单独的计算机作为前端，请在双站感知前端中用 `--host` 或 Backend IP 输入框指向 UE 后端 IP。只有解码/调试类输出需要发往另一台机器时才设置 `default_out_ip`。*

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

### 7. 校准感知通道系统响应

校准前请先进行射频直连：将发射 RF 输出连接到对应的感知 RX 输入，并在校准过程中保持连接稳定。

如果发射功率较大，建议优先降低发射功率；必要时在直连链路中加入合适的衰减器，避免 RX 饱和。衰减器本身应在信号带宽内尽量平坦，否则其带内起伏会被一并计入校准结果。

单站感知校准时，先启动 BS 后端和单站感知前端，在前端中切换到需要校准的感知通道，然后点击 `Calibrate Hsys`。如果是单站多通道配置，请分别对需要校准的通道执行一次。

双站感知校准时，先启动 BS 和 UE 两端后端，打开双站感知前端，并在射频直连状态下点击双站前端里的 `Calibrate Hsys`。

等待后端日志提示校准完成并保存校准文件。当前运行会立即使用新的系统响应校准结果；之后重新启动后端时，程序会自动加载匹配的校准文件。如果没有找到匹配文件，程序会继续运行并打印提示，感知处理会跳过系统响应补偿。

校准完成后，再恢复正常的天线连接或实验连接，然后继续进行空口测量。

### 8. 网页配置控制台
如果希望通过浏览器远程修改配置并控制进程，可运行：
```bash
python3 scripts/config_web_editor.py --host 0.0.0.0 --port 8765
```

随后在浏览器打开 `http://<your-host>:8765`。

功能：
* BS 和 UE 使用不同 tab 分开管理，并额外提供 `Resource Planner` 和 `Sensing Resource Map` 两个规划 tab，分别对应 `data_resource_blocks` 与 `mask_blocks`。
* 以“参数 / 值”表单方式编辑 `build/BS.yaml` 和 `build/UE.yaml`，而不是原始 YAML 文本框。
* 提供按模块放置的 CPU 绑核字段，覆盖下行、上行、感知实时 loop 和主线程模块。
* 保存当前表单后，可在 `build/` 目录中启动/停止 BS 与 UE 进程。
* 提供启动相关选项，例如是否启用 CPU 隔离、以及是否覆盖默认的 isolate CPU 列表。
* 每个 tab 都提供 CPU/CUDA 预设命令，也支持自定义启动命令。
* 可以在较大的时频资源网格画布上直接绘制 `data_resource_blocks` 的 payload / sensing-pilot 矩形块，或绘制 `mask_blocks` 的紧凑感知矩形块；绘制结果会吸附到整数 RE 格点边界，并可分别应用到发射端或接收端 YAML。
* 内置 `Guard Band Grid` 预设，规则与 `scripts/plot_const.py` 一致：默认仅保留 `1..489` 和 `535..N-1` 这两段子载波，然后再继续套用同步 / 梳状导频的剔除规则。

说明：
* 默认命令分别是 `./BS` 和 `./UE`；如果需要 CUDA 版本，可在下拉框里切换。
* 编辑器当前直接面向 `build/` 目录中的运行时 YAML，因为二进制程序会从各自工作目录读取 `BS.yaml` / `UE.yaml`。
* `Resource Planner` 用来编辑 `data_resource_blocks`：它决定哪些 RE 承载 payload，哪些 RE 作为 `sensing_pilot` 保留给感知参考。
* `Sensing Resource Map` 用来编辑 `mask_blocks`：它决定 `output_mode=compact_mask` 时哪些 RE 会被送到感知输出。
* 两个 planner 都可以分别应用到发射端或接收端。实验时 TX 和 RX 可以暂时不同，但正常收发时 `data_resource_blocks` 仍应保持一致。
* 当 CPU 核心不足时，建议先给 `main thread affinity` 预留一个专用核心，然后优先保证 TX/RX 线程，最后再保证调制/解调线程和感知/信号处理线程；这些计算线程通常对应更大的缓冲区，对瞬时抖动更耐受。
* CPU 绑核只配置实时流水线线程和主线程。非实时的服务、输出、辅助线程有意不做绑核。
* 使用 `-1`、`[]` 或省略可选字段表示该模块不做显式绑核。
* 若开启 `Enable runtime CPU isolation`，控制台会根据所有非负的实时 CPU 字段共同计算默认 isolate CPU 列表，并在启动前调用 `scripts/isolate_cpus.bash`。
* 若再开启 `Override CPU isolation list`，右侧文本框会先用默认 isolate 列表初始化，然后允许按本次启动需要手工修改。
* 若关闭 `Enable runtime CPU isolation`，控制台仍会通过特权运行路径启动选中的命令，但不会调用 `scripts/isolate_cpus.bash`。
* 运行面板还提供可选的 sudo 密码输入框，以及 `Reset CPU isolation` 操作。
* 该控制台可以执行网页中输入的启动命令，因此只应绑定到可信网络，或者保持默认 `127.0.0.1`。

## 参数说明

### BS

`BS` (BS 节点) 使用 `BS.yaml` 配置。
可使用 `config/BS_X310.yaml`、`config/BS_B210.yaml` 或 `config/BS_B210_Duplex.yaml` 作为模板。

`BS.yaml` 参数说明：

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `fft_size` | `int` | `1024` | OFDM FFT 点数。 |
| `cp_length` | `int` | `128` | 循环前缀长度（采样点）。 |
| `sync_pos` | `int` | `1` | 同步符号在帧内的位置索引。 |
| `enable_sec_sync_symbol` | `bool` | `false` | 预留 `sync_pos-1` 作为重复 ZC 第二同步符号。启用后，接收端会先用两个连续同步符号做类 Schmidl-Cox 粗定时/模糊 CFO 估计，再用 `sync_pos` 附近的局部 ZC 相关解析 CFO alias 并精修主同步。要求 `sync_pos >= 1`。 |
| `enable_cfo_training_sequence` | `bool` | `false` | 预留 `sync_pos+1` 作为专用重复 CFO training field。接收端仍先依赖 ZC 或可选第二同步符号完成帧定位和 modulo CFO 估计；启用该字段时，它只用于 CFO alias 去模糊。发射端和接收端必须保持一致。 |
| `cfo_training_period_samples` | `int` / 采样点 | `16` | CFO training field 的重复周期，必须整除 `fft_size`；无模糊 CFO 范围约为 `+-sample_rate/(2*period)`。 |
| `sample_rate` | `float` / Hz | `50000000` | 基带采样率。 |
| `bandwidth` | `float` / Hz | `50000000` | 模拟带宽。通常与 `sample_rate` 保持一致。 |
| `center_freq` | `float` / Hz | `2400000000` | 射频中心频率。 |
| `tx_gain` | `float` / dB | `30` | 发射增益。 |
| `tx_channel` | `int` | `0` | TX 通道索引。 |
| `zc_root` | `int` | `29` | Zadoff-Chu 根序号。 |
| `num_symbols` | `int` | `100` | 每帧 OFDM 符号数。 |
| `output_mode` | `string` | `dense` | 感知输出模式。`dense` 保持旧版基于 STRD 的全缓冲区输出；`compact_mask` 切换为按帧提取紧凑感知 RE。 |
| `cuda_mod_pipeline_slots` | `int` | `2` | CUDA 调制流水线 slot 数。小于 `1` 时会钳制到 `1`。 |
| `pilot_positions` | `int[]` | `[571,631,...,451]` | 分布在占用带宽内的可配置梳状导频子载波索引。 |
| `midframe_pilot_symbols` | `int[]` | `[]` | 可选的帧内 BPSK 导频符号索引，例如 `[25,50,75]`。这些符号不参与 payload 映射；配置的梳状导频 RE 会保留梳状导频序列用于相位跟踪，其余 RE 使用确定性 BPSK。 |
| `midframe_pilot_seed` | `int` | `1296453708` | 确定性帧内 BPSK 导频种子，`BS.yaml` 和 `UE.yaml` 必须一致。 |
| `data_resource_blocks` | `object[]` | 缺省 | 可选的通信资源映射，用来回答“哪些 RE 用来放业务数据”。省略该键时保持旧行为：除预留同步符号和梳状导频 RE 外的所有 RE 都承载 payload。设为 `[]` 表示完全不发送 payload。每个块是一个矩形，使用 `symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`，并可选 `kind`。`kind: payload` 表示这些 RE 承载真实业务数据；`kind: sensing_pilot` 表示这些 RE 不承载 payload，而是发送确定性的感知参考序列，便于感知侧把这些 RE 当作已知参考。该感知参考序列使用一个不同于帧同步符号的备选 Zadoff-Chu 根生成，避免把 `sensing_pilot` 误判成专用同步符号。未被 `payload` 块选中的其余非预留同步、非梳状导频、非帧内 BPSK 导频 RE 会发送预生成 QPSK。 |
| `mask_blocks` | `object[]` | 缺省 | 可选的紧凑感知资源映射，用来回答“compact 感知时哪些 RE 要导出”。仅在 `output_mode=compact_mask` 时生效；`dense` 模式下会忽略。每个块也是矩形，坐标使用绝对帧符号索引和原始 FFT bin 索引。这里允许选择 ZC 同步符号、梳状导频或帧内 BPSK 导频 RE；可选 CFO training field 会被拒绝，因为它不是合法感知符号。重叠块会自动并集，输出顺序固定为“先符号、后子载波”。如果每个被选中的符号都使用相同的子载波集合，且这些符号在环形帧轴上等间隔，那么运行时 `MTI` 和本地 Delay-Doppler 处理也可以开启。 |
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
| `rx_channel` | `int` | `0` | BS 上行 RX 使用的 USRP 通道索引。 |
| `rx_wire_format` | `string` | `sc16` | BS 上行 RX 链路数据格式，常用 `sc16` 或 `sc8`。 |
| `rx_wire_format` | `string` | `sc16` | BS 感知 RX 默认链路数据格式，常用 `sc16` 或 `sc8`。 |
| `udp_input_ip` | `string` / IPv4 | `0.0.0.0` | BS 下行业务 UDP 输入绑定地址，即 BS->UE 下行要发送的业务流。 |
| `udp_input_port` | `int` | `50000` | BS 下行业务 UDP 输入绑定端口。 |
| `udp_output_ip` | `string` / IPv4 | `127.0.0.1` | BS 解码后的上行业务 UDP 输出目标 IP，即从 UE->BS 上行恢复出的业务流。 |
| `udp_output_port` | `int` | `50003` | BS 解码后的上行业务 UDP 输出目标端口。 |
| `udp_egress_pacer_enabled` | `bool` | `false` | 启用解码后业务 UDP 输出的排队 pacing，用于平滑 decoder 造成的突发输出。 |
| `udp_egress_pacer_target_mbps` | `float` | `0` | UDP egress pacer 的目标载荷速率。`0` 表示根据入队速率自动估计；正数表示固定 Mbps 速率。 |
| `udp_egress_pacer_queue_packets` | `int` | `10240` | egress pacer 最多缓存的 UDP 数据报数量，超过后丢弃最旧数据报。 |
| `udp_egress_pacer_max_delay_ms` | `float` | `0` | egress pacer 中数据报允许排队的最长时间；设为 `0` 可关闭按年龄丢包。 |
| `duplex_mode` | `string` | `tdd` | 双工方式。`tdd` 将 UE 上行符号按时间复用到 BS 帧内，并使用下行中心频率；`fdd` 保持 BS 下行连续发送，同时 UE 上行使用 `uplink.center_freq`。 |
| `uplink` | `object` | `symbol_start=90`、`symbol_count=10`、`guard_symbols=1`、`center_freq=2500000000` | 上行/双工设置。TDD 下，`symbol_start`、`symbol_count`、`guard_symbols` 以 OFDM 符号为单位定义 DL/UL 边界，`center_freq` 会被忽略；FDD 下，`center_freq` 定义 UE->BS 上行载波，而 `symbol_start`、`symbol_count`、`guard_symbols` 会被忽略，上行按整帧连续传输。开启上行需要 UE 端具备 TX 天线/RF 链路，BS 端具备上行 RX 天线/RF 链路；FDD 还需要足够的频率间隔或收发隔离。 |
| `rx_agc_enable` | `bool` | `false` | 上行设置。启用 BS 上行接收端硬件 RX AGC。硬件 Duplex 模板默认开启；固定的 `uplink.rx_gain` 会作为初始增益。 |
| `rx_agc_low_threshold_db` | `float` / dB | `14.0` | 上行设置。上行跟踪 AGC 窗口下界。只有当滤波后的上行 delay-spectrum 主峰低于该阈值时，才提高 RX 增益。 |
| `rx_agc_high_threshold_db` | `float` / dB | `16.0` | 上行设置。上行跟踪 AGC 窗口上界。只有当滤波后的上行 delay-spectrum 主峰高于该阈值时，才降低 RX 增益。 |
| `rx_agc_max_step_db` | `float` / dB | `1.0` | 上行设置。BS 上行 RX 单次 AGC 更新允许的最大增益步进。饱和保护强制降增益时也使用这个步进上限。 |
| `rx_agc_update_frames` | `int` | `4` | 上行设置。上行跟踪 AGC 两次更新之间最少处理的上行帧数。小于 `1` 时会钳制到 `1`。 |
| `bs_dl_ul_timing_diff` | `int` / 采样点 | `63` | BS 侧上行 RX 窗口相对下行帧锚点的 DL/UL 定时差。启动时会按一帧长度做 modulo 规范化，也可运行时通过 `DUTI` 调整。 |
| `ertm_to_enable` | `bool` | `false` | 开启 CPU 路径 eRTM TO 估计。BS 将最新频域上行信道估计和运行时 `DUTI` 封装到内部下行 LDPC payload；UE 消费该 payload 并打印 centroid3 细化后的 TO 估计，不转发到用户 UDP。 |
| `ertm_report_interval_frames` | `int` / 帧 | `32` | BS eRTM payload/report 的下行 TX 帧间隔；小于 `1` 的值会按 `1` 处理。 |
| `mono_sensing_ip` | `string` / IPv4 | `0.0.0.0` | 单站感知数据流和控制通道的 ZMQ 监听 IP。使用 `0.0.0.0` 可接受远端前端连接；使用 `127.0.0.1` 则仅允许本机连接。 |
| `mono_sensing_port` | `int` | `8888` | 单站感知数据流的 ZeroMQ PUB 绑定端口。 |
| `uplink_channel_ip` | `string` / IPv4 | `0.0.0.0` | BS 上行信道估计调试流的 ZeroMQ PUB 监听 IP。 |
| `uplink_channel_port` | `int` | `12358` | BS 上行信道估计调试流的 ZeroMQ PUB 绑定端口。 |
| `uplink_pdf_ip` | `string` / IPv4 | `0.0.0.0` | BS 上行 delay profile 调试流的 ZeroMQ PUB 监听 IP。 |
| `uplink_pdf_port` | `int` | `12359` | BS 上行 delay profile 调试流的 ZeroMQ PUB 绑定端口。 |
| `uplink_constellation_ip` | `string` / IPv4 | `0.0.0.0` | BS 上行星座图调试流的 ZeroMQ PUB 监听 IP。 |
| `uplink_constellation_port` | `int` | `12356` | BS 上行星座图调试流的 ZeroMQ PUB 绑定端口。 |
| `rx_channel_count` | `int` | `1` | 感知 RX 通道数量（`0` 表示关闭感知 RX）。 |
| `rx_channels` | `object[]` | `[]` | 感知 RX 每通道详细配置，字段见下表。 |
| `tx_circular_buffer_size` | `int` | `32` | 向 TX 供帧的已调制帧队列容量。 |
| `paired_frame_queue_size` | `int` | `64` | 每个感知通道的 RX/TX 帧配对队列容量。建议大于 `tx_circular_buffer_size`，以便在 RX 启动、网络缓冲和对齐完成前保留 TX 参考帧。若启动完成后仍持续满队列，通常说明感知处理吞吐不足，而不是应该无限增加缓冲。 |
| `control_port` | `int` | `9999` | 双向控制通道的 ZeroMQ ROUTER 绑定端口（接收命令，回送参数/心跳）。 |
| `measurement_enable` | `bool` | `false` | 启用 CPU 版内部测量模式。启用后，`BS` 不再监听 `udp_input_*`，而是内部生成确定性的 PRBS 载荷；`UE` 会把测量载荷转入 BER/BLER/EVM 统计。CUDA 二进制忽略该模式。 |
| `measurement_mode` | `string` | `internal_prbs` | 测量模式选择。目前仅支持 `internal_prbs`；非法值会在配置归一化阶段自动关闭测量模式。 |
| `measurement_run_id` | `string` | `""` | 写入测量 CSV 汇总的运行 ID。 |
| `measurement_output_dir` | `string` | `""` | CPU 测量汇总文件输出目录。 |
| `measurement_payload_bytes` | `int` | `1024` | 每个内部测量载荷的字节数。若小于内部测量头长度，会自动钳制到最小合法值。 |
| `measurement_prbs_seed` | `int` | `0x5A` | 用于生成确定性 PRBS 载荷内容的基础种子。 |
| `measurement_packets_per_point` | `int` | `1` | 每个在线 `MRST` epoch 要发送的测量载荷数。小于 `1` 时会钳制到 `1`。 |
| `profiling_modules` | `string` | `""` | 性能统计模块列表，逗号分隔。常用值包括 `modulation`、`latency`、`ldpc_encode`、`sensing_proc`、`agc`、`uplink` 和 `ertm`；`all` 表示全部。BS 端到端时延统计只有在同时包含 `modulation` 和 `latency` 时才启用。 |
| `downlink_cpu_cores` | `int[]` | `[]` | BS 下行 CPU 核列表：索引 `0..3` 分别绑定 `_tx_proc`、`_modulation_proc`、`_ldpc_encode_proc` 和 `_udp_recv_proc`。 |
| `uplink_cpu_cores` | `int[]` | `[]` | BS 上行 CPU 核列表：索引 `0`、`1`、`2` 分别绑定 RX sample ingest、OFDM/LLR signal processing、LDPC decode + UDP output。 |
| `main_cpu_core` | `int` | `-1` | 主线程 CPU 核。 |

快速理解：
* `data_resource_blocks` 决定“通信数据放在哪里”。
* `mask_blocks` 决定“compact 感知时哪些 RE 要导出”。
* 前者影响 payload 映射，后者只影响感知输出，两者不是互相替代的关系。

若启用 `data_resource_blocks`，请把相同的矩形块和 `kind` 同步写入 `UE.yaml`。如果与 `sync_pos`、可选的 `sync_pos-1` 第二同步符号、`midframe_pilot_symbols` 或 `pilot_positions` 重叠，内置的 ZC 同步符号、梳状导频 RE 和帧内 BPSK 导频仍然优先。可选 `sync_pos+1` CFO training field 只服务于 CFO 捕获/去模糊，不是合法的 sensing-pilot 或 sensing-mask 符号。优先级始终是“ZC 同步符号 > CFO training field > 梳状导频 RE > 帧内 BPSK 导频 > sensing_pilot > payload/预生成 QPSK”。

dense 感知模式下，如果配置的 `symbol_stride` 或运行时 `STRD` 会采到可选的 `sync_pos+1` CFO training field，会被直接拒绝。运行时修改 `STRD` 会在计划好的帧边界重启确定性采样相位，不会继承旧 stride 的漂移相位。

当 `output_mode=compact_mask` 时，感知会变成“每个 OFDM 帧发送一条紧凑消息”，其中只包含 `mask_blocks` 选中的 RE。此时 `STRD` 会被忽略，因为采样图样已经由 mask 本身决定。若这个 mask 是“规则”的，也就是每个被选中的符号都使用相同的子载波集合，且这些符号在环形帧轴上等间隔，那么运行时 `MTI` 和本地 Delay-Doppler 处理也可以开启：`SKIP=1` 保持输出紧凑原始 RE，`SKIP=0` 切回基于该规则采样生成的 dense Delay-Doppler 输出。配置归一化还会按需要自动扩展 `range_fft_size` 和 `doppler_fft_size`，确保它们能覆盖所选子载波数和符号数。紧凑感知载荷格式为 `CompactSensingFrameHeader { magic/version, mask_hash, re_count, frame_start_symbol_index }`，后面跟着固定顺序的 `re_count` 个原始 `complex<float>` 数据。当前 `plot_sensing*.py` 还不能处理非“规则”的 compact 载荷。

`rx_channels` 子项字段：

| 字段 | 类型 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `usrp_channel` | `int` | `0` | 对应 USRP RX 通道号。 |
| `device_args` | `string` | `""` | 该通道专用 USRP 参数。 |
| `clock_source` | `string` | `""` | 该通道专用时钟源覆盖。 |
| `time_source` | `string` | `""` | 该通道专用时间源覆盖。 |
| `wire_format` | `string` | `""` | 该感知通道专用 RX 数据格式覆盖。 |
| `rx_gain` | `float` | `30` | 该通道 RX 增益。 |
| `alignment` | `int` | `63` | 该通道对齐偏移（采样点）。 |
| `rx_antenna` | `string` | `""` | 该通道天线口，如 `TX/RX`、`RX1`。 |
| `enable_system_delay_estimation` | `bool` | `false` | 若为 `true`，该通道会在启动时执行一次基于 ZC 相关的系统时延估计，之后每隔 434 个帧再执行一次；同时继续消耗帧数据，但保持常规感知处理链停用。 |
| `rx_cpu_core` | `int` | `-1` | 该通道 RX loop 的 CPU 核。 |
| `processing_cpu_core` | `int` | `-1` | 该通道感知处理 loop 的 CPU 核。 |

说明：
* 当 `rx_channels` 为空且 `rx_channel_count > 0` 时，程序按通道号 `0..N-1` 自动补齐默认项。
* 若两者数量不一致，程序会按 `rx_channel_count` 对通道列表做裁剪或扩展。
* 当某通道设置 `enable_system_delay_estimation=true` 时，该通道会在启动附近执行一次系统时延估计，之后每隔 434 个帧重复执行一次，同时继续消耗帧数据；常规感知处理和感知输出保持停用。
* `device_args`、`wire_format_*`、每通道天线口和输出 IP 等字段通常与硬件平台和部署环境强相关；样本 YAML 只是起点，不应把不同机器/射频平台的值机械互换。
* BS 上行 AGC 复用 UE 下行路径相同的跟踪 `HardwareRxAgc` 逻辑：根据上行同步符号的 delay spectrum、同步符号时域样本是否接近/达到 ADC 满幅，调节上行 RX 通道的 USRP 硬件增益。UE 下行专用的 `SYNC_SEARCH` 增益扫描不用于 BS 上行，因为 BS 接收的是已调度的上行窗口，而不是连续同步搜索状态机。

### UE

`UE` (UE 节点) 使用 `UE.yaml` 配置。
可使用 `config/UE_X310.yaml`、`config/UE_B210.yaml` 或 `config/UE_B210_Duplex.yaml` 作为模板。

`UE.yaml` 参数说明：

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `fft_size` | `int` | `1024` | OFDM FFT 点数。 |
| `cp_length` | `int` | `128` | 循环前缀长度（采样点）。 |
| `sync_pos` | `int` | `1` | 同步符号在帧内的位置索引。 |
| `enable_sec_sync_symbol` | `bool` | `false` | 预留 `sync_pos-1` 作为重复 ZC 第二同步符号。启用后，初始同步先用两个连续同步符号做类 Schmidl-Cox 粗定时/模糊 CFO 估计，再用局部 ZC 相关解析 CFO alias 并精修主同步。发射端必须使用相同设置。 |
| `enable_cfo_training_sequence` | `bool` | `false` | 使用 `sync_pos+1` 的专用 CFO training field 来解析 CFO alias。帧定位仍来自 ZC 或可选第二同步符号；接收端先估计 CP/第二同步 modulo CFO，再用 CFO Field 的重复训练 CFO 估计选择最近的 alias。发射端必须使用相同设置。 |
| `cfo_training_period_samples` | `int` / 采样点 | `16` | CFO training field 的重复周期，必须整除 `fft_size`；无模糊 CFO 范围约为 `+-sample_rate/(2*period)`。 |
| `sync_cfo_alias_search_range_hz` | `float` / Hz | `800000` | 同步 alias 解析覆盖的最大绝对 CFO 范围。接收端会根据 CP 和第二同步符号各自的 modulo 周期，把这个物理范围换算为整数 alias 搜索跨度。`profiling_modules` 包含 `sync` 时会打印逐 alias 峰值比较。 |
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
| `tx_channel` | `int` | `0` | 开启双工/上行时 UE 使用的 TX 通道索引。下行-only 的 UE 运行不需要这一路 TX。 |
| `zc_root` | `int` | `29` | Zadoff-Chu 根序号。 |
| `num_symbols` | `int` | `100` | 每帧 OFDM 符号数。 |
| `sensing_symbol_num` | `int` | `100` | 参与感知处理的符号数。 |
| `output_mode` | `string` | `dense` | 双站感知输出模式。`dense` 保持旧版基于 STRD 的全缓冲区输出；`compact_mask` 切换为按帧提取紧凑感知 RE。 |
| `bi_enabled` | `bool` | `true` | 启用双站感知处理链；设为 `false` 时 `UE` 与 `CUDAUE` 均不会启动双站感知通道。 |
| `duplex_mode` | `string` | `tdd` | 必须与 `BS.yaml` 保持一致。`tdd` 共享下行中心频率并只在配置的上行符号窗口发送；`fdd` 在 `uplink.center_freq` 上按整帧连续发送。 |
| `idle_waveform` | `string` | `random_qpsk` | UE 无上行 UDP 载荷时的 idle 波形。`random_qpsk` 发送 zero-length mini-header 后接确定性随机 QPSK 填充；`zero` 发送 zero-length mini-header，剩余 payload RE 保持为 0。 |
| `uplink` | `object` | `symbol_start=90`、`symbol_count=10`、`guard_symbols=1`、`center_freq=2500000000` | UE 上行设置。TDD 使用 `symbol_start`、`symbol_count`、`guard_symbols` 并忽略 `center_freq`；FDD 使用 `center_freq` 并忽略 TDD 符号窗口字段，按整帧连续传输。开启上行需要 UE 端具备 TX 天线/RF 链路，BS 端也必须具备上行 RX 路径。 |
| `ue_timing_advance` | `int` / 采样点 | `63` | UE 侧上行发送 timing advance。UE 启动时会让 UL TX 和 RX 同时开启，之后根据 RX 同步/对齐结果和运行时可调的 `TADV` 值移动后续上行帧。 |
| `ertm_to_enable` | `bool` | `false` | 开启 CPU 路径 eRTM TO payload 消费和 TO 日志。UE 会根据 BS 发来的上行频域信道和本地下行频域信道，按配置倍数做补零 IFFT 得到 delay spectrum，再进行 eRTM 相关并用 centroid3 做相关峰小数 bin 细化。 |
| `ertm_delay_oversample_factor` | `int` | `10` | eRTM delay spectrum IFFT 过采样倍数。小于 `1` 的值按 `1` 处理，大于 `128` 的值按 `128` 处理；更高倍数可提高 delay 网格分辨率，但会增加 CPU/FFTW/内存开销。 |
| `sensing_delay_correction_mode` | `string` | `los_tracking` | UE `sensing` 设置中的感知时延校正来源。`los_tracking` 使用本地 LoS 同步/SFO tracking delay；`ertm_absolute` 在可用时使用 CPU 路径 eRTM 最新 `TO_UE_samples` 作为绝对感知时延，然后仍叠加运行时 `ALGN` 用户微调。CUDAUE 目前对该模式回退到 `los_tracking`。 |
| `ertm_dl_rf_delay_ns` | `float` / ns | `0.0` | eRTM TO 方程使用的下行 RF 链路校准延迟项。 |
| `ertm_ul_rf_delay_ns` | `float` / ns | `0.0` | eRTM TO 方程使用的上行 RF 链路校准延迟项。 |
| `ertm_debug_output_enabled` | `bool` | `false` | 启用 UE 侧 eRTM debug ZeroMQ 输出，包含按配置倍数过采样的 BS 上行 delay spectrum、UE 下行 delay spectrum、eRTM 相关谱、TO 校正后 debug 谱的中心窗口和峰值元数据。可用 `scripts/plot_ertm_debug.py` 查看。 |
| `ertm_report_interval_frames` | `int` / 帧 | `32` | BS eRTM payload/report 的下行 TX 帧间隔；对比日志时应与 BS 保持一致。 |
| `cuda_demod_pipeline_slots` | `int` | `3` | CUDA 解调流水线 slot 数。小于 `1` 时会钳制到 `1`。 |
| `frame_queue_size` | `int` | `8` | UE RX 帧队列容量。小于 `1` 时会钳制到 `1`。 |
| `sync_queue_size` | `int` | `8` | 同步搜索批队列容量。小于 `1` 时会钳制到 `1`。 |
| `reset_hold_s` | `float` / s | `0.5` | 在强制回到同步搜索前，坏的 delay 条件必须持续累积的时间。内部会按 `samples_per_frame / sample_rate` 换算成帧数阈值。小于 `0` 时会钳制到 `0.5`。 |
| `range_fft_size` | `int` | `1024` | 距离向 FFT 点数。 |
| `doppler_fft_size` | `int` | `100` | 多普勒向 FFT 点数。 |
| `pilot_positions` | `int[]` | `[571,631,...,451]` | 分布在占用带宽内的可配置梳状导频子载波索引。 |
| `midframe_pilot_symbols` | `int[]` | `[]` | 可选的帧内 BPSK 导频符号索引。接收机会把完整已知符号作为额外信道估计 anchor，同时保留梳状导频 RE 用于相位跟踪，并从 payload LLR 提取中排除。 |
| `midframe_pilot_seed` | `int` | `1296453708` | 确定性帧内 BPSK 导频种子，必须与发射端一致。 |
| `equalizer_mode` | `string` | `mmse` | 通信均衡器反演模式。`zf` 使用带下限的信道功率分母；`mmse` 会在该分母上加入 `noise_var`，可降低深衰落处的噪声增强。 |
| `channel_tracking_mode` | `string` | `pilot_phase` | CPU 和 CUDA UE 都支持的每符号梳状导频跟踪模式。`disabled` 使用同步符号得到的固定信道，`pilot_phase` 对每个数据信号符号用梳状导频残差拟合公共相位和线性相位。 |
| `equalizer_mag_floor` | `float` | `1e-6` | 信道幅度平方反演下限，`zf` 和 `mmse` 都会使用。 |
| `channel_tracking_min_pilot_snr` | `float` | `1e-4` | 每符号跟踪接受梳状导频残差的最小功率/权重，低于该值时回退到同步符号信道。 |
| `data_resource_blocks` | `object[]` | 缺省 | 接收侧的通信资源映射，用来回答“哪些 RE 应该被当作 payload 来解调”。省略该键时保持旧行为：除同步符号和梳状导频 RE 外的所有 RE 都参与 payload 提取。设为 `[]` 表示完全不提取 payload LLR。应与发射端使用相同的矩形块和 `kind`。其中 `kind: payload` 的块会产生 payload LLR；`kind: sensing_pilot` 的块则会被当作已知参考 RE，不参与 payload 提取。该已知参考序列与发射端保持一致，也使用不同于帧同步符号的备选 Zadoff-Chu 根。 |
| `mask_blocks` | `object[]` | 缺省 | 接收侧的紧凑感知资源映射，用来回答“在 `compact_mask` 模式下，双站感知要导出哪些 RE”。坐标系和行为与发射端一致：使用绝对帧符号索引和原始 FFT bin 索引，允许选择 ZC 同步符号、梳状导频或帧内 BPSK 导频 RE，但拒绝 CFO training field；重叠块自动并集，输出顺序固定为“先符号、后子载波”。如果 mask 满足规则采样条件，同样可以开启运行时 `MTI` 和本地 Delay-Doppler 处理。 |
| `device_args` | `string` | `""` | USRP 参数。 |
| `clock_source` | `string` | `internal/external/gpsdo` | 时钟源。 |
| `wire_format_tx` | `string` | `sc16` | 可选 UE 上行 TX 链路数据格式，常用 `sc16` 或 `sc8`。 |
| `rx_wire_format` | `string` | `sc16` | UE 下行 RX 链路数据格式，常用 `sc16` 或 `sc8`。 |
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
| `bi_sensing_output_enabled` | `bool` | `true` | 启用双站感知 ZeroMQ PUB 输出。处理链可以保持开启，同时关闭该输出。 |
| `bi_sensing_ip` | `string` / IPv4 | `0.0.0.0` | 双站感知数据流和控制通道的 ZMQ 绑定 IP。使用 `0.0.0.0` 可接受远端前端连接；使用 `127.0.0.1` 则仅允许本机连接。 |
| `bi_sensing_port` | `int` | `8889` | 双站感知数据流的 ZeroMQ PUB 绑定端口。 |
| `ertm_debug_ip` | `string` / IPv4 | `0.0.0.0` | UE eRTM debug 输出的 ZeroMQ PUB 监听 IP。 |
| `ertm_debug_port` | `int` | `12362` | UE eRTM debug 输出的 ZeroMQ PUB 绑定端口。 |
| `channel_ip` | `string` / IPv4 | `0.0.0.0` | 信道估计输出的 ZeroMQ PUB 监听 IP。留空会解析为 `0.0.0.0`，不会使用 `default_out_ip`。 |
| `channel_port` | `int` | `12348` | 信道估计输出的 ZeroMQ PUB 绑定端口。 |
| `pdf_ip` | `string` / IPv4 | `0.0.0.0` | PDP/PDF 输出的 ZeroMQ PUB 监听 IP。留空会解析为 `0.0.0.0`，不会使用 `default_out_ip`。 |
| `pdf_port` | `int` | `12349` | PDP/PDF 输出的 ZeroMQ PUB 绑定端口。 |
| `constellation_ip` | `string` / IPv4 | `0.0.0.0` | 星座图输出的 ZeroMQ PUB 监听 IP。留空会解析为 `0.0.0.0`，不会使用 `default_out_ip`。 |
| `constellation_port` | `int` | `12346` | 星座图输出的 ZeroMQ PUB 绑定端口。 |
| `vofa_debug_ip` | `string` / IPv4 | `127.0.0.1` | VOFA+ 调试输出 IP。 |
| `vofa_debug_port` | `int` | `12347` | VOFA+ 调试输出端口。 |
| `udp_input_ip` | `string` / IPv4 | `0.0.0.0` | UE 上行业务 UDP 输入绑定地址，即 UE->BS 上行要发送的业务流。 |
| `udp_input_port` | `int` | `50002` | UE 上行业务 UDP 输入绑定端口。 |
| `udp_output_ip` | `string` / IPv4 | `127.0.0.1` | UE 解码后的下行业务 UDP 输出目标 IP，即从 BS->UE 下行恢复出的业务流。 |
| `udp_output_port` | `int` | `50001` | UE 解码后的下行业务 UDP 输出目标端口。 |
| `udp_egress_pacer_enabled` | `bool` | `false` | 启用解码后业务 UDP 输出的排队 pacing，用于平滑 decoder 造成的突发输出。 |
| `udp_egress_pacer_target_mbps` | `float` | `0` | UDP egress pacer 的目标载荷速率。`0` 表示根据入队速率自动估计；正数表示固定 Mbps 速率。 |
| `udp_egress_pacer_queue_packets` | `int` | `10240` | egress pacer 最多缓存的 UDP 数据报数量，超过后丢弃最旧数据报。 |
| `udp_egress_pacer_max_delay_ms` | `float` | `0` | egress pacer 中数据报允许排队的最长时间；设为 `0` 可关闭按年龄丢包。 |
| `default_out_ip` | `string` / IPv4 | `127.0.0.1` | UDP 业务数据和 VOFA+ 调试输出的默认目标 IP；对应 IP 字段留空时使用该值。ZeroMQ PUB 监听 IP 不继承该值。 |
| `control_port` | `int` | `10001` | 双向控制通道的 ZeroMQ ROUTER 绑定端口。 |
| `measurement_enable` | `bool` | `false` | 启用 CPU 版内部测量模式。启用后，测量载荷不会再转发到 `udp_output_*`，而是直接用于 BER/BLER/EVM 统计。CUDA 二进制忽略该模式。 |
| `measurement_mode` | `string` | `internal_prbs` | 测量模式选择。目前仅支持 `internal_prbs`；非法值会在配置归一化阶段自动关闭测量模式。 |
| `measurement_run_id` | `string` | `""` | 写入测量 CSV 汇总的运行 ID。 |
| `measurement_output_dir` | `string` | `""` | CPU 测量汇总文件输出目录。 |
| `measurement_payload_bytes` | `int` | `1024` | 每个测量载荷期望的字节数。若小于内部测量头长度，会自动钳制到最小合法值。 |
| `measurement_prbs_seed` | `int` | `0x5A` | 用于重建确定性 PRBS 测量载荷的基础种子。 |
| `measurement_packets_per_point` | `int` | `1` | 每个在线 `MRST` epoch 期望统计的测量载荷数。小于 `1` 时会钳制到 `1`。 |
| `profiling_modules` | `string` | `""` | 性能统计模块列表，逗号分隔。常用值包括 `demodulation`、`cfo`、`sync`、`agc`、`align`、`snr`、`uplink`、`ertm`；`all` 表示全部。`cfo` 控制 CUDA CFO 诊断日志，`sync` 控制逐 alias 同步峰值比较日志，`agc` 控制 AGC 日志，`align` 控制运行时 `ALGN:` 日志，`snr` 会周期性打印当前 UE 接收路径的 `_snr_db / _noise_var / _llr_scale`，`uplink` 控制 `[UL-TX]` timing/waveform 诊断日志，`ertm` 控制 eRTM TO/payload/debug-spectrum 诊断日志。感知时延校正 warning 和 pending alignment 更新日志仍常显。 |
| `downlink_cpu_cores` | `int[]` | `[]` | UE 下行 CPU 核列表：索引 `0..2` 分别绑定 `rx_proc`、`process_proc` 和 `bit_processing_proc`。 |
| `demod_worker_cpu_cores` | `int[]` | `[]` | UE CPU 解调 worker 核列表，用于帧级 OFDM/LLR 处理。每个配置的核对应 1 个 worker；留空时启动 1 个不绑核的 worker（建议配置专用核以获得稳定的实时性能）。 |
| `ldpc_worker_cpu_cores` | `int[]` | `[]` | UE CPU LDPC 解码 worker 核列表。每个配置的核对应 1 个持有独立 LDPC 解码器实例的 worker；留空时启动 1 个不绑核的 worker。 |
| `sensing_cpu_cores` | `int[]` | `[]` | UE 双站感知 CPU 核列表：索引 `0` 绑定 `sensing_process_proc`。 |
| `uplink_cpu_cores` | `int[]` | `[]` | UE 上行 CPU 核列表：索引 `0..3` 分别绑定 `UplinkTxEngine::_ldpc_encode_proc`、`_mod_proc`、`_tx_proc` 和 `_udp_recv_proc`。 |
| `main_cpu_core` | `int` | `-1` | 主线程 CPU 核。 |

说明：
* `data_resource_blocks` 通常应与发射端完全一致，包括 `kind`。
* 如果资源块与 `sync_pos`、可选的 `sync_pos-1` 第二同步符号、`midframe_pilot_symbols` 或 `pilot_positions` 重叠，内置的 ZC 同步符号、梳状导频 RE 和帧内 BPSK 导频仍然优先。可选 `sync_pos+1` CFO training field 会在 sensing-pilot 和 sensing-mask 选择中被拒绝。优先级是“ZC 同步符号 > CFO training field > 梳状导频 RE > 帧内 BPSK 导频 > sensing_pilot > payload/预生成 QPSK”。
* dense 模式下，如果 `symbol_stride` / 运行时 `STRD` 会采到可选的 `sync_pos+1` CFO training field，会被拒绝。运行时修改 `STRD` 会在计划好的帧边界重启确定性采样相位。
* 当 `output_mode=compact_mask` 时，双站感知同样会变成“每个 OFDM 帧发送一条紧凑消息”，只包含 `mask_blocks` 选中的 RE；此时 `STRD` 会被忽略，因为 mask 已经定义了采样图样。
* 紧凑载荷格式与发射端一致：`CompactSensingFrameHeader` 后面跟固定顺序的原始 `complex<float>` 数据。
* UE 下行 RX AGC 分为两个阶段。`SYNC_SEARCH` 阶段会先把增益恢复到配置的 `rx_gain`，然后进行粗搜索扫描（每 10 个帧增加 `1 dB`，达到最大增益后回绕到最小增益）；锁定后进入跟踪阶段，使用 `rx_agc_low_threshold_db` / `rx_agc_high_threshold_db` 定义的窗口来细调增益。
* UE 接收机会检查同步符号时域样本的 I/Q 分量是否接近或达到 ADC 满幅。如果满幅点数过多，会立即强制降低增益，并在短时间内禁止再次升增益，避免在噪声或削顶附近来回摆动。
* 硬 reset 会清空时延/频偏跟踪状态、刷新队列、重置跟踪 AGC 状态，并回到 `SYNC_SEARCH`。`reset_hold_s` 决定坏的 delay 条件需要持续多久才会触发这一动作。
