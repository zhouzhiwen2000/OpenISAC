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
- 网页控制台: [网页配置控制台](#9-网页配置控制台)
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

### 7. 校准系统时延

请先完成系统时延校准，再执行后面的系统响应校准。校准前请先进行射频直连：将发射 RF 输出连接到对应的感知 RX 输入，并在校准过程中保持连接稳定。首先为射频直连降低发射功率。可以先降低 `build/BS.yaml` 中的 `downlink.tx_gain`；如有需要，在直连链路中加入合适的衰减器，避免感知 RX 饱和。

只为当前要测试的感知通道开启系统时延估计：

```yaml
sensing:
  rx_channels:
    - usrp_channel: 1
      alignment: 63
      enable_system_delay_estimation: true
```

从 `build/` 目录启动 BS 后端。该模式下，被选中的感知通道会关闭常规感知处理链，并周期性执行基于 ZC 相关的系统时延测试。观察 BS 控制台输出：CPU 版本会打印 `[SYSDLY CH <n>]`，CUDA 版本会打印 `[CUDA SYSDLY CH <n>]`。CPU 日志中的建议值字段是 `alignment_suggest=<value>`；CUDA 日志中的建议值字段是 `suggest=<value>`。

当建议值稳定后，停止后端，把该值写回同一通道的 `alignment` 字段，并关闭估计模式：

```yaml
sensing:
  rx_channels:
    - usrp_channel: 1
      alignment: <suggested value>
      enable_system_delay_estimation: false
```

每个感知通道都需要单独执行一次射频直连测量。多通道系统中，请把直连线切换到下一个感知 RX 路径，并更新该通道自己的 `alignment`；不要把一个通道的结果直接复用到其他 RF 路径。

### 8. 校准感知通道系统响应

校准前请先进行射频直连：将发射 RF 输出连接到对应的感知 RX 输入，并在校准过程中保持连接稳定。

如果发射功率较大，建议优先降低发射功率；必要时在直连链路中加入合适的衰减器，避免 RX 饱和。衰减器本身应在信号带宽内尽量平坦，否则其带内起伏会被一并计入校准结果。

单站感知校准时，先启动 BS 后端和单站感知前端，在前端中切换到需要校准的感知通道，然后点击系统响应校准。如果是单站多通道配置，请分别对需要校准的通道执行一次。

双站感知校准时，先启动 BS 和 UE 两端后端，打开双站感知前端，并在射频直连状态下点击双站前端里的系统响应校准。

等待后端日志提示校准完成并保存校准文件。当前运行会立即使用新的系统响应校准结果；之后重新启动后端时，程序会自动加载匹配的校准文件。如果没有找到匹配文件，程序会继续运行并打印提示，感知处理会跳过系统响应补偿。

校准完成后，再恢复正常的天线连接或实验连接，然后继续进行空口测量。

### 9. 网页配置控制台
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

运行时配置采用层级化 YAML。下面的表格使用完整 YAML 参数名，例如 `uplink.arq_enabled`，避免 BS/UE 或下行/上行里的同名字段混淆。可选 section 可以省略；缺省值由解析器补齐，`config/` 下的样例文件给出了常用硬件、双工和仿真配置。

### BS

`BS` 从当前工作目录读取 `BS.yaml`。可从 `config/BS_X310.yaml`、`config/BS_B210.yaml`、双工模板或仿真模板开始修改。

#### BS radio

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `radio.radio_backend` | `string` | `uhd` | Radio I/O 后端。`uhd` 表示真实 USRP，`sim` 表示共享内存信道仿真器。 |

#### BS simulation

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `simulation.session` | `string` | `oisac_sim` | BS、UE 和 `ChannelSimulator` 共用的仿真 session 命名空间。 |
| `simulation.enable_comm_rx` | `bool` | `true` | 仿真器生成 UE 通信接收路径。 |
| `simulation.enable_sensing_rx` | `bool` | `true` | 仿真器生成单站感知 RX 路径。 |
| `simulation.enable_uplink` | `bool` | `false` | 仿真器转发 UE 到 BS 的上行流。 |
| `simulation.pacing_enabled` | `bool` | `true` | 按真实采样时间 pacing 仿真输出。 |
| `simulation.noise_power_dbfs` | `float` / dBFS | `-70` | 每个 RX 通道的 AWGN 功率；很低的值近似关闭噪声。 |
| `simulation.snr_control_enable` | `bool` | `false` | 在加 AWGN 前缩放干净信号，以维持 `target_snr_db`。 |
| `simulation.target_snr_db` | `float` / dB | `40` | 开启 SNR 控制时的目标 SNR。 |
| `simulation.control_port` | `int` | `10002` | ChannelSimulator 运行时 SNR 控制 ZMQ 端口。 |
| `simulation.cfo_hz` | `float` / Hz | `0` | UE RX 校正前注入的初始载波频偏。 |
| `simulation.sample_rate_offset_ppm` | `float` / ppm | `0` | UE 相对 BS 的采样钟偏差。 |
| `simulation.timing_offset_samples` | `int` / samples | `0` | 注入到 RX 的固定整数采样延迟。 |
| `simulation.array_spacing_m` | `float` / m | `0.04283` | ULA 物理阵元间距；`<=0` 时使用 `array_spacing_lambda`。 |
| `simulation.array_spacing_lambda` | `float` / lambda | `0.5` | 旧版按波长归一化的 ULA 间距。 |
| `simulation.ring_capacity_samples` | `int` / samples | `262144` | 每条共享内存流的 ring 容量。 |
| `simulation.steering_override_file` | `string` | `""` | 可选 steering matrix 文件；空字符串使用 ULA steering。 |
| `simulation.comm_multipath_taps[]` | `object[]` | 可选 | 通信 tapped-delay-line 信道 tap，字段为 `delay_samples`、`gain_db`、`phase_deg`。 |
| `simulation.targets[]` | `object[]` | 可选 | 单站感知点目标，字段为 `range_m`、`velocity_mps`、`gain_db`、`angle_deg`。 |
| `simulation.bistatic_targets[]` | `object[]` | 可选 | 双站/通信点目标，字段同上。 |

#### BS rf_sampling

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `rf_sampling.sample_rate` | `float` / Hz | `50000000` | 基带采样率。 |
| `rf_sampling.bandwidth` | `float` / Hz | `50000000` | 模拟带宽，通常与采样率一致。 |

#### BS usrp_device

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `usrp_device.device_args` | `string` | `addr=...` | 通用 USRP device args。 |

#### BS clock_time

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `clock_time.clock_source` | `string` | `external` | 全局时钟源：`internal`、`external` 或 `gpsdo`。 |
| `clock_time.time_source` | `string` | `internal` | 全局时间/PPS 源；空字符串表示跟随 `clock_source`。 |

#### BS ofdm_frame

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ofdm_frame.fft_size` | `int` | `1024` | OFDM FFT 点数。 |
| `ofdm_frame.cp_length` | `int` / samples | `128` | 循环前缀长度。 |
| `ofdm_frame.sync_pos` | `int` | `1` | 帧内同步符号索引。 |
| `ofdm_frame.enable_sec_sync_symbol` | `bool` | `false` | 预留 `sync_pos-1` 作为重复 ZC 同步符号。 |
| `ofdm_frame.enable_cfo_training_sequence` | `bool` | `false` | 预留 `sync_pos+1` 作为重复 CFO training field。 |
| `ofdm_frame.cfo_training_period_samples` | `int` / samples | `16` | CFO training field 重复周期，必须整除 `fft_size`。 |
| `ofdm_frame.num_symbols` | `int` | `100` | 每帧 OFDM 符号数。 |
| `ofdm_frame.sensing_symbol_num` | `int` | `100` | 参与感知处理的符号数。 |
| `ofdm_frame.zc_root` | `int` | `29` | 同步/前导 Zadoff-Chu 根。 |
| `ofdm_frame.pilot_positions` | `int[]` | `[571,...]` | 梳状导频子载波索引。 |
| `ofdm_frame.midframe_pilot_symbols` | `int[]` | `[]` | 可选帧内 BPSK 导频符号索引。 |
| `ofdm_frame.midframe_pilot_seed` | `int` | `1296453708` | 确定性帧内 BPSK 导频种子，TX/RX 必须一致。 |

#### BS cuda

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `cuda.cuda_mod_pipeline_slots` | `int` | `3` | CUDA 调制流水线 slot 数，小于 `1` 会钳制。 |

#### BS ldpc

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ldpc.fixed_point` | `bool` | `false` | 使用 int16/Q16 layered-NMS CPU 解码器而不是 float32。 |
| `ldpc.fixed_point_scale` | `int` | `16` | 固定点模式下 int16 饱和前的 LLR 缩放。 |

#### BS downlink

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `downlink.center_freq` | `float` / Hz | `2400000000` | BS 下行 RF 中心频率。 |
| `downlink.tx_gain` | `float` / dB | `60` | BS 下行 TX 增益。 |
| `downlink.tx_channel` | `int` | `0` | BS 下行 TX 通道索引。 |
| `downlink.tx_device_args` | `string` | `""` | TX 专用 device args；空字符串使用 `usrp_device.device_args`。 |
| `downlink.tx_clock_source` | `string` | `""` | TX 时钟源覆盖。 |
| `downlink.tx_time_source` | `string` | `""` | TX 时间源覆盖。 |
| `downlink.wire_format_tx` | `string` | `sc16` | TX wire format，常用 `sc16` 或 `sc8`。 |
| `downlink.arq_enabled` | `bool` | `false` | 在 BS 发射端开启下行 ARQ。 |
| `downlink.arq_window_packets` | `int` | `32767` | 下行 ARQ outstanding packet 窗口。 |
| `downlink.arq_retransmit_timeout_ms` | `int` / ms | `100` | 下行 ARQ 重传超时。 |
| `downlink.arq_max_retries` | `int` | `5` | 下行最大重传次数；`0` 表示窗口内不限次数。 |

#### BS downlink_pipeline

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `downlink_pipeline.tx_circular_buffer_size` | `int` | `8` | 送入 TX 的已调制帧队列容量。 |
| `downlink_pipeline.data_packet_buffer_size` | `int` | `256` | 已编码 packet buffer 容量。 |

#### BS uplink

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `uplink.enabled` | `bool` | `false` | UE 到 BS 上行/双工路径总开关。 |
| `uplink.duplex_mode` | `string` | `tdd` | `tdd` 使用上行符号窗口；`fdd` 使用 `uplink.center_freq` 和整帧上行。 |
| `uplink.center_freq` | `float` / Hz | `2500000000` | 仅 FDD 使用的上行载波；TDD 使用下行中心频率。 |
| `uplink.symbol_start` | `int` | `90` | TDD 下上行窗口起始符号。 |
| `uplink.symbol_count` | `int` | `10` | TDD 下上行窗口长度；`0` 关闭 TDD 上行。 |
| `uplink.guard_symbols` | `int` | `1` | TDD 上行窗口内的前置 guard 符号数。 |
| `uplink.bs_dl_ul_timing_diff` | `int` / samples | `63` | BS 上行 RX 窗口相对下行 TX 帧锚点的偏移。 |
| `uplink.debug_self_channel` | `bool` | `false` | 从上行 RX 窗口估计本机 TX 泄漏/self channel，用于 `DUTI` 调试。 |
| `uplink.ertm_to_enable` | `bool` | `false` | 开启 eRTM timing-offset payload 和 UE 侧 TO 日志。 |
| `uplink.ertm_report_interval_frames` | `int` / frames | `32` | BS eRTM payload/report 的下行 TX 帧间隔。 |
| `uplink.rx_gain` | `float` / dB | `0` | BS 上行 RX 增益。 |
| `uplink.rx_channel` | `int` | `0` | BS 上行 RX 通道索引。 |
| `uplink.rx_wire_format` | `string` | `sc16` | BS 上行 RX wire format。 |
| `uplink.rx_device_args` | `string` | `""` | 上行 RX device args 覆盖。 |
| `uplink.rx_clock_source` | `string` | `""` | 上行 RX 时钟源覆盖。 |
| `uplink.rx_time_source` | `string` | `""` | 上行 RX 时间源覆盖。 |
| `uplink.rx_agc_enable` | `bool` | `false` | 开启 BS 上行硬件 RX AGC。 |
| `uplink.rx_agc_low_threshold_db` | `float` / dB | `14` | 滤波后 delay-spectrum 主峰低于该阈值时提高增益。 |
| `uplink.rx_agc_high_threshold_db` | `float` / dB | `16` | 主峰高于该阈值时降低增益。 |
| `uplink.rx_agc_max_step_db` | `float` / dB | `1` | 单次上行 RX AGC 最大步进。 |
| `uplink.rx_agc_update_frames` | `int` | `4` | 两次上行 AGC 更新之间最少处理帧数。 |
| `uplink.equalizer_mode` | `string` | `mmse` | BS 上行均衡器反演模式：`zf` 或 `mmse`。 |
| `uplink.channel_tracking_mode` | `string` | `pilot_phase` | 上行每符号梳状导频跟踪模式：`disabled` 或 `pilot_phase`。 |
| `uplink.equalizer_mag_floor` | `float` | `1e-6` | 上行信道反演时 `|H|^2` 下限。 |
| `uplink.channel_tracking_min_pilot_snr` | `float` | `1e-4` | 回退前要求的最小梳状导频残差权重。 |
| `uplink.arq_enabled` | `bool` | `false` | 在 BS 接收端开启上行 ARQ。 |
| `uplink.arq_ordered_delivery` | `bool` | `false` | 缓存上行 packet 以按序输出 UDP。 |
| `uplink.arq_window_packets` | `int` | `32767` | 上行 ARQ 接收/重排窗口。 |
| `uplink.arq_feedback_interval_ms` | `int` / ms | `10` | 上行 ARQ ACK feedback 最小间隔。 |

#### BS sensing

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `sensing.rx_wire_format` | `string` | `sc16` | 默认感知 RX wire format。 |
| `sensing.rx_device_args` | `string` | `""` | 默认感知 RX args。 |
| `sensing.rx_clock_source` | `string` | `""` | 默认感知 RX 时钟源覆盖。 |
| `sensing.rx_time_source` | `string` | `""` | 默认感知 RX 时间源覆盖。 |
| `sensing.rx_channel_count` | `int` | `1` | 单站感知 RX 通道数；`0` 关闭感知 RX。 |
| `sensing.rx_channels[]` | `object[]` | 见下表 | 每通道感知 RX 设置。 |
| `sensing.range_fft_size` | `int` | `1024` | 距离向 FFT 点数。 |
| `sensing.doppler_fft_size` | `int` | `100` | 多普勒向 FFT 点数。 |
| `sensing.view_range_bins` | `int` | `0` | 后端 RD view 宽度；`0` 表示完整 range FFT。 |
| `sensing.view_doppler_bins` | `int` | `0` | 后端 RD view 高度；`0` 表示完整 doppler FFT。 |
| `sensing.output_mode` | `string` | `dense` | `dense` 为基于 STRD 的完整输出；`compact_mask` 只导出选中 RE。 |
| `sensing.on_wire_format` | `string` | `complex_float32` | 感知 payload wire format。 |
| `sensing.backend_processing_enabled` | `bool` | `false` | 在支持时输出后端 RD/CFAR/微多普勒 sidecar。 |
| `sensing.symbol_stride` | `int` | `20` | dense 模式启动时默认 STRD。 |
| `sensing.paired_frame_queue_size` | `int` | `64` | 每个感知通道的 RX/TX 帧配对队列容量。 |
| `sensing.mask_blocks` | 来自 `resource_preview.mask_blocks` | 可选 | 由 resource preview 生成的运行时感知 mask。 |

#### BS sensing.rx_channels[] 字段

| 字段 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `usrp_channel` | `int` | `1` | 当前感知路径使用的 USRP RX 通道。 |
| `device_args` | `string` | `""` | 该通道 device args 覆盖。 |
| `clock_source` | `string` | `""` | 该通道时钟源覆盖。 |
| `time_source` | `string` | `""` | 该通道时间源覆盖。 |
| `wire_format` | `string` | `""` | 该通道 wire-format 覆盖。 |
| `rx_gain` | `float` / dB | `30` | 该通道 RX 增益。 |
| `alignment` | `int` / samples | `63` | 该通道定时 alignment。 |
| `rx_antenna` | `string` | `RX2` | RX 天线端口，如 `RX1`、`RX2`、`TX/RX`。 |
| `enable_system_delay_estimation` | `bool` | `false` | 周期执行基于 ZC 的系统时延估计，并关闭该通道常规感知处理。 |
| `enable_sensing_output` | `bool` | 继承输出开关 | 该通道单站感知输出开关。 |
| `rx_cpu_core` | `int` | `-1` | 该通道 RX loop CPU 核。 |
| `processing_cpu_core` | `int` | `-1` | 该通道感知处理 loop CPU 核。 |

### UE

`UE` 从当前工作目录读取 `UE.yaml`。可从 `config/UE_X310.yaml`、`config/UE_B210.yaml`、双工模板或仿真模板开始修改。

#### BS resource_preview

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `resource_preview.data_resource_blocks[]` | `object[]` | 可选 | payload / sensing-pilot RE 矩形；每项包含 `kind`、`symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`。 |
| `resource_preview.mask_blocks[]` | `object[]` | 可选 | compact 感知 RE 矩形；每项包含 `symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`。 |

#### BS measurement

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `measurement.measurement_enable` | `bool` | `false` | 开启内部 PRBS 测量流量。 |
| `measurement.measurement_mode` | `string` | `internal_prbs` | 测量生成/检查模式。 |
| `measurement.measurement_run_id` | `string` | `""` | 写入测量 CSV 的 run ID。 |
| `measurement.measurement_output_dir` | `string` | `""` | 测量 CSV 输出目录。 |
| `measurement.measurement_payload_bytes` | `int` / bytes | `1024` | 每个测量 payload 字节数。 |
| `measurement.measurement_prbs_seed` | `int` | `0x5A` | 确定性 PRBS payload 种子。 |
| `measurement.measurement_packets_per_point` | `int` | `1` | 每个测量 epoch 发送的 packet 数。 |
| `measurement.measurement_max_packets_per_frame` | `int` | `1` | 每帧最多拉取的测量 packet 数；`0` 表示不限。 |

#### BS network_output

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `network_output.udp_input_ip` | `string` / IPv4 | `0.0.0.0` | BS 下行业务 UDP 输入绑定 IP。 |
| `network_output.udp_input_port` | `int` | `50000` | BS 下行业务 UDP 输入端口。 |
| `network_output.udp_output_ip` | `string` / IPv4 | `127.0.0.1` | BS 解码上行 UDP 输出目标 IP。 |
| `network_output.udp_output_port` | `int` | `50003` | BS 解码上行 UDP 输出目标端口。 |
| `network_output.udp_egress_pacer_enabled` | `bool` | `false` | 开启解码 UDP 输出 pacing。 |
| `network_output.udp_egress_pacer_target_mbps` | `float` / Mbps | `0` | pacing 目标速率；`0` 根据入队速率自动估计。 |
| `network_output.udp_egress_pacer_queue_packets` | `int` | `10240` | pacing 队列 datagram 容量。 |
| `network_output.udp_egress_pacer_max_delay_ms` | `float` / ms | `0` | 最大排队时间；`0` 关闭按年龄丢包。 |
| `network_output.mono_sensing_output_enabled` | `bool` | `true` | 开启单站感知 ZMQ 输出。 |
| `network_output.mono_sensing_ip` | `string` / IPv4 | `0.0.0.0` | 单站感知/control ZMQ 绑定 IP。 |
| `network_output.mono_sensing_port` | `int` | `8888` | 单站感知 PUB 端口。 |
| `network_output.control_port` | `int` | `9999` | 运行时控制 ZMQ ROUTER 端口。 |
| `network_output.uplink_channel_ip` | `string` / IPv4 | `0.0.0.0` | BS 上行信道估计 debug PUB IP。 |
| `network_output.uplink_channel_port` | `int` | `12358` | BS 上行信道估计 debug PUB 端口。 |
| `network_output.uplink_pdf_ip` | `string` / IPv4 | `0.0.0.0` | BS 上行 delay-spectrum debug PUB IP。 |
| `network_output.uplink_pdf_port` | `int` | `12359` | BS 上行 delay-spectrum debug PUB 端口。 |
| `network_output.uplink_constellation_ip` | `string` / IPv4 | `0.0.0.0` | BS 上行星座图 debug PUB IP。 |
| `network_output.uplink_constellation_port` | `int` | `12356` | BS 上行星座图 debug PUB 端口。 |
| `network_output.self_channel_ip` | `string` / IPv4 | `0.0.0.0` | BS self-channel debug PUB IP。 |
| `network_output.self_channel_port` | `int` | `12360` | BS self-channel debug PUB 端口。 |
| `network_output.self_pdf_ip` | `string` / IPv4 | `0.0.0.0` | BS self-delay-spectrum debug PUB IP。 |
| `network_output.self_pdf_port` | `int` | `12361` | BS self-delay-spectrum debug PUB 端口。 |
| `network_output.ertm_debug_ip` | `string` / IPv4 | `0.0.0.0` | eRTM debug PUB IP。 |
| `network_output.ertm_debug_port` | `int` | `12362` | eRTM debug PUB 端口。 |

#### BS cpu_cores

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `cpu_cores.downlink_cpu_cores` | `int[]` | `[1,2,3,-1]` | BS 下行核心：TX、调制、LDPC 编码、UDP 接收。 |
| `cpu_cores.uplink_cpu_cores` | `int[]` | `[]` | BS 上行核心：RX ingest、OFDM/LLR 处理、LDPC decode + UDP 输出。 |
| `cpu_cores.main_cpu_core` | `int` | `-1` | 主线程 CPU 核；`-1` 表示不绑定。 |

#### BS runtime

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `runtime.profiling_modules` | `string` | `""` | 逗号分隔 profiling 模块，如 `modulation`、`latency`、`ldpc_encode`、`sensing_proc`、`agc`、`arq`、`uplink`、`ertm` 或 `all`。 |


#### UE radio

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `radio.radio_backend` | `string` | `uhd` | Radio I/O 后端。`uhd` 表示真实 USRP，`sim` 表示信道仿真器。 |

#### UE simulation

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `simulation.session` | `string` | `oisac_sim` | 共享仿真 session 命名空间。 |
| `simulation.enable_comm_rx` | `bool` | `true` | 仿真器生成通信 RX 路径。 |
| `simulation.enable_sensing_rx` | `bool` | `true` | 仿真器生成感知 RX 路径。 |
| `simulation.enable_uplink` | `bool` | `false` | 仿真器转发 UE 到 BS 的上行流。 |
| `simulation.pacing_enabled` | `bool` | `true` | 按真实采样时间 pacing 仿真输出。 |
| `simulation.noise_power_dbfs` | `float` / dBFS | `-100` | 每个 RX 通道的 AWGN 功率。 |
| `simulation.snr_control_enable` | `bool` | `false` | 通过缩放干净仿真信号维持 `target_snr_db`。 |
| `simulation.target_snr_db` | `float` / dB | `40` | 开启 SNR 控制时的目标 SNR。 |
| `simulation.control_port` | `int` | `10002` | ChannelSimulator 控制端口。 |
| `simulation.cfo_hz` | `float` / Hz | `0` | 初始载波频偏。 |
| `simulation.sample_rate_offset_ppm` | `float` / ppm | `0` | UE 相对 BS 的采样钟偏差。 |
| `simulation.timing_offset_samples` | `int` / samples | `0` | 固定整数采样延迟。 |
| `simulation.array_spacing_m` | `float` / m | `0.04283` | ULA 物理阵元间距。 |
| `simulation.array_spacing_lambda` | `float` | `0.5` | 按波长归一化的旧版 ULA 间距。 |
| `simulation.ring_capacity_samples` | `int` | `262144` | 共享内存 ring 容量。 |
| `simulation.steering_override_file` | `string` | `""` | 可选 steering matrix 文件。 |
| `simulation.comm_multipath_taps[]` | `object[]` | 可选 | 通信 tap：`delay_samples`、`gain_db`、`phase_deg`。 |
| `simulation.targets[]` | `object[]` | 可选 | 单站点目标。 |
| `simulation.bistatic_targets[]` | `object[]` | 可选 | 双站/通信点目标。 |

#### UE rf_sampling

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `rf_sampling.sample_rate` | `float` / Hz | `50000000` | 基带采样率。 |
| `rf_sampling.bandwidth` | `float` / Hz | `50000000` | 模拟带宽。 |
| `rf_sampling.rx_gain` | `float` / dB | `10` | UE 下行 RX 增益。 |
| `rf_sampling.rx_agc_enable` | `bool` | `true` | 开启 UE 下行硬件 RX AGC。 |
| `rf_sampling.rx_agc_low_threshold_db` | `float` / dB | `14` | 滤波后 delay-spectrum 主峰低于该阈值时提高增益。 |
| `rf_sampling.rx_agc_high_threshold_db` | `float` / dB | `16` | 主峰高于该阈值时降低增益。 |
| `rf_sampling.rx_agc_max_step_db` | `float` / dB | `1` | 单次 RX AGC 最大步进。 |
| `rf_sampling.rx_agc_update_frames` | `int` | `4` | 两次 AGC 更新之间最少处理帧数。 |

#### UE usrp_device

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `usrp_device.device_args` | `string` | `addr=...` | USRP device args。 |
| `usrp_device.clock_source` | `string` | `external` | UE 时钟源：`internal`、`external` 或 `gpsdo`。 |

#### UE ofdm_frame

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ofdm_frame.fft_size` | `int` | `1024` | OFDM FFT 点数。 |
| `ofdm_frame.cp_length` | `int` / samples | `128` | 循环前缀长度。 |
| `ofdm_frame.sync_pos` | `int` | `1` | 同步符号索引。 |
| `ofdm_frame.enable_sec_sync_symbol` | `bool` | `false` | 期望 `sync_pos-1` 处存在重复 ZC 同步符号。 |
| `ofdm_frame.enable_cfo_training_sequence` | `bool` | `false` | 使用 `sync_pos+1` CFO training field 解析 CFO alias。 |
| `ofdm_frame.cfo_training_period_samples` | `int` / samples | `16` | CFO training 重复周期。 |
| `ofdm_frame.num_symbols` | `int` | `100` | 每帧 OFDM 符号数。 |
| `ofdm_frame.sensing_symbol_num` | `int` | `100` | 参与感知的符号数。 |
| `ofdm_frame.frame_queue_size` | `int` | `32` | 解调 RX 帧队列容量。 |
| `ofdm_frame.zc_root` | `int` | `29` | Zadoff-Chu 根。 |
| `ofdm_frame.pilot_positions` | `int[]` | `[571,...]` | 梳状导频子载波索引。 |
| `ofdm_frame.midframe_pilot_symbols` | `int[]` | `[]` | 可选帧内 BPSK 导频符号。 |
| `ofdm_frame.midframe_pilot_seed` | `int` | `1296453708` | 确定性帧内 BPSK 导频种子。 |

#### UE cuda

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `cuda.cuda_demod_pipeline_slots` | `int` | `3` | CUDA 解调流水线 slot 数。 |
| `cuda.cuda_ldpc_decoder_backend` | `string` | `gpu` | CUDA 解调 LDPC decoder backend：`gpu` 或 `cpu`。 |
| `cuda.cuda_ldpc_worker_buffers` | `int` | `3` | CUDA LDPC 异步 worker batch buffer 数。 |
| `cuda.cuda_ldpc_cross_frame_flush_frames` | `int` | `2` | CUDA LDPC batch decode 前最多累计帧数。 |
| `cuda.cuda_ldpc_cross_frame_flush_us` | `float` / us | `1000` | CUDA LDPC 跨帧 batch 最长等待时间。 |

#### UE ldpc

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ldpc.fixed_point` | `bool` | `false` | 使用 int16/Q16 CPU LDPC 解码路径。 |
| `ldpc.fixed_point_scale` | `int` | `16` | int16 饱和前的 LLR 缩放。 |

#### UE downlink

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `downlink.center_freq` | `float` / Hz | `2400000000` | UE 下行 RF 中心频率。 |
| `downlink.rx_wire_format` | `string` | `sc16` | UE 下行 RX wire format。 |
| `downlink.rx_channel` | `int` | `0` | UE 下行 RX 通道索引。 |
| `downlink.equalizer_mode` | `string` | `mmse` | 下行均衡器反演模式：`zf` 或 `mmse`。 |
| `downlink.channel_tracking_mode` | `string` | `pilot_phase` | 每符号梳状导频跟踪模式。 |
| `downlink.equalizer_mag_floor` | `float` | `1e-6` | 信道反演时 `|H|^2` 下限。 |
| `downlink.channel_tracking_min_pilot_snr` | `float` | `1e-4` | 回退前要求的最小梳状导频残差权重。 |
| `downlink.arq_enabled` | `bool` | `false` | 在 UE 接收端开启下行 ARQ。 |
| `downlink.arq_ordered_delivery` | `bool` | `false` | 缓存下行 packet 以按序输出 UDP。 |
| `downlink.arq_window_packets` | `int` | `32767` | 下行 ARQ 接收/重排窗口。 |
| `downlink.arq_feedback_interval_ms` | `int` / ms | `10` | 下行 ARQ ACK feedback 最小间隔。 |

#### UE uplink

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `uplink.enabled` | `bool` | `false` | UE 上行/双工总开关。 |
| `uplink.duplex_mode` | `string` | `tdd` | 必须与 BS 一致：`tdd` 窗口上行或 `fdd` 整帧上行。 |
| `uplink.center_freq` | `float` / Hz | `2500000000` | 仅 FDD 使用的上行载波；TDD 使用下行中心频率。 |
| `uplink.idle_waveform` | `string` | `random_qpsk` | UE 空闲上行波形：`random_qpsk` 或 `zero`。 |
| `uplink.symbol_start` | `int` | `90` | TDD 上行起始符号。 |
| `uplink.symbol_count` | `int` | `10` | TDD 上行窗口长度。 |
| `uplink.guard_symbols` | `int` | `1` | TDD 上行前置 guard 符号数。 |
| `uplink.tx_gain` | `float` / dB | `0` | UE 上行 TX 增益。 |
| `uplink.tx_channel` | `int` | `0` | UE 上行 TX 通道索引。 |
| `uplink.wire_format_tx` | `string` | `sc16` | UE 上行 TX wire format。 |
| `uplink.ue_timing_advance` | `int` / samples | `63` | UE 上行发送 timing advance。 |
| `uplink.debug_self_channel` | `bool` | `false` | 从 RX 窗口估计 UE self-TX 泄漏信道，用于 `TADV` 调试。 |
| `uplink.ertm_to_enable` | `bool` | `false` | 开启 eRTM TO payload 消费和 TO 日志。 |
| `uplink.ertm_delay_oversample_factor` | `int` | `10` | eRTM delay-spectrum IFFT 过采样倍数。 |
| `uplink.ertm_dl_rf_delay_ns` | `float` / ns | `0` | eRTM 方程中的下行 RF 链路校准延迟。 |
| `uplink.ertm_ul_rf_delay_ns` | `float` / ns | `0` | eRTM 方程中的上行 RF 链路校准延迟。 |
| `uplink.ertm_debug_output_enabled` | `bool` | `false` | 开启 UE 侧 eRTM debug ZMQ 频谱输出。 |
| `uplink.ertm_report_interval_frames` | `int` / frames | `32` | BS eRTM report 间隔；对比日志时应与 BS 保持一致。 |
| `uplink.arq_enabled` | `bool` | `false` | 在 UE 发射端开启上行 ARQ。 |
| `uplink.arq_window_packets` | `int` | `32767` | 上行 ARQ outstanding packet 窗口。 |
| `uplink.arq_retransmit_timeout_ms` | `int` / ms | `100` | 上行 ARQ 重传超时。 |
| `uplink.arq_max_retries` | `int` | `5` | 上行最大重传次数；`0` 表示窗口内不限次数。 |

#### UE sync_tracking

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `sync_tracking.sync_queue_size` | `int` | `32` | 同步搜索 batch 队列容量。 |
| `sync_tracking.sync_cfo_alias_search_range_hz` | `float` / Hz | `800000` | 同步 alias 解析覆盖的 CFO 范围。 |
| `sync_tracking.reset_hold_s` | `float` / s | `0.5` | 坏 delay 条件持续多久后硬 reset 回同步搜索。 |
| `sync_tracking.software_sync` | `bool` | `true` | 开启软件同步跟踪。 |
| `sync_tracking.predictive_delay` | `bool` | `true` | 使用基于 CFO 的预测性时延补偿。 |
| `sync_tracking.hardware_sync` | `bool` | `false` | 开启硬件同步模式。 |
| `sync_tracking.hardware_sync_tty` | `string` | `/dev/ttyUSB0` | 硬件同步控制器串口设备。 |
| `sync_tracking.desired_peak_pos` | `int` | `20` | alignment 逻辑使用的目标 delay peak 位置。 |
| `sync_tracking.ocxo_pi_kp_fast` | `float` | `30` | OCXO PI 快速阶段比例增益。 |
| `sync_tracking.ocxo_pi_ki_fast` | `float` | `1` | OCXO PI 快速阶段积分增益。 |
| `sync_tracking.ocxo_pi_kp_slow` | `float` | `30` | OCXO PI 慢速阶段比例增益。 |
| `sync_tracking.ocxo_pi_ki_slow` | `float` | `0.05` | OCXO PI 慢速阶段积分增益。 |
| `sync_tracking.ocxo_pi_switch_abs_error_ppm` | `float` / ppm | `0.0002` | 切换到慢速阶段的 error 阈值。 |
| `sync_tracking.ocxo_pi_switch_hold_s` | `float` / s | `60` | 低于阈值后切换阶段所需持续时间。 |
| `sync_tracking.ocxo_pi_max_step_fast_ppm` | `float` / ppm | `0.01` | 快速阶段单次最大 OCXO 调整量。 |
| `sync_tracking.ocxo_pi_max_step_slow_ppm` | `float` / ppm | `0.01` | 慢速阶段单次最大 OCXO 调整量。 |
| `sync_tracking.ocxo_pi_max_step_ppm` | `float` / ppm | 可选 | 旧版别名，同时设置 fast/slow max-step。 |
| `sync_tracking.akf_enable` | `bool` | `true` | 对硬件同步 `error_ppm` 开启自适应 Kalman filter。 |
| `sync_tracking.akf_bootstrap_frames` | `int` | `64` | AKF 正常更新前的冷启动帧数。 |
| `sync_tracking.akf_innovation_window` | `int` | `64` | 自适应使用的新息历史窗口。 |
| `sync_tracking.akf_max_lag` | `int` | `4` | 最大新息自相关 lag。 |
| `sync_tracking.akf_adapt_interval` | `int` | `64` | 自适应 `Q/R` 更新帧间隔。 |
| `sync_tracking.akf_gate_sigma` | `float` | `3` | 新息门限 sigma。 |
| `sync_tracking.akf_tikhonov_lambda` | `float` | `1e-3` | LS 自适应 Tikhonov 正则项。 |
| `sync_tracking.akf_update_smooth` | `float` | `0.2` | 更新后 `Q/R` 的指数平滑系数。 |
| `sync_tracking.akf_q_wf_min` | `float` | `1e-10` | 白频率噪声下界。 |
| `sync_tracking.akf_q_wf_max` | `float` | `1e2` | 白频率噪声上界。 |
| `sync_tracking.akf_q_rw_min` | `float` | `1e-12` | 随机游走频率噪声下界。 |
| `sync_tracking.akf_q_rw_max` | `float` | `1e1` | 随机游走频率噪声上界。 |
| `sync_tracking.akf_r_min` | `float` | `1e-8` | 观测噪声方差下界。 |
| `sync_tracking.akf_r_max` | `float` | `1e3` | 观测噪声方差上界。 |

#### UE sensing

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `sensing.sensing_delay_correction_mode` | `string` | `los_tracking` | 双站感知时延校正来源：`los_tracking` 或 `ertm_absolute`。 |
| `sensing.bi_enabled` | `bool` | `true` | 开启双站感知处理。 |
| `sensing.range_fft_size` | `int` | `1024` | 距离向 FFT 点数。 |
| `sensing.doppler_fft_size` | `int` | `100` | 多普勒向 FFT 点数。 |
| `sensing.view_range_bins` | `int` | `0` | 后端 RD view 宽度；`0` 表示完整 range。 |
| `sensing.view_doppler_bins` | `int` | `0` | 后端 RD view 高度；`0` 表示完整 doppler。 |
| `sensing.output_mode` | `string` | `dense` | `dense` 完整输出，或 `compact_mask` 选中 RE 输出。 |
| `sensing.on_wire_format` | `string` | `complex_float32` | 感知 payload wire format。 |
| `sensing.backend_processing_enabled` | `bool` | `false` | 在支持时输出后端 RD/CFAR/微多普勒 sidecar。 |
| `sensing.symbol_stride` | `int` | `20` | dense 模式启动时默认 STRD。 |

#### UE resource_preview

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `resource_preview.data_resource_blocks[]` | `object[]` | 可选 | payload / sensing-pilot RE 矩形；每项包含 `kind`、`symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`。 |
| `resource_preview.mask_blocks[]` | `object[]` | 可选 | compact 感知 RE 矩形；每项包含 `symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`。 |

#### UE measurement

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `measurement.measurement_enable` | `bool` | `false` | 开启内部 PRBS 测量检查。 |
| `measurement.measurement_mode` | `string` | `internal_prbs` | 测量检查模式。 |
| `measurement.measurement_run_id` | `string` | `""` | 写入测量 CSV 的 run ID。 |
| `measurement.measurement_output_dir` | `string` | `""` | 测量 CSV 输出目录。 |
| `measurement.measurement_payload_bytes` | `int` / bytes | `1024` | 期望的测量 payload 字节数。 |
| `measurement.measurement_prbs_seed` | `int` | `0x5A` | 重建确定性 PRBS payload 的种子。 |
| `measurement.measurement_packets_per_point` | `int` | `1` | 每个测量 epoch 期望 packet 数。 |
| `measurement.measurement_max_packets_per_frame` | `int` | `1` | 每帧最多检查的测量 packet 数；`0` 表示不限。 |

#### UE network_output

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `network_output.udp_input_ip` | `string` / IPv4 | `0.0.0.0` | UE 上行业务 UDP 输入绑定 IP。 |
| `network_output.udp_input_port` | `int` | `50002` | UE 上行业务 UDP 输入端口。 |
| `network_output.udp_output_ip` | `string` / IPv4 | `""` | UE 解码下行 UDP 输出目标 IP；空字符串使用 `runtime.default_out_ip`。 |
| `network_output.udp_output_port` | `int` | `50001` | UE 解码下行 UDP 输出端口。 |
| `network_output.udp_egress_pacer_enabled` | `bool` | `false` | 开启解码 UDP 输出 pacing。 |
| `network_output.udp_egress_pacer_target_mbps` | `float` / Mbps | `0` | pacing 目标速率；`0` 自动估计。 |
| `network_output.udp_egress_pacer_queue_packets` | `int` | `10240` | pacing 队列容量。 |
| `network_output.udp_egress_pacer_max_delay_ms` | `float` / ms | `0` | 最大排队时间；`0` 关闭按年龄丢包。 |
| `network_output.bi_sensing_output_enabled` | `bool` | `true` | 开启双站感知 ZMQ 输出。 |
| `network_output.bi_sensing_ip` | `string` / IPv4 | `0.0.0.0` | 双站感知/control ZMQ 绑定 IP。 |
| `network_output.bi_sensing_port` | `int` | `8889` | 双站感知 PUB 端口。 |
| `network_output.control_port` | `int` | `10001` | 运行时控制 ZMQ ROUTER 端口。 |
| `network_output.channel_ip` | `string` / IPv4 | `0.0.0.0` | 信道估计 PUB IP。 |
| `network_output.channel_port` | `int` | `12348` | 信道估计 PUB 端口。 |
| `network_output.pdf_ip` | `string` / IPv4 | `0.0.0.0` | Delay-spectrum PUB IP。 |
| `network_output.pdf_port` | `int` | `12349` | Delay-spectrum PUB 端口。 |
| `network_output.constellation_ip` | `string` / IPv4 | `0.0.0.0` | 星座图 PUB IP。 |
| `network_output.constellation_port` | `int` | `12346` | 星座图 PUB 端口。 |
| `network_output.self_channel_ip` | `string` / IPv4 | `0.0.0.0` | UE self-channel debug PUB IP。 |
| `network_output.self_channel_port` | `int` | `12350` | UE self-channel debug PUB 端口。 |
| `network_output.self_pdf_ip` | `string` / IPv4 | `0.0.0.0` | UE self-delay-spectrum debug PUB IP。 |
| `network_output.self_pdf_port` | `int` | `12351` | UE self-delay-spectrum debug PUB 端口。 |
| `network_output.ertm_debug_ip` | `string` / IPv4 | `0.0.0.0` | UE eRTM debug PUB IP。 |
| `network_output.ertm_debug_port` | `int` | `12362` | UE eRTM debug PUB 端口。 |

#### UE cpu_cores

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `cpu_cores.downlink_cpu_cores` | `int[]` | `[1,2,3]` | UE 下行核心：RX、处理、bit processing。 |
| `cpu_cores.demod_worker_cpu_cores` | `int[]` | `[]` | UE CPU demod worker 核；空列表启动一个不绑定 worker。 |
| `cpu_cores.ldpc_worker_cpu_cores` | `int[]` | `[]` | UE CPU LDPC decode worker 核；空列表启动一个不绑定 worker。 |
| `cpu_cores.sensing_cpu_cores` | `int[]` | `[4]` | UE 双站感知核心。 |
| `cpu_cores.uplink_cpu_cores` | `int[]` | `[]` | UE 上行核心：LDPC 编码、调制、TX 发送、UDP 接收。 |
| `cpu_cores.main_cpu_core` | `int` | `-1` | 主线程 CPU 核；`-1` 表示不绑定。 |

#### UE runtime

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `runtime.default_out_ip` | `string` / IPv4 | `127.0.0.1` | 特定输出 IP 为空时，UDP 和 VOFA+ 输出使用的默认目标 IP。 |
| `runtime.vofa_debug_ip` | `string` / IPv4 | `""` | VOFA+ debug 目标 IP；空字符串使用 `default_out_ip`。 |
| `runtime.vofa_debug_port` | `int` | `12347` | VOFA+ debug 目标端口。 |
| `runtime.profiling_modules` | `string` | `""` | 逗号分隔模块，如 `demodulation`、`cfo`、`sync`、`agc`、`align`、`snr`、`arq`、`uplink`、`ertm` 或 `all`。 |

资源映射说明：
* `resource_preview.data_resource_blocks` 通常应在 BS 和 UE 之间保持一致，包括 `kind`。
* 内置 ZC 同步符号、可选 CFO training field、梳状导频和帧内 BPSK 导频优先级高于资源块配置。
* `resource_preview.mask_blocks` 只控制 compact 感知导出。`output_mode=compact_mask` 时运行时 `STRD` 会被忽略，因为采样图样已经由 mask 定义。
* compact 感知 payload 以 `CompactSensingFrameHeader` 开头，后面跟固定顺序的原始 `complex<float>` 数据。
