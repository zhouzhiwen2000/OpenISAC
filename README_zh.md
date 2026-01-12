<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/logo.svg">
    <img src="docs/images/logo.svg" alt="OpenISAC Logo" width="400">
  </picture>
</p>

# OpenISAC

[English Version](README.md)

OpenISAC 是一个基于 OFDM 的通信与感知一体化（ISAC）系统，专为学术实验和快速算法验证而设计。

其目标是提供一个简洁且易于修改的 OFDM 平台，使研究人员能够快速迭代通信与感知算法，而无需处理复杂的标准协议栈（如WIFI、LTE、NR等）。

由于其实现较为简洁，OpenISAC 通常需要更少的计算资源，因此相较于更复杂、功能更全面的系统，能够实现更高的采样率。

如果您觉得这个仓库有用，请引用我们的论文：

> Z. Zhou, C. Zhang, X. Xu, and Y. Zeng, "OpenISAC: An Open-Source Real-Time Experimentation Platform for OFDM-ISAC with Over-the-Air Synchronization," submitted to *IEEE Trans. Wireless Commun.*, Jan. 2026.
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

## 它是什么，又不是什么？

### OpenISAC 是什么？

- 一个用于通信感知一体化（ISAC）研究的极简 OFDM PHY
- 专为原型设计、学术实验和快速算法验证而设计

### OpenISAC 不是什么？

- 兼容标准的实现（它不符合 Wi-Fi 或 5G NR 等标准）
- 全栈标准实现（如 openwifi 或 OpenAirInterface）的替代品

如果您的目标是互操作性、标准合规性或生产级协议栈，那么上述项目是正确的方向。

### 何时使用它

- 原型设计以及测试新的通信、感知算法
- 使用极简 PHY 快速实现新想法
- 不需要互操作性的研究

### 何时不使用它

- 构建兼容 Wi-Fi/NR 的系统
- 需要标准通信系统的研究（完整的 MAC/协议栈、互操作性等）

## 硬件要求
 
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


## 软件要求
 
### 后端 (C++)
 
#### 操作系统
*   **Ubuntu 24.04 LTS**
    *   下载: [http://www.ubuntu.com/download/desktop](http://www.ubuntu.com/download/desktop)
 
#### 依赖项和安装
 
##### 1. UHD (USRP 硬件驱动程序)
按照 Ettus 官方指南安装 UHD 工具链 (请遵循 Ubuntu 24.04 的教程):
*   [在 Linux 上构建和安装 USRP 开源工具链](https://kb.ettus.com/Building_and_Installing_the_USRP_Open-Source_Toolchain_(UHD_and_GNU_Radio)_on_Linux#Update_and_Install_dependencies)
 
> **注意:** 此代码已在 UHD v4.9.0.1 上测试。您可以使用 `git checkout v4.9.0.1` 检出此版本。
 
##### 2. 安装 Aff3ct
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
 
##### 3. 克隆仓库
克隆 OpenISAC 仓库：
 
```bash
cd ~
git clone https://github.com/zhouzhiwen2000/OpenISAC.git
```
 
##### 4. 构建 OpenISAC
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
 
##### 5. 系统性能调优
运行提供的脚本以优化您的系统设置，以满足实时处理需求：
 
```bash
cd ~/OpenISAC
chmod +x set_performance.bash
./set_performance.bash
```
 
> **注意:** 如果您需要启用 `RT_RUNTIME_SHARE` 功能，则需要在 BIOS 设置中关闭 `secure_boot`。
 
###### UHD 线程优先级配置
 
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
 
##### 6. CPU 隔离和执行
为了确保稳定的实时性能，建议为信号处理任务隔离 CPU 核心。我们提供了一个脚本 `isolate_cpus.bash` 来自动处理此问题。
 
**步骤 1: 隔离系统核心**
此命令将限制系统进程在 E-Core (在 Intel 混合架构上) 或一部分CPU核心上运行，让 P-Core 或其它核心自由用于应用程序。
```bash
cd ~/OpenISAC
chmod +x isolate_cpus.bash
sudo ./isolate_cpus.bash
```
 
**步骤 2: 在隔离核心上运行应用程序**
使用脚本的 `run` 命令在隔离的核心上启动您的应用程序。这确保了应用程序可以独占访问高性能CPU核心。
 
```bash
cd build
sudo ../isolate_cpus.bash run ./OFDMModulator
```
 
> **注意:** 如果您已启用 CPU 隔离，请始终使用 `sudo ../isolate_cpus.bash run ...` 启动应用程序。直接执行或手动使用 `taskset` 可能会因亲和性限制而失败。
 
**重置配置 (可选)**
要将系统恢复到默认状态 (允许所有进程使用所有核心):
```bash
sudo ./isolate_cpus.bash reset
```

##### 7. 配置
系统现已支持 YAML 配置，以便更轻松地管理参数。

*   **默认行为**: 如果执行目录中存在 `Modulator.yaml` 或 `Demodulator.yaml`，它将被自动加载。
*   **指定配置文件**: 使用 `-c` 或 `--config` 指定自定义配置文件：
    ```bash
    ./OFDMModulator -c my_config.yaml
    ```
*   **保存配置**: 使用 `-s` 或 `--save-config` 将当前参数（包括默认值和命令行覆盖的参数）保存到 YAML 文件：
    ```bash
    ./OFDMModulator --save-config
    ```
    这将生成包含当前设置的 `Modulator.yaml`（或 `Demodulator.yaml`）。
    
    您也可以指定自定义文件名：
    ```bash
    ./OFDMModulator -s myconfig.yaml
    ```
 
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
cp ../Modulator_X310.yaml Modulator.yaml
sudo ../isolate_cpus.bash run ./OFDMModulator

# 对于 B210:
cp ../Modulator_B210.yaml Modulator.yaml
sudo ../isolate_cpus.bash run ./OFDMModulator
```
*如果您使用单独的计算机作为前端，请添加 `--default-ip=<your front end IP>`。*

### 2. 启动 UE (用户端)
```bash
sudo -s
cd build
# 对于 X310:
cp ../Demodulator_X310.yaml Demodulator.yaml
sudo ../isolate_cpus.bash run ./OFDMDemodulator

# 对于 B210:
cp ../Demodulator_B210.yaml Demodulator.yaml
sudo ../isolate_cpus.bash run ./OFDMDemodulator
```
*如果您使用单独的计算机作为前端，请添加 `--default-ip=<your front end IP>`。*

### 3. 将视频流传输到 BS
```bash
ffmpeg -re -stream_loop -1 -fflags +genpts -i video.mp4 -an -c:v libx264 -x264-params keyint=5:min-keyint=1 -b:v 3000k -minrate 3000k -maxrate 3000k -bufsize 1M -f rtp -sdp_file video.sdp "rtp://<your IP of the BS>:50000"
```
*如果您在BS本地传输视频流，BS 的 IP 可以设置为 127.0.0.1。*

### 4. 从 UE 播放视频
将 `video.sdp` 复制到视频接收端，将 `m=video 50000 RTP/AVP 96` 修改为 `m=video 50001 RTP/AVP 96`。
```bash
ffplay -protocol_whitelist file,rtp,udp -i video1.sdp
```
*注意：此命令应在前端运行。*

### 5. 运行单站感知前端
```bash
python3 ./plot_sensing.py
```

### 6. 运行双站感知前端
```bash
python3 ./plot_bi_sensing.py
```

### OFDM 调制器 (OFDMModulator)

`OFDMModulator` (BS 节点) 可以使用以下命令行参数配置:

| 参数 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `--args` | `addr=192.168.40.2, master_clock_rate=200e6, num_recv_frames=512, num_send_frames=512` | USRP 设备参数 |
| `--fft-size` | `1024` | FFT 大小 |
| `--cp-length` | `128` | 循环前缀 (CP) 长度 |
| `--sync-pos` | `1` | 同步符号位置索引 |
| `--sample-rate` | `50e6` | 采样率 (Hz) |
| `--bandwidth` | `50e6` | 模拟带宽 (Hz) |
| `--center-freq` | `2.4e9` | 中心频率 (Hz) |
| `--tx-gain` | `20` | 发送增益 (dB) |
| `--rx-gain` | `30` | 接收增益 (dB) |
| `--rx-channel` | `1` | USRP 上的 RX 通道索引 |
| `--zc-root` | `29` | Zadoff-Chu 序列根 |
| `--num-symbols` | `100` | 每帧 OFDM 符号数 |
| `--clock-source` | `external` | 时钟源 (`internal` 或 `external`) |
| `--system-delay` | `63` | 用于对齐的系统延迟样本数 |
| `--wire-format-tx` | `sc16` | USRP TX 传输格式 (`sc8` 或 `sc16`) |
| `--wire-format-rx` | `sc16` | USRP RX 传输格式 (`sc8` 或 `sc16`) |
| `--mod-udp-ip` | `0.0.0.0` | 接收 UDP 数据负载的绑定 IP|
| `--mod-udp-port` | `50000` | 接收 UDP 数据负载的绑定端口 |
| `--sensing-ip` | `127.0.0.1` | 感知数据的目标 IP |
| `--sensing-port` | `8888` | 感知数据的目标端口 |
| `--default-ip` | `127.0.0.1` | 所有服务的默认 IP (Python前端的IP) |
| `--cpu-cores` | `0,1,2,3,4,5` | 要使用的 CPU 核心列表 |

### OFDM 解调器 (OFDMDemodulator)

`OFDMDemodulator` (UE 节点) 支持以下参数:

| 参数 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `--device-args` | `num_recv_frames=512, num_send_frames=512, send_frame_size=11520, recv_frame_size=11520` | USRP 设备参数 |
| `--fft-size` | `1024` | FFT 大小 |
| `--cp-length` | `128` | 循环前缀 (CP) 长度 |
| `--center-freq` | `2.4e9` | 中心频率 (Hz) |
| `--sample-rate` | `50e6` | 采样率 (Hz) |
| `--bandwidth` | `50e6` | 模拟带宽 (Hz) |
| `--rx-gain` | `60` | 接收增益 (dB) |
| `--rx-channel` | `0` | RX 通道索引 |
| `--sync-pos` | `1` | 同步符号位置索引 |
| `--sensing-ip` | `127.0.0.1` | 发送感知数据的 IP |
| `--sensing-port` | `8889` | 发送感知数据的端口 |
| `--control-port` | `9999` | 接收控制命令的端口 (Heartbeat, MTI 等) |
| `--channel-ip` | `127.0.0.1` | 信道估计输出的 IP |
| `--channel-port` | `12348` | 信道估计输出的端口 |
| `--pdf-ip` | `127.0.0.1` | 原始功率延迟分布 (PDF) 数据输出的 IP |
| `--pdf-port` | `12349` | 原始功率延迟分布 (PDF) 数据输出的端口 |
| `--constellation-ip` | `127.0.0.1` | 星座图数据的 IP |
| `--constellation-port` | `12346` | 星座图数据的端口 |
| `--freq-offset-ip` | `127.0.0.1` | 频率偏移数据的 IP |
| `--freq-offset-port` | `12347` | 频率偏移数据的端口 |
| `--udp-output-ip` | `127.0.0.1` | 解码用户数据的目标 IP (用于播放视频的电脑IP) |
| `--udp-output-port` | `50001` | 解码用户数据的目标端口 |
| `--zc-root` | `29` | Zadoff-Chu 序列根 |
| `--num-symbols` | `100` | 每帧符号数 |
| `--sensing-symbol-num`| `100` | 用于感知处理的符号数 |
| `--clock-source` | `external` | 时钟源 (`internal` 或 `external`) |
| `--software-sync` | `true` | 启用软件同步/跟踪 |
| `--hardware-sync` | `false` | 启用硬件同步 (禁用软件同步) |
| `--hardware-sync-tty` | `/dev/ttyUSB0` | 用于硬件同步的 TTY 设备 |
| `--wire-format-rx` | `sc16` | USRP RX 传输格式 |
| `--default-ip` | `127.0.0.1` | 所有服务的默认 IP (Python前端的IP) |
| `--cpu-cores` | `0,1,2,3,4,5` | 要使用的 CPU 核心列表 |
