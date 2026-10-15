---
title: 硬件准备
description: OpenISAC 实验所需的硬件角色和基础准备。
---

## 后端 (C++)

要设置完整的系统，您需要以下硬件：

*   **USRP 设备**: 2 台 (例如 USRP X310, B210 等)
*   **计算机**: 2 台 (推荐使用高主频CPU)
*   **天线**: 仅下行运行需要 3 根；若开启双工/上行，需要 5 根。目前还不支持 BS 端双工接收和感知接收共通道。
*   **OCXO/GPSDO**: 2 个 (两台 USRP 都需要)

### 连接设置

该系统由两个主要节点组成：

1.  **基站 (BS) 节点**
    *   **硬件**: 1x 计算机, 1x USRP。
    *   **天线**: 仅下行运行时连接 2 根天线到此 USRP (1 路下行 TX，1 路感知 RX)。若开启双工/上行，需要额外连接独立的 BS 上行 RX 天线/RF 链路；目前不能与感知 RX 共通道。
    *   **时钟**: 将 OCXO 或 GPSDO 连接到 USRP 的 REFIN 端口。
    *   **功能**: 发送 OFDM 信号并接收雷达回波。

2.  **用户 (UE) 节点**
    *   **硬件**: 1x 计算机, 1x USRP。
    *   **天线**: 仅下行运行时连接 1 根天线到此 USRP 的 RX 端口。若开启双工/上行，还需要连接 UE TX 天线/RF 链路；FDD 模式还要求设备支持配置的上行载波。
    *   **时钟**: 将 OCXO 或 GPSDO 连接到 USRP 的 REFIN 端口。
    *   **高精度 DAC (可选)**: 使用高精度 DAC 来微调 OCXO。
    *   **功能**: 接收 OFDM 信号用于通信和双站感知；开启双工/上行后还会发送 UE->BS 上行业务。

### 接口要求
为了支持高带宽和采样率，请确保计算机和 USRP 之间的连接使用：
*   **>= 10 Gigabit Ethernet (10GbE)** (X 系列)
*   **USB 3.0** (B 系列)

## 前端 (Python)
*   **计算机**: 1x 计算机 (Windows 或 Linux)。
    *   可以是后端计算机之一，也可以是单独的机器。
*   **CPU**: 如果没有 GPU，建议使用高性能 CPU (i7 10700 或更高)。
*   **GPU**: 建议使用 Nvidia GPU 进行加速。

## 使用 XR1710G 作为纯接收 UE

运行 [`xr1710g-sdr`](https://github.com/zhouzhiwen2000/xr1710g-sdr) 的 XR1710G
可以通过 `radio.radio_backend: xr1710g` 接入 CPU 版 `UE`。UE 在**同一台 Linux
接收电脑**上读取现有 `w1700k-stream` POSIX 共享内存。外部接收程序负责万兆数据接入、
重排和解包，新后端将有符号 14 位 IQ 转为 FC32，默认使用软件频偏校正。支持 AVX2/FMA 的
CPU 原生构建将转换、归一化和 NCO 混频合并在一个 SIMD 循环中，其他 CPU 使用可移植实现。
同步、信道估计、
LDPC 译码和 UDP 输出沿用现有 UE 流程。

此后端支持单路 **40、80、160 或 320 MS/s**，不做重采样。源采样率和 BS 采样率必须与
`rf_sampling.sample_rate` 一致。实验性的双路交织格式暂不支持。
初始射频 LO、模拟滤波器、增益和数据流启动由 XR1710G 控制工具配置。
UE 的启动和停止只控制自己的读取游标，频谱程序可以继续读取同一单路数据源。

载频校正默认使用软件 NCO。下方可选的 PLL 控制程序通过硬件 LO 微调实现同一个后端
调频接口，沿用 UE 已有的调节顺序和过时观测过滤逻辑。硬件模式只将 IQ 转为 FC32，
不执行软件混频。

此后端不支持自动晶振调节；请使用软件 NCO 或 PLL 载频校正，并沿用 UE 的同步跟踪。

### 硬件载频校正

在 **Linux 接收电脑**的 UE YAML 中设置以下命令，将路径替换为本机仓库和已有路由器
配置文件的绝对路径：

```yaml
radio:
  radio_backend: xr1710g
  xr1710g_pll_command: >-
    python3 /home/user/OpenISAC/scripts/xr1710g_pll_control.py
    --sdr-repo /home/user/xr1710g-sdr
    --config /home/user/xr1710g-config.json
```

`xr1710g_center_freq` 仍填写初始源实际 LO。后端启动一个常驻控制进程，同步应用
±100 kHz 内的总物理频偏；正值提高 LO。UHD RX 的有效中心频率为 RF − DSP，
因此手动 DSP 请求与物理 LO 偏移符号相反。接口将量化后的物理偏移取负后返回到
`TuneResult.actual_dsp_freq`，`actual_rf_freq` 保持初始 LO，兼容 UE 原有的累积和复位
语义。这是接口的逻辑拆分，实际频移发生在 PLL 中。band 0 的步进约 2.98 Hz，band 1/2
随 LO 区间约为 11.92 或 14.31 Hz；微调不允许跨越 PLL 分频比切换点。

控制程序检查数据源身份、寄存器读回和 PLL 锁定状态。运行期间由此后端独占射频配置；
频谱程序可以继续读 IQ，但不能同时调谐或重新配置数据源。检测到外部改动或控制错误时，
调频接口报错。控制进程正常退出时恢复保存的分频字，恢复失败会输出到标准错误日志。

调节时保留读取游标和缓冲样本，不清空 IQ、不注入流中断、不重置时间戳。硬件调谐不保证
相位连续。PLL 微调不校正采样时钟。
命令留空时继续使用软件校正。

### 启动接收

1. 在连接 XR1710G 的接收电脑上，按 XR1710G 仓库说明准备可工作的接收链路。
   如果主机接收服务已运行且共享内存名称匹配，直接复用。否则在单独终端编译并运行：

   ```bash
   cd ~/xr1710g-sdr
   make -C host/stream
   host/stream/build/w1700k-stream --name /xr1710g-live --port 7700
   ```

2. 使用现有控制程序配置射频。同一时刻保留一个射频控制方；使用 CLI 前先停止占用控制权的
   频谱控制程序。以 40 MS/s、2.442 GHz 为例：

   ```bash
   cd ~/xr1710g-sdr
   host/bin/w1700k --config /path/to/unit-config.json stop
   host/bin/w1700k --config /path/to/unit-config.json band 0 --rate 40M --lo 2442M
   host/bin/w1700k --config /path/to/unit-config.json filter 40
   host/bin/w1700k --config /path/to/unit-config.json gain --agc
   host/bin/w1700k --config /path/to/unit-config.json start
   host/bin/w1700k --config /path/to/unit-config.json status
   ```

   记录返回的**实际** `radio.lo_hz`。UE 运行期间保持采样率固定，已启用的 PLL 校正由后端控制。
   共享内存 ABI 不含 LO 或天线交织布局标识，因此源 LO 由配置明确声明。
   后端在连接和接收期间校验非零的源采样率字段；字段为零时，依赖配置声明的采样率。

3. 在同一电脑上编译 CPU 接收机，将预设复制到运行目录：

   ```bash
   cd ~/OpenISAC
   cmake -S . -B build-xr1710g
   cmake --build build-xr1710g --target UE Xr1710gRxProbe -j4
   cp config/UE_XR1710G.yaml build-xr1710g/UE.yaml
   ```

   将 `radio.xr1710g_center_freq` 设置为源实际 LO，`downlink.center_freq` 设置为 BS 载频。
   两端的采样率、FFT/CP、前导、导频和资源分配必须匹配。预设为 40 MS/s；原有 50 或
   100 MS/s BS 配置需要改成支持的采样率。USRP BS 继续使用原有 UHD 后端。

4. 先检查传输，再从运行目录启动 UE。探针命令中的 LO 应替换为实际读回值：

   ```bash
   cd ~/OpenISAC/build-xr1710g
   ./Xr1710gRxProbe /xr1710g-live 40000000 2442000000 5
   ./UE
   ```

   探针应报告非零样本，且错误和超时均为零。UE 随后应进入同步，并将解码净荷输出到配置的
   UDP 目的端口，预设为 `127.0.0.1:50001`。探针通过表示样本传输正常；收到匹配 BS 的
   解码净荷才能确认通信链路正常。

### 时间戳与恢复

时间戳是按样本计数的**接收机局部时间线**，没有硬件 PPS 或与 BS 的时间对齐。
固件事件不占样本时间；丢包和覆盖报告 overflow，源 run 变化或共享内存重建报告
broken-chain。跨越断点的读取被丢弃，恢复的数据沿用 UE 已有的上行关闭时失步重捕路径。
覆盖期间未观测到的事件使间隔时长不确定，因此不能将这些时间戳用于绝对测距或 eRTM 标定。

预设关闭上行/ARQ 反馈、硬件时钟控制和预测时延，保留 CFO 跟踪。
XR1710G 的射频频偏与采样时钟偏差关系尚未标定，不能直接用于 CFO 推算 SFO。
硬件增益由外部控制器配置，不按 USRP 的 dB 增益解释。
运行中发现源采样率不匹配时，暂停样本交付，直到声明的采样率恢复；切换采样率应同步修改
数据源与 YAML 后重启 UE。
