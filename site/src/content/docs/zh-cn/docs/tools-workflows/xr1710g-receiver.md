---
title: XR1710G 接收 UE
description: 使用 XR1710G 接收 OpenISAC 下行信号，配置数据源、UE 和可选的硬件频偏校正。
---

XR1710G 可以替代 UE 侧的 USRP，接收 BS 发出的 OFDM 信号并输出解码后的业务数据。BS 仍使用 USRP，按原有方式发送。

**XR1710G 接入目前仍处于测试阶段，不建议用于正式实验。** 本页以单路接收、40 MS/s 和 2.442 GHz 为例，说明从数据源配置到 UE 接收的完整流程。

## 接入条件

- XR1710G 已按 [xr1710g-sdr 的 XR1710G 安装说明](https://github.com/zhouzhiwen2000/xr1710g-sdr/blob/main/platform/xr1710g/README.md)完成设备准备，接收电脑能够控制设备并接收 IQ 数据。
- XR1710G 通过万兆网络连接 Linux 接收电脑。`w1700k-stream` 和 OpenISAC `UE` 必须运行在**同一台电脑**，两者通过共享内存传递采样数据。
- 接收电脑已完成 OpenISAC 的[软件安装](/zh-cn/docs/getting-started/installation/)。以下假定两个仓库分别位于 `~/xr1710g-sdr` 和 `~/OpenISAC`。
- 当前接入方式使用 CPU `UE` 程序，只支持单路接收，不支持 UE 上行发送、ARQ 反馈或双路交织数据。

可选择的源采样率为 **40、80、160 或 320 MS/s**。XR1710G、BS 和 `UE.yaml` 中的采样率必须一致；OpenISAC 不会自动重采样。首次接入可按本页使用 40 MS/s。已有的 50 或 100 MS/s 配置需要先调整，并确认 BS 硬件也支持所选速率。

## 数据路径

`USRP BS → 无线信号 → XR1710G → 万兆网络 → w1700k-stream → OpenISAC UE → UDP 接收应用`

三个程序分别负责不同操作：

| 程序 | 运行位置 | 作用 |
| :--- | :--- | :--- |
| `w1700k` | Linux 接收电脑 | 配置 XR1710G 的频率、采样率、滤波器和增益，启动或停止设备数据流 |
| `w1700k-stream` | Linux 接收电脑 | 接收设备发来的 IQ 数据，供本机 UE 和频谱程序读取 |
| `UE` | 同一台 Linux 接收电脑 | 同步、解调和解码，将业务数据发送到配置的 UDP 地址 |

以下步骤中，接收电脑需要保留多个终端：一个运行 `w1700k-stream`，一个用于设备控制和配置，另一个运行 `UE`。

## 1. 启动主机接收程序

**操作位置：Linux 接收电脑，终端 1。**

如果已有 `w1700k-stream` 正在运行，且共享内存名称为 `/xr1710g-live`，可直接复用。否则执行：

```bash
cd ~/xr1710g-sdr
make -C host/stream
host/stream/build/w1700k-stream --name /xr1710g-live --port 7700
```

保持此终端运行。`--name` 指定 UE 要读取的共享内存名称，`--port` 必须与设备的数据流发送配置一致。

## 2. 配置 XR1710G 并启动数据流

**操作位置：Linux 接收电脑，终端 2。**

将下方的 `/path/to/unit-config.json` 替换为安装 XR1710G 时创建的设备配置文件。该文件供 `w1700k` 连接和控制设备使用，与后续的 `UE.yaml` 是两个不同文件。

如果频谱控制程序正在管理设备，先停止它的射频控制，再执行以下命令：

```bash
cd ~/xr1710g-sdr
host/bin/w1700k --config /path/to/unit-config.json stop
host/bin/w1700k --config /path/to/unit-config.json band 0 --rate 40M --lo 2442M
host/bin/w1700k --config /path/to/unit-config.json filter 40
host/bin/w1700k --config /path/to/unit-config.json gain --agc
host/bin/w1700k --config /path/to/unit-config.json start
host/bin/w1700k --config /path/to/unit-config.json status
```

该示例选择 band 0、40 MS/s、约 2.442 GHz 的接收中心频率、40 MHz 模拟滤波器和自动增益。记录 `status` 返回的实际本振频率 `radio.lo_hz`，后续 UE 配置应填写这个读回值。实际频率可能因硬件调谐步进而与请求值略有差异。

确认终端 1 已持续收到样本，再进行下一步。接收过程中保持源采样率不变。射频增益和滤波器由 XR1710G 控制工具设置，不使用 USRP 的增益参数代替。

## 3. 编译 UE 并复制配置

**操作位置：Linux 接收电脑，终端 2。**

```bash
cd ~/OpenISAC
cmake -S . -B build-xr1710g -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=NOTFOUND
cmake --build build-xr1710g --target UE Xr1710gRxProbe -j4
cp config/UE_XR1710G.yaml build-xr1710g/UE.yaml
```

编辑 `build-xr1710g/UE.yaml`。保留预设的完整配置，并核对以下字段：

| 字段 | 本页示例 | 填写方法 |
| :--- | :--- | :--- |
| `radio.radio_backend` | `xr1710g` | 选择 XR1710G 接收 |
| `radio.xr1710g_shm` | `/xr1710g-live` | 与第 1 步的 `--name` 一致 |
| `radio.xr1710g_center_freq` | `2442000000` | 填写第 2 步读回的实际 `radio.lo_hz`，单位 Hz；此字段本身不会调谐硬件 |
| `radio.xr1710g_pll_command` | `""` | 首次接入保持为空，使用软件频偏校正 |
| `rf_sampling.sample_rate` | `40000000` | 与 XR1710G 和 BS 的采样率一致，单位 samples/s |
| `rf_sampling.bandwidth` | `40000000` | 记录本例接收带宽；硬件滤波器仍在第 2 步设置 |
| `downlink.center_freq` | `2442000000` | BS 的发射中心频率，单位 Hz |
| `network_output.udp_output_ip` | `127.0.0.1` | 解码数据接收应用的 IP 地址 |
| `network_output.udp_output_port` | `50001` | 解码数据接收应用的 UDP 端口 |

`xr1710g_center_freq` 描述接收机实际调到了哪里，`downlink.center_freq` 描述 BS 在哪里发射，两者不应混淆。先将 XR1710G 调到 BS 载频附近，再由 UE 跟踪和校正剩余频偏。

预设已关闭上行、ARQ 反馈、硬件同步和预测时延。保持这些设置；当前接入方式不提供与 BS 对齐的硬件时间，也不提供自动采样时钟校正。频偏跟踪仍然启用。

## 4. 配置并启动 BS

**操作位置：BS 电脑。**

在 BS 运行目录的 `BS.yaml` 中，将采样率设为 `40000000`，发射中心频率设为 `2442000000`。BS 继续使用 UHD 后端。

对照两端 YAML，核对以下配置：

- 采样率、FFT 大小、CP 长度和每帧符号数。
- 同步符号位置、ZC 根、重复同步符号和 CFO 训练字段开关。
- 导频位置、帧中导频配置及数据资源分配。

`UE_XR1710G.yaml` 是起始预设，不会自动匹配现有 BS。两端参数一致后，从已编辑 `BS.yaml` 的目录启动 BS，并保持发射。以下假定运行目录为 `~/OpenISAC/build`：

```bash
cd ~/OpenISAC/build
sudo ../scripts/isolate_cpus.py run ./BS
```

## 5. 检查样本并启动 UE

**操作位置：Linux 接收电脑，终端 3。**

先运行接收检查程序。下方参数依次为共享内存名称、采样率、实际接收本振频率和检查时长（秒）；请将 `2442000000` 替换为第 2 步的读回值。

```bash
cd ~/OpenISAC/build-xr1710g
./Xr1710gRxProbe /xr1710g-live 40000000 2442000000 5
```

正常情况下，输出中的 `samples` 和 `nonzero_probes` 大于零，`errors`、`timeouts` 和 `overflows` 为零。此检查用于确认样本能持续送达 UE；通信是否正常，还需要看下一步是否收到 BS 发送的业务数据。

```bash
./UE
```

保持 UE 运行，等待同步完成，然后在 BS 端发送业务数据。UE 默认将解码数据输出到本机 `127.0.0.1:50001`。需要视频演示时，按[视频与 UDP 工作流](/zh-cn/docs/tools-workflows/video-udp-workflows/)配置推流和播放，首次使用较低视频码率。

## 6. 停止接收或更换参数

在运行 `UE` 的终端按 **Ctrl+C**，等待程序退出。停止 UE 后，`w1700k-stream` 和 XR1710G 数据流仍会继续运行。

若要停止设备数据流，在控制终端执行：

```bash
cd ~/xr1710g-sdr
host/bin/w1700k --config /path/to/unit-config.json stop
```

再在终端 1 按 Ctrl+C，退出 `w1700k-stream`。更换采样率或初始接收频率时，先停止 UE，再配置设备、更新两端 YAML，并按上述顺序重新启动。

## 可选：使用硬件频偏校正

默认的软件频偏校正在接收电脑上处理 IQ 数据。若需由 XR1710G 的 PLL 微调本振频率，在已能正常接收的基础上，停止 UE，并在现有 `radio` 配置中填写：

```yaml
radio:
  radio_backend: xr1710g
  xr1710g_pll_command: >-
    python3 /home/user/OpenISAC/scripts/xr1710g_pll_control.py
    --sdr-repo /home/user/xr1710g-sdr
    --config /home/user/xr1710g-config.json
```

将三个路径替换为接收电脑上的实际绝对路径。`--config` 使用第 2 步同一份设备配置；`radio.xr1710g_center_freq` 仍填写启动前读回的实际本振频率。保存后重新启动 UE，控制程序会随 UE 启动。

硬件校正范围为初始本振附近 **±100 kHz**，且不能跨越 PLL 分频区间。它用于修正载频偏差，不调整采样时钟。运行期间由 UE 控制本振，其他程序可以读取同一份 IQ，但不能同时改频率、采样率或滤波器。

控制程序正常退出时会恢复启动前的 PLL 设置。若日志报告控制失败、锁定失败或恢复失败，停止 UE，用 `w1700k status` 核对设备状态并重新配置后再启动。将 `xr1710g_pll_command` 清空并重启 UE，可恢复软件频偏校正。

## 常见问题与使用限制

| 现象或需求 | 检查与处理 |
| :--- | :--- |
| UE 找不到数据源或等待超时 | 确认 `w1700k-stream` 持续运行、共享内存名称一致，且它与 UE 位于同一台电脑 |
| 接收检查没有样本 | 检查 XR1710G 是否已启动数据流、万兆链路及数据端口是否正确 |
| 有样本，但 UE 无法同步或解码 | 核对实际接收频率、BS 载频、采样率、帧结构、导频和资源分配，再检查接收信号强度 |
| 持续出现 `overflow` | 检查主机接收程序的 `lost` 计数、万兆网络和电脑处理负载；降低采样率时同时修改数据源和两端配置 |
| 更换采样率后停止接收 | 源采样率与 YAML 不一致时会暂停样本交付；统一参数后重启 UE |
| 数据源重启后短时失步 | UE 会重新捕获同步；若不能恢复，先检查源数据是否已恢复，再重启 UE |
| 需要上行或 ARQ 反馈 | 此接入方式只提供接收，请使用支持收发的 USRP UE 配置 |
| 需要绝对测距或 eRTM 标定 | XR1710G 接收时间戳是本机样本计数，没有与 BS 对齐的硬件 PPS，不适用于这些用途 |

完整参数说明见 [UE YAML 参考](/zh-cn/docs/reference/ue-yaml/)。
