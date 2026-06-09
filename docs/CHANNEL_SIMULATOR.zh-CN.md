# 信道仿真器 —— 无需 USRP 运行 OpenISAC

> 英文版见 [CHANNEL_SIMULATOR.md](CHANNEL_SIMULATOR.md)。


`ChannelSimulator` 让你在没有任何射频硬件的情况下运行完整的**通信**与**多通道感知**流水线。
它用一块共享内存"空口"替代 USRP，并施加可配置的信道模型：具有时延 / 多普勒 / 增益 /
导向矢量的点目标、加性高斯白噪声 AWGN、载波频偏 CFO、定时偏移，以及用于直射/LoS 路径的
通信抽头延迟线。

每个感知 RX 通道被视为一根天线；导向矢量既可以由均匀线阵（ULA，参数 `array_spacing_lambda`）
参数化生成，也可以通过 `steering_override_file` 加载完全自定义的矩阵。通信信道是一个**真实的
多径信道**：直射（LoS）抽头加上同一组运动目标反射，因此解调器的**双基地**感知同样能检测到
距离-多普勒目标。

## 工作原理

```
 OFDMModulator ──TX 采样──▶ ChannelSimulator ──各天线 RX──▶ OFDMDemodulator (通信 RX)
 (SimTxStreamer)            ("空口" + 采样时钟)  └─感知 RX×N──▶ OFDMModulator 单基地感知
```

三个进程通过以 `simulation.session` 命名的 POSIX 共享内存环形缓冲区相互连接。
各引擎通过 YAML 中的 `radio_backend: sim` 选择后端；由于仿真 streamer 实现了抽象的
`uhd::tx_streamer` / `uhd::rx_streamer` 接口，热路径上的 `send()/recv()` 无需任何改动。
`radio_backend: uhd`（默认值）保持原有的硬件行为不变。

## 编译

```
cd build && cmake .. && make -j ChannelSimulator OFDMModulator OFDMDemodulator
```

## 运行（同一台机器，三个终端）

```
cd build
cp ../config/Modulator_Sim.yaml   Modulator.yaml
cp ../config/Demodulator_Sim.yaml Demodulator.yaml

# 1) 先启动"空口"（它创建共享内存，并充当采样时钟）
./ChannelSimulator                 # 读取 Modulator.yaml（及其中的 simulation: 块）

# 2) 启动发射机 / 单基地感知
./OFDMModulator

# 3) 启动接收机 / 通信解调
./OFDMDemodulator
```

务必**先启动** `ChannelSimulator`。默认情况下 hub 通过背压（backpressure）流控，因此某条路径
若其消费者缺席，会使整个仿真暂停 —— 请运行所有已启用的接收端。若只想运行其中一侧，用下面的
开关关闭另一侧（此时 hub 既不创建也不产生该路径，其消费者也就无需运行）：

- **仅感知** —— 设 `enable_comm_rx: false`；运行 `ChannelSimulator` + `OFDMModulator`（不运行解调器）。
- **仅通信** —— 设 `enable_sensing_rx: false`；三个进程都运行（调制器不创建任何感知通道）。

也可以用旧方式关闭感知：将 `sensing_rx_channel_count` 设为 0。

用 `python3 scripts/plot_sensing_fast.py` 可视化感知结果（与硬件一样，RD 数据流由查看器触发启动）。

## 配置信道（Modulator_Sim.yaml 中的 `simulation:` 块）

```yaml
radio_backend: sim
simulation:
  session: oisac_sim             # 三个进程必须保持一致
  enable_comm_rx: true           # 生成通信 RX 路径（false = 仅感知，不运行解调器）
  enable_sensing_rx: true        # 生成感知 RX 路径（false = 仅通信）
  noise_power_dbfs: -50          # 每个 RX 通道的 AWGN 功率；<= -200 表示关闭
  cfo_hz: 0.0                    # 施加在 RX 上的载波频偏
  timing_offset_samples: 0       # 固定的 RX 采样时延
  array_spacing_m: 0.04283       # ULA 物理阵元间距（米）；电气间距 d/λ 随 center_freq 变化
                                 # （42.83 mm = 3.5 GHz 时的 λ/2，与感知可视化端一致）。
                                 # 设为 <= 0 时回退到旧的 array_spacing_lambda。
  array_spacing_lambda: 0.5      # 旧参数：固定波长间距，仅当 array_spacing_m <= 0 时使用
  ring_capacity_samples: 262144  # 约 2 帧；取小值可让 TX 紧贴 hub
  steering_override_file: ""     # [目标数 x 通道数] complex<float>，行优先；为空则用 ULA
  comm_multipath_taps:           # 通信直射/LoS + 静态多径（可解调分量）
    - { delay_samples: 0, gain_db: 0, phase_deg: 0 }
  targets:                       # 单基地感知散射体（带导向矢量）
    - { range_m: 30, velocity_mps: 5,  gain_db: -6,  angle_deg: 20 }
    - { range_m: 75, velocity_mps: -3, gain_db: -12, angle_deg: -10 }
  # bistatic_targets:            # 可选：双基地（通信）信道的独立场景
  #   - { range_m: 45, velocity_mps: 8, gain_db: -8, angle_deg: 0 }
```

感知天线数量等于 `sensing_rx_channels` 条目的数量。

`targets` 馈入单基地感知天线（带导向矢量）。双基地（通信）信道在 `bistatic_targets` 非空时
使用它，否则回退到 `targets`，因此默认情况下同一个场景同时驱动两种视角。`angle_deg` 在双基地
信道上被忽略（通信 RX 为单天线）。
自定义导向矢量文件为 `目标数 × 通道数` 个小端 `complex<float>`，按行优先（目标在外层）排列。

## 信道模型

- 目标往返时延 `τ = 2·range_m/c`，采样时延 `round(τ·fs) + timing_offset`。
- 多普勒 `fd = 2·velocity_mps/λ`，其中 `λ = c/center_freq`。
- ULA 导向矢量 `a_k(θ) = exp(j·2π·(d/λ)·k·sinθ)`，`k` 为天线索引；电气间距
  `d/λ = array_spacing_m·center_freq/c` 由物理间距与载频共同决定（因此任意 `center_freq`
  下还原出的角度都正确，可视化端用同一物理间距反算角度）。当 `array_spacing_m <= 0` 时退回到
  与频率无关的旧参数 `array_spacing_lambda`。
- 单基地感知 RX（第 `k` 根天线）：
  `rx_sens_k[n] = Σ_targets gain·a_k(θ)·tx[n−τ]·e^{j2π fd n/fs} + AWGN`。
- 通信 / 双基地 RX（单天线）：一个可解调的直射路径，**外加**相同的运动目标反射，
  从而让解调器的双基地感知拥有可检测的距离-多普勒目标：
  `rx_comm[n] = (Σ_taps gain·tx[n−delay] + Σ_targets gain·tx[n−τ]·e^{j2π fd n/fs})·e^{j2π cfo n/fs} + AWGN`。

默认情况下，`targets` 列表同时驱动单基地感知天线（带导向矢量）和双基地通信信道（不带导向矢量），
对应同一个相干场景。设置 `bistatic_targets` 可让双基地/通信信道拥有独立的散射体。请让直射路径
`comm_multipath_taps` 强于目标（例如 0 dB 对 −6/−12 dB），以保证通信链路仍能同步并解调。

## 说明 / 限制

- 每个进程都能通过 Ctrl-C（SIGINT）干净退出，且彼此独立；hub 退出时会清理（unlink）其共享内存。
- 非实时：流控由共享内存背压完成（以正确性优先于速度）。
- 定时（timed）TX 突发被近似为连续流；RX 同步会自行恢复帧边界。
- 将 `ring_capacity_samples` 设小（约几帧），使发射机不会远远跑在 hub 前面，
  从而避免单基地 TX/RX 配对队列溢出。
