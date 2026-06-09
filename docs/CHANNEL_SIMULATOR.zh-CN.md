# 信道仿真器 —— 无需 USRP 运行 OpenISAC

> 英文版见 [CHANNEL_SIMULATOR.md](CHANNEL_SIMULATOR.md)。


`ChannelSimulator` 让你在没有任何射频硬件的情况下运行完整的**通信**与**多通道感知**流水线。它用一块共享内存"空口"替代 USRP，并在空口中构造一个基于散射体的多径信道：发射信号经过 LoS 路径、可选静态多径路径，以及一组带时延 / 多普勒 / 增益 / 阵列响应的运动散射体；各路径的整数时延中可以并入固定定时偏移，通信/双站接收链路再叠加相对载波频偏（CFO），最后为每个 RX 输出加入 AWGN。这里的定时偏移是整数样本级的固定偏移；当前仿真不建模采样频率偏差（SFO）。

每个感知 RX 通道被视为一根天线；阵列响应既可以由均匀线阵（ULA，参数 `array_spacing_lambda`）参数化生成，也可以通过 `steering_override_file` 加载完全自定义的阵列流形矩阵。通信信道是一个**真实的多径信道**：LoS 路径加上散射体散射，因此解调器的**双站**感知同样能检测到距离-多普勒目标。

## 工作原理

```
 OFDMModulator ──TX 采样──▶ ChannelSimulator ──各天线 RX──▶ OFDMDemodulator (通信 RX)
 (SimTxStreamer)            ("空口" + 采样时钟)  └─感知 RX×N──▶ OFDMModulator 单站感知
```

三个进程通过以 `simulation.session` 命名的 POSIX 共享内存环形缓冲区相互连接。各引擎通过 YAML 中的 `radio_backend: sim` 选择后端；由于仿真 streamer 实现了抽象的 `uhd::tx_streamer` / `uhd::rx_streamer` 接口，热路径上的 `send()/recv()` 无需任何改动。`radio_backend: uhd`（默认值）保持原有的硬件行为不变。

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

# 2) 启动发射机 / 单站感知
./OFDMModulator

# 3) 启动接收机 / 通信解调
./OFDMDemodulator
```

务必**先启动** `ChannelSimulator`。默认情况下 hub 通过背压（backpressure）流控：如果启用了通信 RX 或感知 RX，却没有启动对应的消费者进程读取共享内存，相关环形缓冲区填满后，整个仿真都会等待。因此，请启动所有已启用的接收端；如果只需要其中一侧，请在 `simulation:` 中关闭不用的输出：

- **仅感知** —— 设 `enable_comm_rx: false`；hub 不生成通信 RX 输出，只需运行 `ChannelSimulator` + `OFDMModulator`，无需运行 `OFDMDemodulator`。
- **仅通信** —— 设 `enable_sensing_rx: false`；hub 不生成感知 RX 输出，但仍需运行 `ChannelSimulator` + `OFDMModulator` + `OFDMDemodulator`，因为调制器负责发射，解调器负责通信接收。

关闭感知 RX 有两种方式：设置 `enable_sensing_rx: false`，或将 `sensing_rx_channel_count` 设为 0。

需要查看单站感知结果时，运行 `python3 scripts/plot_sensing_fast.py`。与硬件模式一致，RD 数据流由查看器连接后触发启动。

## 配置信道（Modulator_Sim.yaml 中的 `simulation:` 块）

```yaml
radio_backend: sim
simulation:
  session: oisac_sim             # 三个进程必须保持一致
  enable_comm_rx: true           # 生成通信 RX 路径（false = 仅感知，不运行解调器）
  enable_sensing_rx: true        # 生成感知 RX 路径（false = 仅通信）
  noise_power_dbfs: -50          # 每个 RX 通道的 AWGN 功率；<= -200 表示关闭
  cfo_hz: 0.0                    # 通信/双站 RX 上的相对载波频偏
  timing_offset_samples: 0       # 固定整数采样偏移，并入路径时延
  array_spacing_m: 0.04283       # ULA 物理阵元间距（米）；电气间距 d/λ 随 center_freq 变化
                                 # （42.83 mm = 3.5 GHz 时的 λ/2，与感知可视化端一致）。
                                 # 设为 <= 0 时回退到旧的 array_spacing_lambda。
  array_spacing_lambda: 0.5      # 旧参数：固定波长间距，仅当 array_spacing_m <= 0 时使用
  ring_capacity_samples: 262144  # 约 2 帧；取小值可让 TX 紧贴 hub
  steering_override_file: ""     # 阵列流形矩阵：[目标数 x 通道数] complex<float>，行优先；为空则用 ULA
  comm_multipath_taps:           # 通信 LoS 路径 + 静态多径路径
    - { delay_samples: 0, gain_db: 0, phase_deg: 0 }
  targets:                       # 单站感知散射体（带阵列响应）
    - { range_m: 30, velocity_mps: 5,  gain_db: -6,  angle_deg: 20 }
    - { range_m: 75, velocity_mps: -3, gain_db: -12, angle_deg: -10 }
  # bistatic_targets:            # 可选：双站（通信）信道的独立场景
  #   - { range_m: 45, velocity_mps: 8, gain_db: -8, angle_deg: 0 }
```

感知天线数量等于 `sensing_rx_channels` 条目的数量。

`targets` 描述单站感知看到的散射体场景，并为每个感知天线生成对应的阵列响应。双站（通信）信道可以通过 `bistatic_targets` 单独配置；如果不配置，则复用 `targets`，也就是用同一组散射体同时驱动单站和双站两条链路。由于通信 RX 是单天线，`angle_deg` 不参与双站信道计算。若需要自定义阵列流形，`steering_override_file` 应提供 `目标数 × 通道数` 个小端 `complex<float>`，按行优先存储，每一行对应一个目标。

## 信道模型

模型按下面的顺序作用于发射采样 \(x[n]\)。

1. 对每个散射体计算整数传播时延、多普勒与复增益；单站感知路径还会乘以阵列流形向量。
2. 通信/双站路径先叠加 LoS 路径与静态多径路径，再叠加散射体散射分量。
3. 通信/双站路径对整个接收信号施加相对 CFO；单站感知路径不施加 CFO。
4. 每个 RX 输出最后加入 AWGN。

目标往返时延、采样时延与多普勒为：

$$
\tau_i = \frac{2 R_i}{c}, \qquad
\ell_i = \operatorname{round}(\tau_i f_s) + n_0, \qquad
f_{D,i} = \frac{2 v_i}{\lambda}, \qquad
\lambda = \frac{c}{f_c}
$$

其中 \(R_i\) 对应 `range_m`，\(v_i\) 对应 `velocity_mps`，\(f_s\) 是 `sample_rate`，\(f_c\) 是 `center_freq`，\(n_0\) 是 `timing_offset_samples`。这里的 \(n_0\) 是固定整数样本偏移；当前仿真不建模采样频率偏差（SFO）。

ULA 阵列流形为：

$$
a_{i,k}(\theta_i) =
\exp\!\left(j 2\pi \frac{d}{\lambda} k \sin\theta_i\right)
$$

其中 \(k\) 为天线索引；电气间距

$$
\frac{d}{\lambda} = \frac{\texttt{array\_spacing\_m}\, f_c}{c}
$$

由物理间距与载频共同决定，因此任意 `center_freq` 下还原出的角度都正确，可视化端用同一物理间距反算角度。当 `array_spacing_m <= 0` 时退回到与频率无关的旧参数 `array_spacing_lambda`。如果配置 `steering_override_file`，则直接从阵列流形矩阵读取 \(a_{i,k}\)。

单站感知 RX 的第 \(k\) 根天线为：

$$
y_{\mathrm{mono},k}[n]
= \sum_i g_i\, a_{i,k}(\theta_i)\, x[n-\ell_i]\,
  e^{j 2\pi f_{D,i} n / f_s}
  + w_k[n]
$$

单站感知信道与发射机共享同一个仿真时钟，不叠加相对 CFO；当前仿真不建模采样频率偏差（SFO）。上式中的 \(w_k[n]\) 为 AWGN。

通信/双站 RX 为单天线。LoS/静态多径路径先形成通信多径分量：

$$
u_{\mathrm{LoS}}[n] =
\sum_p h_p\, x[n-\ell_p],
\qquad
\ell_p = d_p + n_0
$$

其中 \(d_p\) 来自 `comm_multipath_taps[].delay_samples`，\(h_p\) 由 `gain_db` 和 `phase_deg` 决定。散射体散射分量叠加到同一个通信信道上：

$$
u[n] =
u_{\mathrm{LoS}}[n]
+ \sum_i g_i\, x[n-\ell_i]\, e^{j 2\pi f_{D,i} n / f_s}
$$

随后通信/双站链路施加相对 CFO，并加入 AWGN：

$$
y_{\mathrm{comm}}[n] =
u[n]\, e^{j 2\pi f_{\mathrm{CFO}} n / f_s}
+ w_{\mathrm{comm}}[n]
$$

默认情况下，`targets` 列表同时驱动单站感知天线（带阵列流形）和双站通信信道（单天线，不使用阵列流形），对应同一个相干场景。设置 `bistatic_targets` 可让双站/通信信道拥有独立的散射体。`comm_multipath_taps` 用于配置 LoS 路径和静态多径路径的时延、增益与初始相位。

## 说明 / 限制

- 每个进程都能通过 Ctrl-C（SIGINT）干净退出，且彼此独立；hub 退出时会清理（unlink）其共享内存。
- 非实时：流控由共享内存背压完成（以正确性优先于速度）。
- 将 `ring_capacity_samples` 设小（约几帧），使发射机不会远远跑在 hub 前面，从而避免单站 TX/RX 配对队列溢出。
- 当前仿真不建模采样频率偏差（SFO）；`timing_offset_samples` 只是固定整数样本偏移。
- 当前仿真没有建模分数时延；所有传播时延与配置定时偏移都会取整到样本级。
