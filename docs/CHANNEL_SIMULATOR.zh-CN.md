# 信道仿真器 —— 无需 USRP 运行 OpenISAC

> 英文版见 [CHANNEL_SIMULATOR.md](CHANNEL_SIMULATOR.md)。


`ChannelSimulator` 让你在没有任何射频硬件的情况下运行完整的**通信**与**多通道感知**流水线。它用一块共享内存"空口"替代 USRP，并在空口中构造一个基于散射体的多径信道：发射信号经过 LoS 路径、可选静态多径路径，以及一组带时延 / 多普勒 / 增益 / 阵列响应的运动散射体；各路径的整数时延中可以并入固定定时偏移，通信/双站接收链路可以叠加 UE 采样时钟偏差，再叠加相对载波频偏（CFO），最后为每个 RX 输出加入 AWGN。这里的定时偏移是整数样本级的固定偏移；`sample_rate_offset_ppm` 用于建模 BS 与 UE 采样时钟之间的采样频率偏差（SFO）。

每个感知 RX 通道被视为一根天线；阵列响应既可以由均匀线阵（ULA，参数 `array_spacing_lambda`）参数化生成，也可以通过 `steering_override_file` 加载完全自定义的阵列流形矩阵。通信信道是一个**真实的多径信道**：LoS 路径加上散射体散射，因此UE的**双站**感知同样能检测到距离-多普勒目标。

## 工作原理

```
 BS ──TX 采样──▶ ChannelSimulator ──各天线 RX──▶ UE (通信 RX)
 (radio::SimBackend)        ("空口" + 采样时钟)  └─感知 RX×N──▶ BS 单站感知
```

三个进程通过以 `simulation.session` 命名的 POSIX 共享内存环形缓冲区相互连接。各引擎通过 YAML 中的 `radio_backend: sim` 选择后端；`radio::SimBackend` 实现与 UHD 后端相同的 OpenISAC radio HAL，因此 BS/UE/感知热路径始终驱动 `radio::ITxStream` / `radio::IRxStream` 接口，而不关心底层是共享内存还是 USRP。`radio_backend: uhd`（默认值）保持硬件行为不变。

## 编译

```
cd build && cmake .. && make -j ChannelSimulator BS UE
```

## 运行（同一台机器，三个终端）

```
cd build
cp ../config/BS_Sim.yaml   BS.yaml
cp ../config/UE_Sim.yaml UE.yaml

# 1) 先启动"空口"（它创建共享内存，并充当采样时钟）
./ChannelSimulator                 # 读取 BS.yaml（及其中的 simulation: 块）

# 2) 启动发射机 / 单站感知
./BS

# 3) 启动接收机 / 通信解调
./UE
```

务必**先启动** `ChannelSimulator`。默认情况下 hub 按 wall-clock 采样时间做 pacing，同时仍通过共享内存背压（backpressure）流控：如果启用了通信 RX 或感知 RX，却没有启动对应的消费者进程读取共享内存，相关环形缓冲区填满后，整个仿真都会等待。因此，请启动所有已启用的接收端；如果只需要其中一侧，请在 `simulation:` 中关闭不用的输出：

- **仅感知** —— 设 `enable_comm_rx: false`；hub 不生成通信 RX 输出，只需运行 `ChannelSimulator` + `BS`，无需运行 `UE`。
- **仅通信** —— 设 `enable_sensing_rx: false`；hub 不生成感知 RX 输出，但仍需运行 `ChannelSimulator` + `BS` + `UE`，因为BS负责发射，UE负责通信接收。

关闭感知 RX 有两种方式：设置 `enable_sensing_rx: false`，或将 `sensing_rx_channel_count` 设为 0。

需要查看单站感知结果时，运行 `python3 scripts/plot_sensing_fast.py`。与硬件模式一致，RD 数据流由查看器连接后触发启动。

## 配置信道（BS_Sim.yaml 中的 `simulation:` 块）

```yaml
radio_backend: sim
simulation:
  session: oisac_sim             # 三个进程必须保持一致
  enable_comm_rx: true           # 生成通信 RX 路径（false = 仅感知，不运行UE）
  enable_sensing_rx: true        # 生成感知 RX 路径（false = 仅通信）
  enable_uplink: false           # 将 UE ul.tx 路由到 BS rx.ul；BS_Sim_Duplex.yaml 中为 true
  pacing_enabled: true           # 按 wall-clock 采样时间释放共享内存输出
  noise_power_dbfs: -50          # 每个 RX 通道的 AWGN 功率；<= -200 表示关闭
  cfo_hz: 0.0                    # BS->UE 初始 CFO；UE->BS 反号/按载频比缩放并跟随 TX retune
  sample_rate_offset_ppm: 0.0    # UE 采样时钟相对 BS 时钟的 ppm 偏差
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

`sample_rate_offset_ppm` 建模一个端点级采样时钟误差：UE 采样率为
`BS_rate * (1 + ppm * 1e-6)`。BS TX、BS 上行 RX 与所有 BS 单站感知 RX
通道保持在同一个 BS 采样时钟上；UE 下行 RX 与 UE 上行 TX 保持在同一个
UE 采样时钟上。因此仿真器会按该 ratio 重采样 BS->UE 通信路径，并在
UE->BS 上行路径上使用 reciprocal ratio；单站 BS 感知通道不重采样。

`targets` 描述单站感知看到的散射体场景，并为每个感知天线生成对应的阵列响应。双站（通信）信道可以通过 `bistatic_targets` 单独配置；如果不配置，则复用 `targets`，也就是用同一组散射体同时驱动单站和双站两条链路。由于通信 RX 是单天线，`angle_deg` 不参与双站信道计算。若需要自定义阵列流形，`steering_override_file` 应提供 `目标数 × 通道数` 个小端 `complex<float>`，按行优先存储，每一行对应一个目标。配置自定义阵列流形矩阵后，`targets[].angle_deg` 不再参与单站阵列响应计算；用户需要根据每个目标的角度，预先计算好各阵元的幅度和相位并写入矩阵。

启用 `simulation.enable_uplink` 后，UE->BS 仿真上行复用与 BS->UE 下行相同的通信信道场景。在该场景中，`comm_multipath_taps` 配置的静态多径分量会与选定的双站散射体分量一起与直达径叠加；双站散射体优先取自 `bistatic_targets`，未配置时复用 `targets`。

## 信道模型

模型按下面的顺序作用于发射采样 $x[n]$。

1. 对每个散射体计算整数传播时延、多普勒与复增益；单站感知路径还会乘以阵列流形向量。
2. 通信/双站路径先叠加 LoS 路径与静态多径路径，再叠加散射体散射分量。
3. BS->UE 通信路径按 `1 + sample_rate_offset_ppm * 1e-6` 重采样；启用上行仿真时，UE->BS 路径按 reciprocal ratio 重采样。
4. 通信/双站路径对整个接收信号施加相对 CFO：BS→UE 下行使用 `cfo_hz`，UE→BS 上行使用其相反数；单站感知路径不施加 CFO。
5. 每个 RX 输出最后加入 AWGN。

目标往返时延、采样时延与多普勒为：

$$
\tau_i = \frac{2 R_i}{c}, \qquad \ell_i = \mathrm{round}(\tau_i f_s) + n_0, \qquad f_{D,i} = \frac{2 v_i}{\lambda}, \qquad \lambda = \frac{c}{f_c}
$$

其中 $R_i$ 对应 `range_m`，$v_i$ 对应 `velocity_mps`，$f_s$ 是 BS `sample_rate`，$f_c$ 是 `center_freq`，$n_0$ 是 `timing_offset_samples`。这里的 $n_0$ 是固定整数样本偏移；SFO 由通信路径重采样器单独建模。

ULA 阵列流形为：

$$
a_{i,k}(\theta_i) = \exp\!\left(j 2\pi \frac{d}{\lambda} k \sin\theta_i\right)
$$

其中 $k$ 为天线索引；电气间距

$$
d_\lambda = \frac{d_m f_c}{c}
$$

其中 $d_m$ 是 `array_spacing_m`，$d_\lambda$ 是以波长为单位的阵元间距。它由物理间距与载频共同决定，因此任意 `center_freq` 下还原出的角度都正确，可视化端用同一物理间距反算角度。当 `array_spacing_m <= 0` 时退回到与频率无关的旧参数 `array_spacing_lambda`。如果配置 `steering_override_file`，则直接从阵列流形矩阵读取 $a_{i,k}$，不再根据 `angle_deg` 生成阵列响应。因此，自定义矩阵必须已包含目标角度对各阵元幅相响应的影响。

单站感知 RX 的第 $k$ 根天线为：

$$
y_{\mathrm{mono},k}[n] = \sum_i g_i\, a_{i,k}(\theta_i)\, x[n-\ell_i]\, e^{j 2\pi f_{D,i} n / f_s} + w_k[n]
$$

单站感知信道与发射机共享同一个仿真时钟，不叠加相对 CFO，也不叠加 UE 采样时钟偏差。上式中的 $w_k[n]$ 为 AWGN。

通信/双站 RX 为单天线。LoS/静态多径路径先形成通信多径分量：

$$
u_{\mathrm{LoS}}[n] = \sum_p h_p\, x[n-\ell_p], \qquad \ell_p = d_p + n_0
$$

其中 $d_p$ 来自 `comm_multipath_taps[].delay_samples`，$h_p$ 由 `gain_db` 和 `phase_deg` 决定。散射体散射分量叠加到同一个通信信道上：

$$
u[n] = u_{\mathrm{LoS}}[n] + \sum_i g_i\, x[n-\ell_i]\, e^{j 2\pi f_{D,i} n / f_s}
$$

禁用 SFO 时，随后通信/双站链路施加相对 CFO，并加入 AWGN。同一对 BS/UE 本振的相对频偏在反向链路上符号相反：

$$
y_{\mathrm{DL}}[n] = u_{\mathrm{DL}}[n]\, e^{j 2\pi f_{\mathrm{CFO}} n / f_{s,\mathrm{UE}}} + w_{\mathrm{DL}}[n]
$$

$$
y_{\mathrm{UL}}[n] = u_{\mathrm{UL}}[n]\, e^{-j 2\pi f_{\mathrm{CFO}} n / f_{s,\mathrm{BS}}} + w_{\mathrm{UL}}[n]
$$

下行跟踪产生 RX 频率校正后，仿真 UE 会像真实 USRP 路径一样同步 retune 上行 TX。设 $\alpha=f_{\mathrm{UL}}/f_{\mathrm{DL}}$（TDD 下 $\alpha=1$），则

$$
f_{\mathrm{DL,res}} = f_{\mathrm{CFO}} + f_{\mathrm{RX,corr}}, \qquad
f_{\mathrm{UL,res}} = -\alpha f_{\mathrm{CFO}} - f_{\mathrm{TX,corr}}.
$$

TDD 下 $f_{\mathrm{TX,corr}}=f_{\mathrm{RX,corr}}$；FDD 下上行 TX 校正按载频比 $\alpha$ 缩放。共享控制块传递的是 TX 目标载频校正量，而不是 UHD TX `dsp_freq` 的 API 符号。

启用 `sample_rate_offset_ppm` 时，仿真器先在源时钟上形成 $u[n]$，再经过
32-tap、1024-phase Kaiser-windowed sinc polyphase resampler。正 ppm 表示 UE
时钟快于 BS 时钟，因此 BS->UE 通信路径输出采样率为
$f_{s,\mathrm{UE}} = f_{s,\mathrm{BS}}(1+\mathrm{ppm}\cdot10^{-6})$。下行 CFO
相位步进使用这个 UE 侧采样率。UE->BS 上行通过 reciprocal resampler 回到 BS 时钟域后，再以 BS 侧采样率施加当前残余 CFO，频率步进变化时不重置连续相位。

默认情况下，`targets` 列表同时驱动单站感知天线（带阵列流形）和双站通信信道（单天线，不使用阵列流形），对应同一个相干场景。设置 `bistatic_targets` 可让双站/通信信道拥有独立的散射体。`comm_multipath_taps` 用于配置 LoS 路径和静态多径路径的时延、增益与初始相位。UE->BS 上行复用同一个通信信道模型，因此仿真 TDD 上行与下行信道形状互易。

## 说明 / 限制

- 每个进程都能通过 Ctrl-C（SIGINT）干净退出，且彼此独立；hub 退出时会清理（unlink）其共享内存。
- hub 默认启用实时节拍：每个 BS-clock chunk 按 `chunk_samples / sample_rate`
  的 wall-clock 时间释放。若需要最快 batch 仿真，可设置
  `simulation.pacing_enabled: false`。
- 将 `ring_capacity_samples` 设小（约几帧），使发射机不会远远跑在 hub 前面，从而避免单站 TX/RX 配对队列溢出。
- `sample_rate_offset_ppm` 建模一个端点级 UE/BS SFO；`timing_offset_samples`
  仍只是固定整数样本偏移，不会随时间漂移。
- 当前仿真没有建模分数时延；所有传播时延与配置定时偏移都会取整到样本级。
