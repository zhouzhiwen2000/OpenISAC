---
title: UE YAML 参考
description: UE 运行配置参数、取值和作用说明。
---

## 使用说明

`UE` 从启动时的当前工作目录读取 `UE.yaml`。建议先复制与硬件和场景匹配的 `config/UE_*.yaml` 模板，再在副本上修改。配置不会在运行中自动重载；修改后需要重启 UE。

下表按 YAML 顶层结构排列，并使用 `uplink.arq_enabled` 这样的完整路径区分同名字段。“典型值”用于说明常见配置，不代表所有模板的默认值。可选配置节省略时，程序使用解析器默认值。

> 帧结构、上下行双工、频率和资源映射等联动参数必须与 BS 配置一致。

## 参数表

### `radio`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `radio.radio_backend` | `string` | `uhd` | 无线设备输入/输出后端。`uhd` 使用 USRP，`sim` 使用信道仿真器，`xr1710g` 使用 CPU 纯接收后端。 |
| `radio.xr1710g_shm` | `string` | `/xr1710g-live` | 外部单路主机接收程序的 POSIX 共享内存名称。 |
| `radio.xr1710g_center_freq` | `float` / Hz | `2442000000` | 必填：从 XR1710G 控制器读取的实际源 LO；此字段不执行射频调谐。 |
| `radio.xr1710g_attach_timeout_s` | `float` / s | `5` | 等待主机接收服务的时间，范围 0–3600 秒。 |
| `radio.xr1710g_pll_command` | `string` | `""` | 可选的常驻 PLL 控制命令，在接收电脑执行。留空使用软件 NCO；配置后通过硬件 LO 微调校正 ±100 kHz 内的频偏，见下方接入说明。 |

CPU 纯接收 UE 可选择 `radio.radio_backend: xr1710g`。
支持的采样率、控制方式和时间戳限制见 [XR1710G 接入说明](../../tools-workflows/xr1710g-receiver/)。

### `simulation`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `simulation.session` | `string` | `oisac_sim` | BS、UE 和 `ChannelSimulator` 共用的仿真会话名称；三者必须一致。 |
| `simulation.enable_comm_rx` | `bool` | `true` | 仿真器生成通信 RX 路径。 |
| `simulation.enable_sensing_rx` | `bool` | `true` | 仿真器生成感知 RX 路径。 |
| `simulation.enable_uplink` | `bool` | `false` | 仿真器转发 UE 到 BS 的上行流。 |
| `simulation.pacing_enabled` | `bool` | `true` | 按实时采样速率调节仿真输出节奏。 |
| `simulation.noise_power_dbfs` | `float` / dBFS | `-100` | 每个 RX 通道的 AWGN 功率。 |
| `simulation.snr_control_enable` | `bool` | `false` | 通过缩放干净仿真信号维持 `target_snr_db`。 |
| `simulation.target_snr_db` | `float` / dB | `40` | 开启 SNR 控制时的目标 SNR。 |
| `simulation.control_port` | `int` | `10002` | ChannelSimulator 控制端口。 |
| `simulation.cfo_hz` | `float` / Hz | `0` | UE 下行 RX 校正前的 BS→UE 初始载波频偏。UE→BS 初始频偏符号相反，FDD 下还会按上下行载频比缩放；之后随 UE 上行 TX retune 动态更新残余 CFO。 |
| `simulation.sample_rate_offset_ppm` | `float` / ppm | `0` | UE 相对 BS 的采样钟偏差。 |
| `simulation.timing_offset_samples` | `int` / samples | `0` | 固定整数采样延迟。 |
| `simulation.array_spacing_m` | `float` / m | `0.04283` | ULA 物理阵元间距。 |
| `simulation.array_spacing_lambda` | `float` | `0.5` | 按波长归一化的旧版 ULA 间距。 |
| `simulation.ring_capacity_samples` | `int` | `262144` | 每条共享内存数据流的环形缓冲区容量。 |
| `simulation.steering_override_file` | `string` | `""` | 可选的阵列流形矩阵文件；留空时按 `angle_deg` 生成 ULA 阵列流形。配置该文件后，`angle_deg` 不再参与阵列响应计算，角度对应的幅相响应需预先写入矩阵。 |
| `simulation.comm_multipath_taps[]` | `object[]` | 可选 | 通信抽头延迟线的静态多径抽头，字段为 `delay_samples`、`gain_db`、`phase_deg`。 |
| `simulation.targets[]` | `object[]` | 可选 | 单站点目标。 |
| `simulation.bistatic_targets[]` | `object[]` | 可选 | 双站/通信点目标。 |

### `modulation`

| 路径 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `modulation.scheme` | `string` | `QPSK` | 业务载荷调制方式：`BPSK`、`QPSK`、`8PSK`、`16QAM`、`32QAM` 或 `64QAM`。同时用于上下行，BS 与 UE 必须一致，重启后生效。 |

在两端 YAML 中添加或修改：

```yaml
modulation:
  scheme: 16QAM
```

也可在 `config_web_editor` 的 BS、UE 页面中选择 **Payload modulation** / `scheme` 并保存。两端停止后修改配置，再使用原启动脚本重新启动。省略该配置时使用 QPSK。

所有载荷星座均采用 Gray 编码，并归一化到单位平均符号能量。32QAM 为 **8×4 矩形星座**。同步序列和导频保持原设计，包标记、包头及帧标识仍使用 QPSK。调制方式由两端配置约定，不进行自动协商。

每个载荷符号分别承载 1、2、3、4、5、6 个编码比特。较高阶调制需要更好的信噪比、相位稳定性和射频线性度；在相同采样率和资源网格下可提高理论载荷容量，实际 UDP 吞吐还取决于 LDPC 处理能力、帧开销和丢包情况。详见[软解调与 LDPC](/zh-cn/docs/signal-processing/ue-reception/#软信息与-ldpc)。

### `rf_sampling`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `rf_sampling.sample_rate` | `float` / Hz | `50000000` | 基带采样率。 |
| `rf_sampling.bandwidth` | `float` / Hz | `50000000` | 模拟带宽。 |
| `rf_sampling.rx_gain` | `float` / dB | `10` | UE 下行 RX 增益。 |
| `rf_sampling.rx_agc_enable` | `bool` | `true` | 开启 UE 下行硬件 RX AGC。 |
| `rf_sampling.rx_agc_low_threshold_db` | `float` / dB | `14` | 滤波后时延谱主峰低于该阈值时提高增益。 |
| `rf_sampling.rx_agc_high_threshold_db` | `float` / dB | `16` | 主峰高于该阈值时降低增益。 |
| `rf_sampling.rx_agc_max_step_db` | `float` / dB | `1` | 单次 RX AGC 最大步进。 |
| `rf_sampling.rx_agc_update_frames` | `int` | `4` | 两次 AGC 更新之间最少处理帧数。 |

### `usrp_device`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `usrp_device.device_args` | `string` | `addr=...` | USRP 设备参数，例如网络地址。 |
| `usrp_device.clock_source` | `string` | `external` | UE 时钟源：`internal`、`external` 或 `gpsdo`。 |
| `usrp_device.time_source` | `string` | `internal` | UE 时间/PPS 源。空值按 `internal` 处理；`pps_phase` 要求它与 `clock_source` 都显式为 `external`。 |

只要 `usrp_device.clock_source` 非空，CPU、CUDA 和 HIP UE 都会在任何 stream
开始前执行严格的 device-time 对齐，该步骤与所选 `sync_tracking.ocxo_mode` 无关。
这也包括 `time_source: internal`；`time_source` 只选择 PPS 输入，不再决定是否
执行启动对齐。启动过程先观察一次 PPS 跳变，再武装 `set_time_next_pps(0)`，并等待
device time 计数器确实回跳且 `time_last_pps=0`；PPS 缺失、backend 不支持或验证
失败都会中止启动。对 `gpsdo`，对齐会有意
用零点替换原有的类 UTC device-time epoch。该步骤对齐软件时间原点，但 X310
在 200 MHz master clock 下仍保留 5 ns 的 PPS 捕获量化。

### `ofdm_frame`

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

### `cuda`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `cuda.cuda_precision_profile` | `string` | `fp32` | `fp32` 完全保留现有路径；`mixed_fp16` 对批量 OFDM FFT/IFFT 和 GPU LDPC 消息使用 FP16，信道估计、跟踪、均衡、感知和 LLR 生成仍使用 FP32。不支持的 FP16 FFT 形状会自动回退 FP32。 |
| `cuda.cuda_demod_pipeline_slots` | `int` | `3` | CUDA 解调流水线 slot 数。 |
| `cuda.cuda_ldpc_decoder_backend` | `string` | `gpu` | CUDA 解调使用的 LDPC 解码后端：`gpu` 或 `cpu`。 |
| `cuda.cuda_ldpc_worker_buffers` | `int` | `3` | CUDA LDPC 异步批处理工作缓冲区数。 |
| `cuda.cuda_ldpc_cross_frame_flush_frames` | `int` | `2` | CUDA LDPC batch decode 前最多累计帧数。 |
| `cuda.cuda_ldpc_cross_frame_flush_us` | `float` / us | `1000` | CUDA LDPC 跨帧 batch 最长等待时间。 |

### `ldpc`

在 Web 编辑器的 **LDPC → code_rate** 中选择载荷码率，或在两端 YAML 中设置：

```yaml
ldpc:
  code_rate: "2/3"
```

码率定义和使用取舍见[码率匹配与译码](/zh-cn/docs/signal-processing/ue-reception/#软信息与-ldpc)。

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ldpc.code_rate` | `string` | `1/2` | 载荷码率：`1/3`、`1/2`、`2/3` 或 `3/4`；两端须一致，修改后重启两端。 |
| `ldpc.decoder_iterations` | `int` | `12` | CPU LDPC 最大迭代次数，必须大于等于 1；保留 syndrome 提前结束。 |
| `ldpc.nms_normalization_factor` | `float` | `0.75` | CPU NMS 归一化系数，有限值且在 `(0, 1]` 内；Q16 模式使用 `0.125` 的整数倍。 |
| `ldpc.fixed_point` | `bool` | `false` | 使用 int16/Q16 CPU LDPC 解码路径。 |
| `ldpc.fixed_point_scale` | `int` | `16` | int16 饱和前的 LLR 缩放。 |

这两项用于 UE 下行的 CPU 译码，float32 和 Q16 共用设置。省略时使用 12 次迭代和系数 0.75。已有 YAML 的显式值继续生效；要在已有配置中使用这组默认值，请更新这两项。可在 [Web 配置控制台](/zh-cn/docs/tools-workflows/web-config-console/) 的 **LDPC** 中修改，保存后重启后端。Q16 支持 `0.125、0.25、0.375、0.5、0.625、0.75、0.875、1.0`；LLR 比例仅在 `fixed_point: true` 时参与计算。增加迭代次数可能改善纠错，同时增加弱信号时的计算量。

### `downlink`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `downlink.center_freq` | `float` / Hz | `2400000000` | UE 下行 RF 中心频率。 |
| `downlink.rx_wire_format` | `string` | `sc16` | UE 下行接收的 UHD 样本格式。 |
| `downlink.rx_channel` | `int` | `0` | UE 下行 RX 通道索引。 |
| `downlink.equalizer_mode` | `string` | `mmse` | 下行均衡器反演模式：`zf` 或 `mmse`。 |
| `downlink.channel_tracking_mode` | `string` | `pilot_phase` | 每符号梳状导频跟踪模式。 |
| `downlink.equalizer_mag_floor` | `float` | `1e-6` | 信道反演时 `|H|^2` 下限。 |
| `downlink.channel_tracking_min_pilot_snr` | `float` | `1e-4` | 回退前要求的最小梳状导频残差权重。 |
| `downlink.arq_enabled` | `bool` | `false` | 在 UE 接收端开启下行 ARQ。 |
| `downlink.arq_ordered_delivery` | `bool` | `false` | 缓存下行数据包，再按序输出 UDP。 |
| `downlink.arq_window_packets` | `int` | `256` | 下行 ARQ 接收/重排窗口；有效范围为 `1`–`32767`。 |
| `downlink.arq_feedback_interval_ms` | `int` / ms | `10` | 下行 ARQ ACK 反馈的最小时间间隔。 |

### `uplink`

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
| `uplink.wire_format_tx` | `string` | `sc16` | UE 上行发送的 UHD 样本格式。 |
| `uplink.ue_timing_advance` | `int` / samples | `50` | UE 上行发送 timing advance。 |
| `uplink.debug_self_channel` | `bool` | `false` | 从接收窗口估计 UE 本机发送泄漏信道，用于 `TADV` 调试。 |
| `uplink.ertm_to_enable` | `bool` | `false` | 开启 eRTM 时偏测量负载的接收处理和 TO 日志。 |
| `uplink.ertm_timing_metric` | `string` | `delay_magnitude` | 差分 TO 指标：抗相位失配的 `delay_magnitude`，或相干复信道相关的 `maximum_likelihood`。 |
| `uplink.ertm_delay_oversample_factor` | `int` | `10` | eRTM 时延谱 IFFT 过采样倍数。 |
| `uplink.ertm_dl_rf_delay_samples` | `float` / samples | `67.0` | eRTM 方程中的下行 RF 链路校准延迟。 |
| `uplink.ertm_ul_rf_delay_samples` | `float` / samples | `67.0` | eRTM 方程中的上行 RF 链路校准延迟。 |
| `uplink.ertm_debug_output_enabled` | `bool` | `false` | 开启 UE 侧 eRTM debug ZMQ 频谱输出。 |
| `uplink.ertm_report_interval_frames` | `int` / frames | `32` | BS eRTM report 间隔；对比日志时应与 BS 保持一致。 |
| `uplink.arq_enabled` | `bool` | `false` | 在 UE 发射端开启上行 ARQ。 |
| `uplink.arq_window_packets` | `int` | `256` | 上行 ARQ 最多允许的未确认数据包数；有效范围为 `1`–`32767`。 |
| `uplink.arq_retransmit_timeout_ms` | `int` / ms | `100` | 上行 ARQ 重传超时。 |
| `uplink.arq_max_retries` | `int` | `5` | 上行最大重传次数；`0` 表示窗口内不限次数。 |

真实无线设备开启 timed uplink 时，CPU、CUDA 和 HIP UE 会在启动阶段读取设备实际的
RX/TX 采样率。两者相差超过 1 Hz 会在创建工作线程前中止启动，因为精确的上行身份映射
要求收发方向使用同一采样时间网格。

### `sync_tracking`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `sync_tracking.sync_queue_size` | `int` | `32` | 同步搜索 batch 队列容量。 |
| `sync_tracking.sync_cfo_alias_search_range_hz` | `float` / Hz | `800000` | 同步 alias 解析覆盖的 CFO 范围。 |
| `sync_tracking.reset_hold_s` | `float` / s | `0.5` | 坏 delay 或配置要求的前导字段不匹配连续持续多久后，硬 reset 回同步搜索。 |
| `sync_tracking.software_sync` | `bool` | `true` | 开启软件同步跟踪。 |
| `sync_tracking.predictive_delay` | `bool` | `true` | 使用基于 CFO 的预测性时延补偿。若外部变频器引入的 CFO 不随采样时钟误差变化，应关闭此项。 |
| `sync_tracking.hardware_sync` | `bool` | `false` | 开启硬件同步模式。 |
| `sync_tracking.hardware_sync_tty` | `string` | `/dev/ttyUSB0` | 硬件同步控制器串口设备。 |
| `sync_tracking.desired_peak_pos` | `int` | `20` | 定时对齐逻辑使用的目标时延峰位置。 |
| `sync_tracking.ocxo_mode` | `string` | `auto` | `auto`、`freq`、`phase` 保留原 TO 锚定语义；`pps_phase` 显式选择 PPS 相位锁定。 |
| `sync_tracking.ocxo_dac_ppm_per_lsb` | `float` | `9.71e-7` | 标定后的 DAC 每 LSB 对应 ppm（MV197）。 |
| `sync_tracking.ocxo_dac_zero_word` | `int` | `493367` | 0 ppm 对应的 DAC 码字。 |
| `sync_tracking.ocxo_dac_state_file` | `string` | `.openisac_ocxo_dac_state.yaml` | 保存 DAC 已确认 `OK` 的最后一个 SF 码字。相对路径按 UE 当前工作目录解析；空值关闭跨重启持久化。 |
| `sync_tracking.ocxo_max_abs_ppm` | `float` / ppm | `0.45` | OCXO 绝对命令限幅。 |
| `sync_tracking.ocxo_actuator_max_slew_ppm_per_s` | `float` / ppm/s | `0.01` | 执行器硬斜率限制；实际执行量会反馈给控制器。 |
| `sync_tracking.ocxo_coarse_*` | 多项 | 见预设 | 启动单目标粗调：Mode A 平均 residual CFO；Mode B 在 coarse 窗内对 Alignment 连续化后的 `TO_clock` 与绝对帧序号做线性回归。 |
| `sync_tracking.ocxo_fll_fast_*`, `ocxo_fll_slow_*` | 多项 | 见预设 | 快/慢 FLL：Mode A 使用 residual CFO，Mode B 使用滚动 `TO_clock` 回归得到的钟 SFO，并带阈值滞回和 dwell。 |
| `sync_tracking.ocxo_fll_slow_kp` | `float` | `0.1` | Slow FLL 比例增益。 |
| `sync_tracking.ocxo_fll_slow_ki` | `float` | `0.007` | Slow FLL 积分增益。 |
| `sync_tracking.ocxo_fll_slow_max_slew_ppm_per_s` | `float` / ppm/s | `0.005` | Slow FLL 命令变化率上限。 |
| `sync_tracking.ocxo_fll_slow_enter_abs_ppm` | `float` / ppm | `0.0005` | 从 Fast FLL 进入 Slow FLL 的残差阈值。 |
| `sync_tracking.ocxo_phase_setpoint_mode` | `string` | `freeze_on_lock` | 原 `auto`/`phase` 的目标：进入 Phase 时冻结 `TO_clock`、取零或使用固定值。 |
| `sync_tracking.ocxo_phase_setpoint_fixed_samples` | `float` / samples | `0.0` | setpoint mode 为 `fixed` 时的原 TO_clock 固定目标。 |
| `sync_tracking.ocxo_phase_*` | 多项 | 见预设 | Phase 增量式 PI：原 `auto`/`phase` 使用 `setpoint-TO_clock`，`pps_phase` 使用 wrap 后的 `P_U-P_B`；进退判据仍使用 `TO_clock` 钟 SFO 回归。 |
| `sync_tracking.ocxo_pps_initial_ap_enable` | `bool` | `true` | 保留的旧键名。在 `pps_phase` 中，每次 `FllSlow -> Phase` 都启动一次 PPSOUT 对齐事务，再放行 SF Phase 跟踪。关闭时启动日志会告警：大 residual 只能由限斜率 PI 处理，可能收敛很慢。 |
| `sync_tracking.ocxo_pps_initial_ap_max_abs_ns` | `float` / ns | `500000000` | AP 绝对限幅，与 PPSDO 硬件范围一致。 |
| `sync_tracking.ocxo_pps_initial_ap_settle_s` | `float` / s | `2.0` | AP 最短稳定时间；同时必须等到更新后的 eRTM/PPS 观测。 |
| `sync_tracking.ocxo_pps_initial_ap_command_timeout_s` | `float` / s | `3.0` | AP worker 的总墙钟预算，包含全部 `JAMSYNCING` 重试；aligner 额外等待 0.5 s 终态余量，worker 截止后不再发送该 AP。 |
| `sync_tracking.ocxo_pps_initial_ap_max_retries` | `int` | `1` | AP 已接受但在 `settle_s + command_timeout_s` 内没有新 publication 时的有界重试次数（`0..3`）；重试会等待新的有效观测并按残差重算 AP。 |
| `sync_tracking.ocxo_pps_initial_ap_deadzone_ns` | `float` / ns | `10.0` | 进入 Phase 时 PPS 误差已落在该范围内则跳过 AP 和稳定等待。 |
| `sync_tracking.ocxo_pps_initial_ap_polarity` | `int` | `1` | `P_U-P_B` 为正时的 AP 命令符号。默认值采用 PPSOUT 台架实测确认的极性；不同硬件或固件极性仍可显式设为 `-1`。 |
| `sync_tracking.cfo_ewma_enable` | `bool` | `true` | 对 residual CFO ppm 做 EWMA 预滤波，仅供 Mode A Coarse/FLL 使用。 |
| `sync_tracking.cfo_ewma_alpha` | `float` | `0.02` | EWMA 对新样本权重（`0..1`；越大跟踪越快、噪声越大）。 |

硬件预设使用 `ocxo_fll_slow_kp=0.1`、`ocxo_fll_slow_ki=0.007`、
`ocxo_fll_slow_max_slew_ppm_per_s=0.005` 和
`ocxo_fll_slow_enter_abs_ppm=0.0005`，并保留
`ocxo_fll_slow_deadzone_ppm=5.0e-5`、`ocxo_phase_kp=2.0e-4`、
`ocxo_phase_ki=1.0e-6` 与 `ocxo_phase_enter_sfo_abs_ppm=2.0e-4`。
每次进入 Phase 的 PPSOUT AP 校正默认开启，并使用台架实测确认的正命令极性。

普通重新同步只重置 OCXO 观测、dwell 计时和控制阶段，不再向 DAC 写入零码字。
控制器从执行器保留的 command 重新起步；执行器在重新入锁后的第一条实际非零修正时
才重置 slew 时间基准，因此 search 时长不会放大第一步。每条 `SF` 收到 `OK` 后，
非实时串口 worker 会持久化更新
`ocxo_dac_state_file`；下次启动 UE/CUDAUE 进程时，DAC 与控制器都从该已确认的 raw
word 恢复，而不是从零开始。正常退出时先结束串口 worker，再次保存最后一条实际收到
`OK` 的 SF word；更新但未确认的 queued word 不会被提升为下次启动的稳态值。
恢复时还要求文件中的串口设备、`zero_word` 和 `ppm_per_lsb` 与当前配置一致。文件缺失时
从校准零码字启动，并在命令确认后创建状态；文件损坏、设备或标定不匹配、word 越界时也
回退到零码字，但原文件保持不变，且本次进程关闭持久化，需修复或删除该文件后再重启。
该机制假定仅重启 UE 进程时外部 DAC 仍保持供电并保留原码字。

`auto` 和 `phase` 保留原来的 Phase anchor：PI 控制
`setpoint - TO_clock`，继续支持 `freeze_on_lock`、`zero`、`fixed`；这两个
模式不会读取 `time_last_pps()`，也不会进入 AP gate。

`pps_phase` 是单独的显式模式。对下行帧 `N`，eRTM 根据每份 ERTM5 报告
打包前实时刷新的 BS frame-0/PPS 锚点重建 `r_B=B_N-P_B`，根据 UE 帧边界
与 PPS 时间戳得到
`r_U=U_N-P_U`。Phase 误差为
`wrap_1s(tau_DL_RF - TO_UE + r_B - r_U) = wrap_1s(P_U-P_B)`，目标为零。
该模式要求 `sync_tracking.hardware_sync=true`、`uplink.ertm_to_enable=true`，
并要求 `usrp_device.clock_source` 与 `usrp_device.time_source` 都显式配置为
`external`；空 `time_source` 按 `internal` 处理。任一条件不满足都会报配置错误。
BS 侧生成 PPS anchor 时也必须使用连接 PPSDO 的 external clock/time source。
BS anchor 的刷新频率由 `uplink.ertm_report_interval_frames` 的 eRTM 报告周期决定，
并在 TX/调制实时路径之外执行。每次 `FllSlow -> Phase` 时，AP gate 都会重新武装，
并根据最新有效 PPS residual 计算一次校正；在同一次 Phase 停留期间保持 Done，
普通 Phase 更新和 Freeze 不会重复 AP。若 Phase 退回 FLL 后再次进入，则允许一次
新的校正。`AP` 仅改变 PPSDO 的 PPSOUT，不会重设 USRP 时间、重锚 air-frame
clock、清空 eRTM 状态或重新建立 TO。若旧 AP 仍处于 Queued/InFlight，新一次进入会
等待硬件回到 Idle，而不是把 Busy 当作终端失败；`JAMSYNCING` 重试仍属于同一次 AP 请求。

### `sensing`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `sensing.sensing_delay_correction_mode` | `string` | `los_tracking` | 双站感知时延校正来源：`los_tracking` 或 `ertm_absolute`。 |
| `sensing.bi_enabled` | `bool` | `true` | 开启双站感知处理。 |
| `sensing.bistatic_phase_reference` | `bool` | `false` | LoS 相位参考，可改善较差时钟下的感知表现；要求收发机具有 LoS 且两者相对静止，重启 UE 生效。 |
| `sensing.phase_reference_min_coherence` | `float` | `0.05` | 最低参考相干度，范围 `(0, 1]`；开启相位参考时有效。 |
| `sensing.range_fft_size` | `int` | `1024` | 距离向 FFT 点数。 |
| `sensing.doppler_fft_size` | `int` | `100` | 多普勒向 FFT 点数。 |
| `sensing.view_range_bins` | `int` | `0` | 后端距离-多普勒视图的距离维宽度；`0` 表示完整距离维。 |
| `sensing.view_doppler_bins` | `int` | `0` | 后端距离-多普勒视图的多普勒维高度；`0` 表示完整多普勒维。 |
| `sensing.output_mode` | `string` | `dense` | `dense` 完整输出，或 `compact_mask` 选中 RE 输出。 |
| `sensing.on_wire_format` | `string` | `complex_float32` | 感知负载的网络传输格式。 |
| `sensing.backend_processing_enabled` | `bool` | `false` | 在支持时额外输出后端距离-多普勒、CFAR 和微多普勒处理结果。 |
| `sensing.symbol_stride` | `int` | `20` | dense 模式启动时默认 STRD。 |
| `sensing.frame_decimation` | `int` | `1` | 每 N 个完整感知帧处理并输出 1 帧，范围 `1..1000`；不改变帧内 STRD。上位机 `Frame every N` 可在线设置。 |


#### LoS 相位参考

该选项可在收发机使用独立时钟或时钟稳定性较差、同步后仍存在残余公共相位漂移时，改善双站感知的慢时间相干性和多普勒稳定性。**启用前必须保证收发机之间有稳定的视距（LoS）直达路径，而且收发机两者相对静止。** 被感知目标仍可运动。仅有非视距路径，或收发机之间存在相对运动时，不应启用：参考路径自身的运动也会被从测得的多普勒中减去。

在 UE Linux 主机的实际运行配置 `UE.yaml` 中合并以下字段：

```yaml
sensing:
  bi_enabled: true
  output_mode: dense
  sensing_delay_correction_mode: los_tracking
  bistatic_phase_reference: true
  phase_reference_min_coherence: 0.05
```

这是**启动时读取的 YAML 配置开关**，也可以在 Web 配置页编辑。保存后需重启 UE 才会生效，目前没有对应的在线启停命令。

相位校正在现有 CFO/SFO/时延及系统响应校正之后、MTI 之前，消除双站 CSI 的残余公共相位，适用于完整信道矩阵输出（`dense`）。

每个慢时间采样行用 Hamming 加权的直达径分量作为参考，对所有子载波施加模长为 1 的共轭旋转，保留幅度。参考跟随已知的 `ALGN` 显示平移：只在测量参考时抵消对应相位斜坡，输出仍保留用户指定的时延平移。平移变化会清空未完成的感知批次和 MTI/微多普勒历史，避免混合两个显示坐标。直达径需要静止且占主导：显示的多普勒相对于该参考路径；若参考路径运动，其运动也会被减去。此功能不校正射频采样时钟。

`sensing.phase_reference_min_coherence` 默认 `0.05`，范围为 `(0, 1]`。相干度定义为 `|sum(w H)|² / (sum(w) sum(w |H|²))`，其中 `w` 为 Hamming 权重，`H` 为抵消已知显示相位斜坡后的信道值。参考过弱、接近零或出现非有限值时，整批感知数据被丢弃并清空 MTI/微多普勒历史，通信样本不受影响。缺少慢时间采样点时会重建感知窗口和历史，以保持多普勒处理所需的均匀时间轴。

修改这些 YAML 字段后重启 UE，在 UE Linux 主机启动日志中确认 `static LoS phase reference enabled before MTI`。若频繁出现 `weak/invalid LoS phase reference`，说明该路径不适合作为参考，应关闭此功能或改善直达径信号。

### `resource_preview`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `resource_preview.data_resource_blocks[]` | `object[]` | 可选 | payload / sensing-pilot RE 矩形；每项包含 `kind`、`symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`。 |
| `resource_preview.mask_blocks[]` | `object[]` | 可选 | compact 感知 RE 矩形；每项包含 `symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`。 |

### `measurement`

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

### `network_output`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `network_output.udp_input_ip` | `string` / IPv4 | `0.0.0.0` | UE 上行业务 UDP 输入绑定 IP。 |
| `network_output.udp_input_port` | `int` | `50002` | UE 上行业务 UDP 输入端口。 |
| `network_output.udp_output_ip` | `string` / IPv4 | `""` | UE 解码下行 UDP 输出目标 IP；空字符串使用 `runtime.default_out_ip`。 |
| `network_output.udp_output_port` | `int` | `50001` | UE 解码下行 UDP 输出端口。 |
| `network_output.udp_egress_pacer_enabled` | `bool` | `false` | 开启解码 UDP 输出速率调节。 |
| `network_output.udp_egress_pacer_target_mbps` | `float` / Mbps | `0` | UDP 输出目标速率；`0` 时自动估计。 |
| `network_output.udp_egress_pacer_queue_packets` | `int` | `10240` | UDP 输出调速队列的数据包容量。 |
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
| `network_output.ocxo_debug_ip` | `string` / IPv4 | `0.0.0.0` | OCXO debug ZMQ PUB 绑定 IP。 |
| `network_output.ocxo_debug_port` | `int` | `12347` | OCXO debug ZMQ PUB 端口。 |

### `cpu_cores`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `cpu_cores.downlink_cpu_cores` | `int[]` | `[1,2,3]` | UE 下行核心：RX、处理、bit processing。 |
| `cpu_cores.demod_worker_cpu_cores` | `int[]` | `[]` | UE CPU demod worker 核；空列表启动一个不绑定 worker。 |
| `cpu_cores.ldpc_worker_cpu_cores` | `int[]` | `[]` | UE CPU LDPC decode worker 核；空列表启动一个不绑定 worker。 |
| `cpu_cores.sensing_cpu_cores` | `int[]` | `[4]` | UE 双站感知核心。 |
| `cpu_cores.uplink_cpu_cores` | `int[]` | `[]` | UE 上行核心：LDPC 编码、调制、TX 发送、UDP 接收。 |
| `cpu_cores.main_cpu_core` | `int` | `-1` | 主线程 CPU 核；`-1` 表示不绑定。 |

### `runtime`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `runtime.default_out_ip` | `string` / IPv4 | `127.0.0.1` | 特定输出 IP 为空时，UDP 输出使用的默认目标 IP。 |

硬件同步启用时，`OCXODBG3` ZeroMQ 数据包保留以下 16 个固定通道：

| 通道 | 信号 | 通道 | 信号 |
| :--- | :--- | :--- | :--- |
| 0 | 模值 CFO（Hz） | 8 | 当前阶段窗口的 `TO_clock` 回归钟 SFO（ppm） |
| 1 | 解模糊 CFO（Hz） | 9 | 累计 Alignment（samples） |
| 2 | 原始 residual CFO（ppm） | 10 | Alignment 前馈（samples） |
| 3 | 滤波后的控制 CFO（ppm） | 11 | 期望 OCXO 步进（ppm） |
| 4 | OFDM delay-track SFO（ppm；仅信道补偿） | 12 | 实际 OCXO 步进（ppm） |
| 5 | 原始 `TO_UE`（samples） | 13 | 已应用 OCXO command（ppm） |
| 6 | 连续 `TO_clock`（samples） | 14 | 粗调未完成量（ppm） |
| 7 | 当前 Phase 误差：原 `setpoint-TO_clock` 或 `pps_phase` 的 `P_U-P_B`（samples） | 15 | stage（`0..4`：Coarse、Fast、Slow、Phase、Freeze） |

数据包是固定 152 字节的小端帧：magic、版本、通道数、序号、Unix 纳秒源时间戳，
随后为原有 16 个 `float32` 和 typed PPS diagnostic trailer。trailer 与 CSV 追加
`r_bs_samples`、`r_ue_samples`（`float64`）、`bs_last_pps_ticks`、
`bs_t0_to_pps_ticks`、`ue_last_pps_ns`（精确 `int64`）、`bs_tick_rate` 和
`sample_rate`（`float64`）。这些字段用于区分 BS anchor、UE PPS latch 与 TO
造成的相位跳变；绝对 timestamp 不经过 `float32`。viewer 仍兼容旧
`OCXODBG2` 数据包。使用 `python3 scripts/plot_ocxo_debug.py` 启动滚动查看器；
`--channels 6,7,8,13,15` 可指定初始显示通道，`--record capture/ocxo.csv`
可在启动时记录 CSV。自动文件名按天使用 `ocxo_debug_YYYYMMDD_01.csv`，同日
依次递增为 `_02`、`_03`。无论界面勾选哪些曲线，文件始终保存全部通道。除周期状态包外，
控制、阶段切换、TO jump 和 Alignment 会立即额外发布事件包，避免单帧 delta 通道被降采样遗漏。

### `logging`

性能剖析与诊断日志共用层级化 `logging.*`。性能项路径以 `_profiling` 结尾（默认关闭，需显式打开，例如 `demod_profiling: info`）。

资源映射说明：
* `resource_preview.data_resource_blocks` 通常应在 BS 和 UE 之间保持一致，包括 `kind`。
* 内置 ZC 同步符号、可选 CFO training field、梳状导频和帧内 BPSK 导频优先级高于资源块配置。
* `resource_preview.mask_blocks` 只控制 compact 感知导出。`output_mode=compact_mask` 时运行时 `STRD` 会被忽略，因为采样图样已经由 mask 定义。
* compact 感知 payload 以 `CompactSensingFrameHeader` 开头，后面跟固定顺序的原始 `complex<float>` 数据。

## 感知帧抽取

当较低 STRD 导致感知处理或上位机负载过高时，可设置 `sensing.frame_decimation: 5`，每 5 个完整感知帧（CPI）输出 1 帧。默认 `1` 输出全部帧。上位机的 `Frame every N` 输入框和 `Set` 按钮可在线调整，`Params` 中的 `every=N` 显示后端回报值；在线修改不写回 YAML，启动默认值可通过 config_web_editor 保存。

抽取保留每个输出帧内的符号间隔、采样点数和多普勒无模糊范围，降低画面刷新率。采样、信道估计和 MTI 保持连续；跳过帧的距离–多普勒 FFT、检测及数据发送不再执行，因此主要降低后级处理和传输负载。多通道使用相同的帧选择规则。

帧间空档会中断微多普勒历史累积，较大的抽取倍数可能无法积累足够长的连续窗口；需要连续微多普勒分析时使用 `1`。这项设置不改变通信数据传输或射频发射时序。更新后应配套使用同一版本的后端和上位机。
