---
title: BS YAML 参考
description: BS 运行配置参数、取值和作用说明。
---

## 使用说明

`BS` 从启动时的当前工作目录读取 `BS.yaml`。建议先复制与硬件和场景匹配的 `config/BS_*.yaml` 模板，再在副本上修改。配置不会在运行中自动重载；修改后需要重启 BS。

下表按 YAML 顶层结构排列，并使用 `uplink.arq_enabled` 这样的完整路径区分同名字段。“典型值”用于说明常见配置，不代表所有模板的默认值。可选配置节省略时，程序使用解析器默认值。

> 帧结构、上下行双工、频率和资源映射等联动参数必须与 UE 配置一致。

## 参数表

### `radio`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `radio.radio_backend` | `string` | `uhd` | 无线设备输入/输出后端。`uhd` 使用真实 USRP，`sim` 使用共享内存信道仿真器。 |

### `simulation`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `simulation.session` | `string` | `oisac_sim` | BS、UE 和 `ChannelSimulator` 共用的仿真会话名称；三者必须一致。 |
| `simulation.enable_comm_rx` | `bool` | `true` | 仿真器生成 UE 通信接收路径。 |
| `simulation.enable_sensing_rx` | `bool` | `true` | 仿真器生成单站感知 RX 路径。 |
| `simulation.enable_uplink` | `bool` | `false` | 仿真器转发 UE 到 BS 的上行流。 |
| `simulation.pacing_enabled` | `bool` | `true` | 按实时采样速率调节仿真输出节奏。 |
| `simulation.noise_power_dbfs` | `float` / dBFS | `-70` | 每个 RX 通道的 AWGN 功率；很低的值近似关闭噪声。 |
| `simulation.snr_control_enable` | `bool` | `false` | 在加 AWGN 前缩放干净信号，以维持 `target_snr_db`。 |
| `simulation.target_snr_db` | `float` / dB | `40` | 开启 SNR 控制时的目标 SNR。 |
| `simulation.control_port` | `int` | `10002` | ChannelSimulator 运行时 SNR 控制 ZMQ 端口。 |
| `simulation.cfo_hz` | `float` / Hz | `0` | UE 下行 RX 校正前的 BS→UE 初始载波频偏。UE→BS 初始频偏符号相反，FDD 下还会按上下行载频比缩放；之后随 UE 上行 TX retune 动态更新残余 CFO。 |
| `simulation.sample_rate_offset_ppm` | `float` / ppm | `0` | UE 相对 BS 的采样钟偏差。 |
| `simulation.timing_offset_samples` | `int` / samples | `0` | 注入到 RX 的固定整数采样延迟。 |
| `simulation.array_spacing_m` | `float` / m | `0.04283` | ULA 物理阵元间距；`<=0` 时使用 `array_spacing_lambda`。 |
| `simulation.array_spacing_lambda` | `float` / lambda | `0.5` | 旧版按波长归一化的 ULA 间距。 |
| `simulation.ring_capacity_samples` | `int` / samples | `262144` | 每条共享内存数据流的环形缓冲区容量。 |
| `simulation.steering_override_file` | `string` | `""` | 可选的阵列流形矩阵文件；留空时按 `angle_deg` 生成 ULA 阵列流形。配置该文件后，`angle_deg` 不再参与阵列响应计算，角度对应的幅相响应需预先写入矩阵。 |
| `simulation.comm_multipath_taps[]` | `object[]` | 可选 | 通信抽头延迟线的静态多径抽头，字段为 `delay_samples`、`gain_db`、`phase_deg`。 |
| `simulation.targets[]` | `object[]` | 可选 | 单站感知点目标，字段为 `range_m`、`velocity_mps`、`gain_db`、`angle_deg`。 |
| `simulation.bistatic_targets[]` | `object[]` | 可选 | 双站/通信点目标，字段同上。 |

### `rf_sampling`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `rf_sampling.sample_rate` | `float` / Hz | `50000000` | 基带采样率。 |
| `rf_sampling.bandwidth` | `float` / Hz | `50000000` | 模拟带宽，通常与采样率一致。 |

### `usrp_device`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `usrp_device.device_args` | `string` | `addr=...` | 通用 USRP 设备参数，例如网络地址。 |

### `clock_time`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `clock_time.clock_source` | `string` | `external` | 全局时钟源：`internal`、`external` 或 `gpsdo`。 |
| `clock_time.time_source` | `string` | `internal` | 全局时间/PPS 源；空字符串表示跟随 `clock_source`。 |

### `ofdm_frame`

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

### `cuda`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `cuda.cuda_mod_pipeline_slots` | `int` | `3` | CUDA 调制流水线 slot 数，小于 `1` 会钳制。 |

### `ldpc`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ldpc.fixed_point` | `bool` | `false` | 使用 int16/Q16 layered-NMS CPU 解码器而不是 float32。 |
| `ldpc.fixed_point_scale` | `int` | `16` | 固定点模式下 int16 饱和前的 LLR 缩放。 |

### `downlink`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `downlink.center_freq` | `float` / Hz | `2400000000` | BS 下行 RF 中心频率。 |
| `downlink.tx_gain` | `float` / dB | `60` | BS 下行 TX 增益。 |
| `downlink.tx_channel` | `int` | `0` | BS 下行 TX 通道索引。 |
| `downlink.tx_device_args` | `string` | `""` | 下行发送专用的 USRP 设备参数；留空时使用 `usrp_device.device_args`。 |
| `downlink.tx_clock_source` | `string` | `""` | TX 时钟源覆盖。 |
| `downlink.tx_time_source` | `string` | `""` | TX 时间源覆盖。 |
| `downlink.wire_format_tx` | `string` | `sc16` | UHD 发送样本格式，常用 `sc16` 或 `sc8`。 |
| `downlink.arq_enabled` | `bool` | `false` | 在 BS 发射端开启下行 ARQ。 |
| `downlink.arq_window_packets` | `int` | `32767` | 下行 ARQ 最多允许的未确认数据包数。 |
| `downlink.arq_retransmit_timeout_ms` | `int` / ms | `100` | 下行 ARQ 重传超时。 |
| `downlink.arq_max_retries` | `int` | `5` | 下行最大重传次数；`0` 表示窗口内不限次数。 |

### `downlink_pipeline`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `downlink_pipeline.tx_circular_buffer_size` | `int` | `8` | 送入 TX 的已调制帧队列容量。 |
| `downlink_pipeline.data_packet_buffer_size` | `int` | `256` | 已编码数据包缓冲区容量。 |

### `uplink`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `uplink.enabled` | `bool` | `false` | UE 到 BS 上行/双工路径总开关。 |
| `uplink.duplex_mode` | `string` | `tdd` | `tdd` 使用上行符号窗口；`fdd` 使用 `uplink.center_freq` 和整帧上行。 |
| `uplink.center_freq` | `float` / Hz | `2500000000` | 仅 FDD 使用的上行载波；TDD 使用下行中心频率。 |
| `uplink.symbol_start` | `int` | `90` | TDD 下上行窗口起始符号。 |
| `uplink.symbol_count` | `int` | `10` | TDD 下上行窗口长度；`0` 关闭 TDD 上行。 |
| `uplink.guard_symbols` | `int` | `1` | TDD 上行窗口内的前置 guard 符号数。 |
| `uplink.bs_dl_ul_timing_diff` | `int` / samples | `50` | BS 上行 RX 窗口相对下行 TX 帧锚点的偏移。 |
| `uplink.debug_self_channel` | `bool` | `false` | 从上行接收窗口估计本机发送泄漏信道，用于 `DUTI` 调试。 |
| `uplink.ertm_to_enable` | `bool` | `false` | 开启 eRTM 时偏测量负载和 UE 侧 TO 日志。 |
| `uplink.ertm_report_interval_frames` | `int` / frames | `32` | BS 每隔多少个下行发送帧生成一次 eRTM 报告。 |
| `uplink.rx_gain` | `float` / dB | `0` | BS 上行 RX 增益。 |
| `uplink.rx_channel` | `int` | `0` | BS 上行 RX 通道索引。 |
| `uplink.rx_wire_format` | `string` | `sc16` | BS 上行接收的 UHD 样本格式。 |
| `uplink.rx_device_args` | `string` | `""` | 上行接收专用的 USRP 设备参数。 |
| `uplink.rx_clock_source` | `string` | `""` | 上行 RX 时钟源覆盖。 |
| `uplink.rx_time_source` | `string` | `""` | 上行 RX 时间源覆盖。 |
| `uplink.rx_agc_enable` | `bool` | `false` | 开启 BS 上行硬件 RX AGC。 |
| `uplink.rx_agc_low_threshold_db` | `float` / dB | `14` | 滤波后时延谱主峰低于该阈值时提高增益。 |
| `uplink.rx_agc_high_threshold_db` | `float` / dB | `16` | 主峰高于该阈值时降低增益。 |
| `uplink.rx_agc_max_step_db` | `float` / dB | `1` | 单次上行 RX AGC 最大步进。 |
| `uplink.rx_agc_update_frames` | `int` | `4` | 两次上行 AGC 更新之间最少处理帧数。 |
| `uplink.equalizer_mode` | `string` | `mmse` | BS 上行均衡器反演模式：`zf` 或 `mmse`。 |
| `uplink.channel_tracking_mode` | `string` | `pilot_phase` | 上行每符号梳状导频跟踪模式：`disabled` 或 `pilot_phase`。 |
| `uplink.equalizer_mag_floor` | `float` | `1e-6` | 上行信道反演时 `|H|^2` 下限。 |
| `uplink.channel_tracking_min_pilot_snr` | `float` | `1e-4` | 回退前要求的最小梳状导频残差权重。 |
| `uplink.arq_enabled` | `bool` | `false` | 在 BS 接收端开启上行 ARQ。 |
| `uplink.arq_ordered_delivery` | `bool` | `false` | 缓存上行数据包，再按序输出 UDP。 |
| `uplink.arq_window_packets` | `int` | `32767` | 上行 ARQ 接收/重排窗口。 |
| `uplink.arq_feedback_interval_ms` | `int` / ms | `10` | 上行 ARQ ACK 反馈的最小时间间隔。 |

### `sensing`

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
| `sensing.view_range_bins` | `int` | `0` | 后端距离-多普勒视图的距离维宽度；`0` 表示完整距离 FFT。 |
| `sensing.view_doppler_bins` | `int` | `0` | 后端距离-多普勒视图的多普勒维高度；`0` 表示完整多普勒 FFT。 |
| `sensing.output_mode` | `string` | `dense` | `dense` 为基于 STRD 的完整输出；`compact_mask` 只导出选中 RE。 |
| `sensing.on_wire_format` | `string` | `complex_float32` | 感知负载的网络传输格式。 |
| `sensing.backend_processing_enabled` | `bool` | `false` | 在支持时额外输出后端距离-多普勒、CFAR 和微多普勒处理结果。 |
| `sensing.symbol_stride` | `int` | `20` | dense 模式启动时默认 STRD。 |
| `sensing.paired_frame_queue_size` | `int` | `64` | 每个感知通道的 RX/TX 帧配对队列容量。 |
| `sensing.mask_blocks` | 来自 `resource_preview.mask_blocks` | 可选 | 由 resource preview 生成的运行时感知 mask。 |

### `sensing.rx_channels[]` 字段

| 字段 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `usrp_channel` | `int` | `1` | 当前感知路径使用的 USRP RX 通道。 |
| `device_args` | `string` | `""` | 该通道 device args 覆盖。 |
| `clock_source` | `string` | `""` | 该通道时钟源覆盖。 |
| `time_source` | `string` | `""` | 该通道时间源覆盖。 |
| `wire_format` | `string` | `""` | 该通道 wire-format 覆盖。 |
| `rx_gain` | `float` / dB | `30` | 该通道 RX 增益。 |
| `alignment` | `int` / samples | `63` | 该通道的定时对齐校正量。 |
| `rx_antenna` | `string` | `RX2` | RX 天线端口，如 `RX1`、`RX2`、`TX/RX`。 |
| `enable_system_delay_estimation` | `bool` | `false` | 周期执行基于 ZC 的系统时延估计，并关闭该通道常规感知处理。 |
| `enable_sensing_output` | `bool` | 继承输出开关 | 该通道单站感知输出开关。 |
| `rx_cpu_core` | `int` | `-1` | 该通道 RX loop CPU 核。 |
| `processing_cpu_core` | `int` | `-1` | 该通道感知处理 loop CPU 核。 |

### `resource_preview`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `resource_preview.data_resource_blocks[]` | `object[]` | 可选 | payload / sensing-pilot RE 矩形；每项包含 `kind`、`symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`。 |
| `resource_preview.mask_blocks[]` | `object[]` | 可选 | compact 感知 RE 矩形；每项包含 `symbol_start`、`symbol_count`、`subcarrier_start`、`subcarrier_count`。 |

### `measurement`

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

### `network_output`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `network_output.udp_input_ip` | `string` / IPv4 | `0.0.0.0` | BS 下行业务 UDP 输入绑定 IP。 |
| `network_output.udp_input_port` | `int` | `50000` | BS 下行业务 UDP 输入端口。 |
| `network_output.udp_output_ip` | `string` / IPv4 | `127.0.0.1` | BS 解码上行 UDP 输出目标 IP。 |
| `network_output.udp_output_port` | `int` | `50003` | BS 解码上行 UDP 输出目标端口。 |
| `network_output.udp_egress_pacer_enabled` | `bool` | `false` | 开启解码 UDP 输出速率调节。 |
| `network_output.udp_egress_pacer_target_mbps` | `float` / Mbps | `0` | UDP 输出目标速率；`0` 根据入队速率自动估计。 |
| `network_output.udp_egress_pacer_queue_packets` | `int` | `10240` | UDP 输出调速队列可容纳的数据报数。 |
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

### `cpu_cores`

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `cpu_cores.downlink_cpu_cores` | `int[]` | `[1,2,3,-1]` | BS 下行核心：TX、调制、LDPC 编码、UDP 接收。 |
| `cpu_cores.uplink_cpu_cores` | `int[]` | `[]` | BS 上行核心：RX ingest、OFDM/LLR 处理、LDPC decode + UDP 输出。 |
| `cpu_cores.main_cpu_core` | `int` | `-1` | 主线程 CPU 核；`-1` 表示不绑定。 |

### `logging`

性能剖析输出与诊断日志统一由层级化 logging 过滤（不再使用 `profiling_modules`）。

| 参数 | 类型/单位 | 典型值 | 说明 |
| :--- | :--- | :--- | :--- |
| `logging.default_level` | `string` | `warn` | 全局级别：`off` / `error` / `warn` / `info` / `debug`。 |
| `logging.force_error` | `bool` | `true` | 不可恢复 `Error` 始终输出。 |
| `logging.timestamps` | `bool` | `false` | 行首加本地时间戳 `HH:MM:SS.uuuuuu`。 |
| `logging.modules` | `map` | `{}` | 按模块覆盖。**诊断**模块继承父级（如 `mod`、`sensing`、`ertm`）。**性能**模块以 `_profiling` 结尾（如 `mod_profiling`），默认关闭，需显式打开。 |
