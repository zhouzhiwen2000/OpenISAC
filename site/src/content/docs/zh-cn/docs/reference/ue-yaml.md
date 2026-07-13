---
title: UE YAML 参考
description: UE 运行时配置的分组参考。
---

## 参数说明

运行时配置采用层级化 YAML。下面的表格使用完整 YAML 参数名，例如 `uplink.arq_enabled`，避免 BS/UE 或下行/上行里的同名字段混淆。可选 section 可以省略；缺省值由解析器补齐，`config/` 下的样例文件给出了常用硬件、双工和仿真配置。

### UE

`UE` 从当前工作目录读取 `UE.yaml`。可从 `config/UE_X310.yaml`、`config/UE_B210.yaml`、双工模板或仿真模板开始修改。

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
