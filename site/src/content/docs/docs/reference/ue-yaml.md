---
title: UE YAML Reference
description: UE runtime configuration fields, values, and behavior.
---

## How to use this page

`UE` reads `UE.yaml` from its current working directory at startup. Copy a `config/UE_*.yaml` preset that matches your hardware and scenario, then edit the copy. Configuration is not hot-reloaded; restart the UE after making changes.

The tables follow the top-level YAML structure and use full paths such as `uplink.arq_enabled` to distinguish similarly named fields. A typical value is guidance, not a guaranteed default for every preset. If an optional section is omitted, the parser supplies its default values.

> Keep coupled frame-structure, duplex, frequency, and resource-mapping fields consistent with the BS configuration.

## Parameter tables

### `radio`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `radio.radio_backend` | `string` | `uhd` | Radio I/O backend. Use `uhd` for real USRPs or `sim` for the channel simulator. |

### `simulation`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `simulation.session` | `string` | `oisac_sim` | Shared simulator session namespace. |
| `simulation.enable_comm_rx` | `bool` | `true` | Simulator produces the communication RX path. |
| `simulation.enable_sensing_rx` | `bool` | `true` | Simulator produces sensing RX paths. |
| `simulation.enable_uplink` | `bool` | `false` | Simulator routes the UE-to-BS uplink stream. |
| `simulation.pacing_enabled` | `bool` | `true` | Pace simulator output to wall-clock sample time. |
| `simulation.noise_power_dbfs` | `float` / dBFS | `-100` | AWGN power per RX channel. |
| `simulation.snr_control_enable` | `bool` | `false` | Maintain `target_snr_db` by scaling clean simulated signal. |
| `simulation.target_snr_db` | `float` / dB | `40` | Initial target SNR when SNR control is enabled. |
| `simulation.control_port` | `int` | `10002` | ChannelSimulator control port. |
| `simulation.cfo_hz` | `float` / Hz | `0` | Initial BS-to-UE CFO before UE RX correction. The initial UE-to-BS CFO has the opposite sign, is carrier-ratio-scaled in FDD, and its residual then follows UE uplink TX retuning. |
| `simulation.sample_rate_offset_ppm` | `float` / ppm | `0` | UE sample-clock offset relative to BS. |
| `simulation.timing_offset_samples` | `int` / samples | `0` | Constant integer sample delay. |
| `simulation.array_spacing_m` | `float` / m | `0.04283` | Physical ULA spacing. |
| `simulation.array_spacing_lambda` | `float` | `0.5` | Legacy ULA spacing in wavelengths. |
| `simulation.ring_capacity_samples` | `int` | `262144` | Shared-memory ring capacity. |
| `simulation.steering_override_file` | `string` | `""` | Optional array-manifold matrix file. When empty, the simulator generates a ULA manifold from `angle_deg`. When set, `angle_deg` no longer affects the array response; encode the angle-dependent amplitude and phase directly in the matrix. |
| `simulation.comm_multipath_taps[]` | `object[]` | optional | Communication taps with `delay_samples`, `gain_db`, and `phase_deg`. |
| `simulation.targets[]` | `object[]` | optional | Monostatic point scatterers. |
| `simulation.bistatic_targets[]` | `object[]` | optional | Bistatic/communication point scatterers. |

### `rf_sampling`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `rf_sampling.sample_rate` | `float` / Hz | `50000000` | Baseband sample rate. |
| `rf_sampling.bandwidth` | `float` / Hz | `50000000` | Analog bandwidth. |
| `rf_sampling.rx_gain` | `float` / dB | `10` | UE downlink RX gain. |
| `rf_sampling.rx_agc_enable` | `bool` | `true` | Enable UE downlink hardware RX AGC. |
| `rf_sampling.rx_agc_low_threshold_db` | `float` / dB | `14` | Increase RX gain below this filtered delay-spectrum threshold. |
| `rf_sampling.rx_agc_high_threshold_db` | `float` / dB | `16` | Decrease RX gain above this threshold. |
| `rf_sampling.rx_agc_max_step_db` | `float` / dB | `1` | Maximum RX gain step per AGC update. |
| `rf_sampling.rx_agc_update_frames` | `int` | `4` | Minimum processed-frame interval between AGC updates. |

### `usrp_device`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `usrp_device.device_args` | `string` | `addr=...` | USRP device args. |
| `usrp_device.clock_source` | `string` | `external` | UE clock source: `internal`, `external`, or `gpsdo`. |

### `ofdm_frame`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `ofdm_frame.fft_size` | `int` | `1024` | OFDM FFT size. |
| `ofdm_frame.cp_length` | `int` / samples | `128` | Cyclic prefix length. |
| `ofdm_frame.sync_pos` | `int` | `1` | Sync symbol index. |
| `ofdm_frame.enable_sec_sync_symbol` | `bool` | `false` | Expect a duplicate ZC sync symbol at `sync_pos-1`. |
| `ofdm_frame.enable_cfo_training_sequence` | `bool` | `false` | Use the `sync_pos+1` CFO training field to resolve CFO aliases. |
| `ofdm_frame.cfo_training_period_samples` | `int` / samples | `16` | CFO training repetition period. |
| `ofdm_frame.num_symbols` | `int` | `100` | OFDM symbols per frame. |
| `ofdm_frame.sensing_symbol_num` | `int` | `100` | Symbols used for sensing. |
| `ofdm_frame.frame_queue_size` | `int` | `32` | Demodulated RX frame queue capacity. |
| `ofdm_frame.zc_root` | `int` | `29` | Zadoff-Chu root. |
| `ofdm_frame.pilot_positions` | `int[]` | `[571,...]` | Comb-pilot subcarrier indices. |
| `ofdm_frame.midframe_pilot_symbols` | `int[]` | `[]` | Optional mid-frame BPSK pilot symbols. |
| `ofdm_frame.midframe_pilot_seed` | `int` | `1296453708` | Deterministic mid-frame BPSK pilot seed. |

### `cuda`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `cuda.cuda_demod_pipeline_slots` | `int` | `3` | CUDA demodulation pipeline slots. |
| `cuda.cuda_ldpc_decoder_backend` | `string` | `gpu` | CUDA demod LDPC decoder backend: `gpu` or `cpu`. |
| `cuda.cuda_ldpc_worker_buffers` | `int` | `3` | CUDA LDPC async worker batch buffers. |
| `cuda.cuda_ldpc_cross_frame_flush_frames` | `int` | `2` | Max frames accumulated before CUDA LDPC batch decode. |
| `cuda.cuda_ldpc_cross_frame_flush_us` | `float` / us | `1000` | Max CUDA LDPC cross-frame batch wait time. |

### `ldpc`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `ldpc.fixed_point` | `bool` | `false` | Use int16/Q16 CPU LDPC decode path. |
| `ldpc.fixed_point_scale` | `int` | `16` | LLR scale before int16 saturation. |

### `downlink`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `downlink.center_freq` | `float` / Hz | `2400000000` | UE downlink RF center frequency. |
| `downlink.rx_wire_format` | `string` | `sc16` | UE downlink RX wire format. |
| `downlink.rx_channel` | `int` | `0` | UE downlink RX channel index. |
| `downlink.equalizer_mode` | `string` | `mmse` | Downlink equalizer inverse mode: `zf` or `mmse`. |
| `downlink.channel_tracking_mode` | `string` | `pilot_phase` | Per-symbol comb-pilot tracking mode. |
| `downlink.equalizer_mag_floor` | `float` | `1e-6` | Lower bound for `|H|^2` in channel inversion. |
| `downlink.channel_tracking_min_pilot_snr` | `float` | `1e-4` | Minimum comb-pilot residual weight before fallback. |
| `downlink.arq_enabled` | `bool` | `false` | Enable downlink ARQ on the UE receiver. |
| `downlink.arq_ordered_delivery` | `bool` | `false` | Buffer downlink packets for in-order UDP delivery. |
| `downlink.arq_window_packets` | `int` | `32767` | Downlink ARQ receive/reorder window. |
| `downlink.arq_feedback_interval_ms` | `int` / ms | `10` | Minimum interval between downlink ARQ ACK feedback packets. |

### `uplink`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `uplink.enabled` | `bool` | `false` | Master switch for UE uplink/duplex. |
| `uplink.duplex_mode` | `string` | `tdd` | Must match BS: `tdd` windowed uplink or `fdd` full-frame uplink. |
| `uplink.center_freq` | `float` / Hz | `2500000000` | FDD-only uplink carrier. TDD uses the downlink center frequency. |
| `uplink.idle_waveform` | `string` | `random_qpsk` | UE uplink idle waveform: `random_qpsk` or `zero`. |
| `uplink.symbol_start` | `int` | `90` | TDD-only first uplink symbol. |
| `uplink.symbol_count` | `int` | `10` | TDD-only uplink window length. |
| `uplink.guard_symbols` | `int` | `1` | TDD-only leading guard symbols. |
| `uplink.tx_gain` | `float` / dB | `0` | UE uplink TX gain. |
| `uplink.tx_channel` | `int` | `0` | UE uplink TX channel index. |
| `uplink.wire_format_tx` | `string` | `sc16` | UE uplink TX wire format. |
| `uplink.ue_timing_advance` | `int` / samples | `50` | UE uplink transmit timing advance. |
| `uplink.debug_self_channel` | `bool` | `false` | Estimate UE self-TX leakage channel from RX windows for `TADV` debugging. |
| `uplink.ertm_to_enable` | `bool` | `false` | Enable eRTM TO payload consumption and TO logs. |
| `uplink.ertm_delay_oversample_factor` | `int` | `10` | eRTM delay-spectrum IFFT oversampling factor. |
| `uplink.ertm_dl_rf_delay_samples` | `float` / samples | `67.0` | Calibrated downlink RF-chain delay for eRTM equations. |
| `uplink.ertm_ul_rf_delay_samples` | `float` / samples | `67.0` | Calibrated uplink RF-chain delay for eRTM equations. |
| `uplink.ertm_debug_output_enabled` | `bool` | `false` | Enable UE-side eRTM debug ZMQ spectra. |
| `uplink.ertm_report_interval_frames` | `int` / frames | `32` | BS eRTM report cadence; keep matched with BS. |
| `uplink.arq_enabled` | `bool` | `false` | Enable uplink ARQ on the UE transmitter. |
| `uplink.arq_window_packets` | `int` | `32767` | Uplink ARQ outstanding packet window. |
| `uplink.arq_retransmit_timeout_ms` | `int` / ms | `100` | Uplink ARQ retransmission timeout. |
| `uplink.arq_max_retries` | `int` | `5` | Max uplink retransmission retries; `0` means unlimited within the window. |

### `sync_tracking`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `sync_tracking.sync_queue_size` | `int` | `32` | Sync-search batch queue capacity. |
| `sync_tracking.sync_cfo_alias_search_range_hz` | `float` / Hz | `800000` | CFO range covered by the sync alias resolver. |
| `sync_tracking.reset_hold_s` | `float` / s | `0.5` | Invalid-delay duration before a hard reset to sync search. |
| `sync_tracking.software_sync` | `bool` | `true` | Enable software sync tracking. |
| `sync_tracking.predictive_delay` | `bool` | `true` | Use CFO-based predictive delay compensation. |
| `sync_tracking.hardware_sync` | `bool` | `false` | Enable hardware synchronization mode. |
| `sync_tracking.hardware_sync_tty` | `string` | `/dev/ttyUSB0` | Serial device for the hardware sync controller. |
| `sync_tracking.ocxo_pi_kp_fast` | `float` | `30` | Fast-stage OCXO PI proportional gain. |
| `sync_tracking.ocxo_pi_ki_fast` | `float` | `1` | Fast-stage OCXO PI integral gain. |
| `sync_tracking.ocxo_pi_kp_slow` | `float` | `30` | Slow-stage OCXO PI proportional gain. |
| `sync_tracking.ocxo_pi_ki_slow` | `float` | `0.05` | Slow-stage OCXO PI integral gain. |
| `sync_tracking.ocxo_pi_switch_abs_error_ppm` | `float` / ppm | `0.0002` | Error threshold for switching to slow OCXO PI stage. |
| `sync_tracking.ocxo_pi_switch_hold_s` | `float` / s | `60` | Hold time below threshold before switching stages. |
| `sync_tracking.ocxo_pi_max_step_fast_ppm` | `float` / ppm | `0.01` | Fast-stage maximum OCXO adjustment per update. |
| `sync_tracking.ocxo_pi_max_step_slow_ppm` | `float` / ppm | `0.01` | Slow-stage maximum OCXO adjustment per update. |
| `sync_tracking.ocxo_pi_max_step_ppm` | `float` / ppm | optional | Legacy alias applied to both fast and slow max-step fields. |
| `sync_tracking.akf_enable` | `bool` | `true` | Enable adaptive Kalman filter on hardware-sync `error_ppm`. |
| `sync_tracking.akf_bootstrap_frames` | `int` | `64` | Cold-start frames before normal AKF updates. |
| `sync_tracking.akf_innovation_window` | `int` | `64` | Innovation history window for adaptation. |
| `sync_tracking.akf_max_lag` | `int` | `4` | Maximum innovation autocorrelation lag. |
| `sync_tracking.akf_adapt_interval` | `int` | `64` | Frame interval for adaptive `Q/R` updates. |
| `sync_tracking.akf_gate_sigma` | `float` | `3` | Innovation gate in sigma units. |
| `sync_tracking.akf_tikhonov_lambda` | `float` | `1e-3` | Tikhonov regularization for LS adaptation. |
| `sync_tracking.akf_update_smooth` | `float` | `0.2` | Exponential smoothing for updated `Q/R`. |
| `sync_tracking.akf_q_wf_min` | `float` | `1e-10` | White-frequency-noise lower bound. |
| `sync_tracking.akf_q_wf_max` | `float` | `1e2` | White-frequency-noise upper bound. |
| `sync_tracking.akf_q_rw_min` | `float` | `1e-12` | Random-walk-frequency-noise lower bound. |
| `sync_tracking.akf_q_rw_max` | `float` | `1e1` | Random-walk-frequency-noise upper bound. |
| `sync_tracking.akf_r_min` | `float` | `1e-8` | Observation-noise variance lower bound. |
| `sync_tracking.akf_r_max` | `float` | `1e3` | Observation-noise variance upper bound. |
| `sync_tracking.desired_peak_pos` | `int` | `20` | Target delay-peak position for alignment logic. |

### `sensing`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `sensing.sensing_delay_correction_mode` | `string` | `los_tracking` | Bistatic sensing delay correction source: `los_tracking` or `ertm_absolute`. |
| `sensing.bi_enabled` | `bool` | `true` | Enable bistatic sensing processing. |
| `sensing.range_fft_size` | `int` | `1024` | Range FFT size. |
| `sensing.doppler_fft_size` | `int` | `100` | Doppler FFT size. |
| `sensing.view_range_bins` | `int` | `0` | Backend RD view width; `0` means full range. |
| `sensing.view_doppler_bins` | `int` | `0` | Backend RD view height; `0` means full Doppler size. |
| `sensing.output_mode` | `string` | `dense` | `dense` full-buffer output or `compact_mask` selected-RE output. |
| `sensing.on_wire_format` | `string` | `complex_float32` | Sensing payload wire format. |
| `sensing.backend_processing_enabled` | `bool` | `false` | Publish backend RD/CFAR/micro-Doppler sidecars when supported. |
| `sensing.symbol_stride` | `int` | `20` | Default dense-mode STRD applied at startup. |

### `resource_preview`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `resource_preview.data_resource_blocks[]` | `object[]` | optional | Payload / sensing-pilot RE rectangles; each item has `kind`, `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`. |
| `resource_preview.mask_blocks[]` | `object[]` | optional | Compact sensing RE rectangles with `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`. |

### `measurement`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `measurement.measurement_enable` | `bool` | `false` | Enable internal PRBS measurement checking. |
| `measurement.measurement_mode` | `string` | `internal_prbs` | Measurement checker mode. |
| `measurement.measurement_run_id` | `string` | `""` | Run ID written to measurement CSV summaries. |
| `measurement.measurement_output_dir` | `string` | `""` | Output directory for measurement CSV summaries. |
| `measurement.measurement_payload_bytes` | `int` / bytes | `1024` | Expected bytes per measurement payload. |
| `measurement.measurement_prbs_seed` | `int` | `0x5A` | Base seed for rebuilding deterministic PRBS payloads. |
| `measurement.measurement_packets_per_point` | `int` | `1` | Expected packets for one measurement epoch. |
| `measurement.measurement_max_packets_per_frame` | `int` | `1` | Max measurement packets checked per frame; `0` means unlimited. |

### `network_output`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `network_output.udp_input_ip` | `string` / IPv4 | `0.0.0.0` | UE uplink payload UDP bind IP. |
| `network_output.udp_input_port` | `int` | `50002` | UE uplink payload UDP bind port. |
| `network_output.udp_output_ip` | `string` / IPv4 | `""` | UE decoded downlink UDP destination IP; empty uses `runtime.default_out_ip`. |
| `network_output.udp_output_port` | `int` | `50001` | UE decoded downlink UDP destination port. |
| `network_output.udp_egress_pacer_enabled` | `bool` | `false` | Enable queued pacing for decoded UDP egress. |
| `network_output.udp_egress_pacer_target_mbps` | `float` / Mbps | `0` | Egress pacer target rate; `0` auto-estimates. |
| `network_output.udp_egress_pacer_queue_packets` | `int` | `10240` | Egress pacer queue capacity. |
| `network_output.udp_egress_pacer_max_delay_ms` | `float` / ms | `0` | Max queued packet age; `0` disables age drops. |
| `network_output.bi_sensing_output_enabled` | `bool` | `true` | Enable bistatic sensing ZMQ output. |
| `network_output.bi_sensing_ip` | `string` / IPv4 | `0.0.0.0` | Bistatic sensing/control ZMQ bind IP. |
| `network_output.bi_sensing_port` | `int` | `8889` | Bistatic sensing PUB port. |
| `network_output.channel_ip` | `string` / IPv4 | `0.0.0.0` | Channel-estimate PUB IP. |
| `network_output.channel_port` | `int` | `12348` | Channel-estimate PUB port. |
| `network_output.pdf_ip` | `string` / IPv4 | `0.0.0.0` | Delay-spectrum PUB IP. |
| `network_output.pdf_port` | `int` | `12349` | Delay-spectrum PUB port. |
| `network_output.constellation_ip` | `string` / IPv4 | `0.0.0.0` | Constellation PUB IP. |
| `network_output.constellation_port` | `int` | `12346` | Constellation PUB port. |
| `network_output.self_channel_ip` | `string` / IPv4 | `0.0.0.0` | UE self-channel debug PUB IP. |
| `network_output.self_channel_port` | `int` | `12350` | UE self-channel debug PUB port. |
| `network_output.self_pdf_ip` | `string` / IPv4 | `0.0.0.0` | UE self-delay-spectrum debug PUB IP. |
| `network_output.self_pdf_port` | `int` | `12351` | UE self-delay-spectrum debug PUB port. |
| `network_output.ertm_debug_ip` | `string` / IPv4 | `0.0.0.0` | UE eRTM debug PUB IP. |
| `network_output.ertm_debug_port` | `int` | `12362` | UE eRTM debug PUB port. |
| `network_output.control_port` | `int` | `10001` | ZMQ ROUTER port for runtime control. |

### `cpu_cores`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `cpu_cores.downlink_cpu_cores` | `int[]` | `[1,2,3]` | UE downlink cores: RX, processing, bit processing. |
| `cpu_cores.demod_worker_cpu_cores` | `int[]` | `[]` | UE CPU demod worker cores; empty starts one unbound worker. |
| `cpu_cores.ldpc_worker_cpu_cores` | `int[]` | `[]` | UE CPU LDPC decode worker cores; empty starts one unbound worker. |
| `cpu_cores.sensing_cpu_cores` | `int[]` | `[4]` | UE bistatic sensing cores. |
| `cpu_cores.uplink_cpu_cores` | `int[]` | `[]` | UE uplink cores: LDPC encode, modulation, TX send, UDP receive. |
| `cpu_cores.main_cpu_core` | `int` | `-1` | Main-thread CPU core; `-1` disables binding. |

### `runtime`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `runtime.default_out_ip` | `string` / IPv4 | `127.0.0.1` | Default destination IP for UDP and VOFA+ outputs when specific IP fields are empty. |
| `runtime.vofa_debug_ip` | `string` / IPv4 | `""` | VOFA+ debug destination IP; empty uses `default_out_ip`. |
| `runtime.vofa_debug_port` | `int` | `12347` | VOFA+ debug destination port. |

### `logging`

Same hierarchical filter as BS (`logging.*`). **Debug** examples: `ertm: debug`, `recovery: info`, `demod.snr: info`. **Performance** (paths end with `_profiling`, off until set): `demod_profiling: info`, `demod_eq_profiling: info`.

Resource-map notes:
* `resource_preview.data_resource_blocks` should normally match between BS and UE, including `kind`.
* Built-in ZC sync symbols, the optional CFO training field, comb pilots, and mid-frame BPSK pilots take precedence over configured resource blocks.
* `resource_preview.mask_blocks` controls compact sensing export only. In `output_mode=compact_mask`, runtime `STRD` is ignored because the mask defines the sampling pattern.
* Compact sensing payloads begin with `CompactSensingFrameHeader`, followed by fixed-order raw `complex<float>` samples.
