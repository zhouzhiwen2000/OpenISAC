---
title: BS YAML Reference
description: BS runtime configuration fields, values, and behavior.
---

## How to use this page

`BS` reads `BS.yaml` from its current working directory at startup. Copy a `config/BS_*.yaml` preset that matches your hardware and scenario, then edit the copy. Configuration is not hot-reloaded; restart the BS after making changes.

The tables follow the top-level YAML structure and use full paths such as `uplink.arq_enabled` to distinguish similarly named fields. A typical value is guidance, not a guaranteed default for every preset. If an optional section is omitted, the parser supplies its default values.

> Keep coupled frame-structure, duplex, frequency, and resource-mapping fields consistent with the UE configuration.

## Parameter tables

### `radio`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `radio.radio_backend` | `string` | `uhd` | Radio I/O backend. Use `uhd` for real USRPs or `sim` for the shared-memory channel simulator. |

### `simulation`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `simulation.session` | `string` | `oisac_sim` | Shared simulator session namespace used by BS, UE, and `ChannelSimulator`. |
| `simulation.enable_comm_rx` | `bool` | `true` | Simulator produces the communication RX path for UE. |
| `simulation.enable_sensing_rx` | `bool` | `true` | Simulator produces monostatic sensing RX paths. |
| `simulation.enable_uplink` | `bool` | `false` | Simulator routes the UE-to-BS uplink stream. |
| `simulation.pacing_enabled` | `bool` | `true` | Pace simulator output to wall-clock sample time. |
| `simulation.noise_power_dbfs` | `float` / dBFS | `-70` | AWGN power per RX channel; very low values effectively disable noise. |
| `simulation.snr_control_enable` | `bool` | `false` | Scale the clean simulated signal before AWGN to maintain `target_snr_db`. |
| `simulation.target_snr_db` | `float` / dB | `40` | Initial SNR target when SNR control is enabled. |
| `simulation.control_port` | `int` | `10002` | ChannelSimulator ZMQ control port for runtime SNR commands. |
| `simulation.cfo_hz` | `float` / Hz | `0` | Initial BS-to-UE CFO before UE RX correction. The initial UE-to-BS CFO has the opposite sign, is carrier-ratio-scaled in FDD, and its residual then follows UE uplink TX retuning. |
| `simulation.sample_rate_offset_ppm` | `float` / ppm | `0` | UE sample-clock offset relative to the BS clock. |
| `simulation.timing_offset_samples` | `int` / samples | `0` | Constant integer sample delay injected on RX. |
| `simulation.array_spacing_m` | `float` / m | `0.04283` | Physical ULA element spacing; set `<=0` to use `array_spacing_lambda`. |
| `simulation.array_spacing_lambda` | `float` / lambda | `0.5` | Legacy ULA spacing in wavelengths. |
| `simulation.ring_capacity_samples` | `int` / samples | `262144` | Per-stream shared-memory ring capacity. |
| `simulation.steering_override_file` | `string` | `""` | Optional array-manifold matrix file. When empty, the simulator generates a ULA manifold from `angle_deg`. When set, `angle_deg` no longer affects the array response; encode the angle-dependent amplitude and phase directly in the matrix. |
| `simulation.comm_multipath_taps[]` | `object[]` | optional | Communication tapped-delay-line taps with `delay_samples`, `gain_db`, and `phase_deg`. |
| `simulation.targets[]` | `object[]` | optional | Monostatic point scatterers with `range_m`, `velocity_mps`, `gain_db`, and `angle_deg`. |
| `simulation.bistatic_targets[]` | `object[]` | optional | Bistatic/communication point scatterers with the same target fields. |

### `rf_sampling`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `rf_sampling.sample_rate` | `float` / Hz | `50000000` | Baseband sample rate. |
| `rf_sampling.bandwidth` | `float` / Hz | `50000000` | Analog bandwidth, usually matching `sample_rate`. |

### `usrp_device`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `usrp_device.device_args` | `string` | `addr=...` | Shared USRP device args fallback. |

### `clock_time`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `clock_time.clock_source` | `string` | `external` | Global clock source: `internal`, `external`, or `gpsdo`. |
| `clock_time.time_source` | `string` | `internal` | Global time/PPS source; empty follows `clock_source`. |

### `ofdm_frame`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `ofdm_frame.fft_size` | `int` | `1024` | OFDM FFT size. |
| `ofdm_frame.cp_length` | `int` / samples | `128` | Cyclic prefix length. |
| `ofdm_frame.sync_pos` | `int` | `1` | Sync symbol index inside each frame. |
| `ofdm_frame.enable_sec_sync_symbol` | `bool` | `false` | Reserve `sync_pos-1` as a duplicate ZC sync symbol. |
| `ofdm_frame.enable_cfo_training_sequence` | `bool` | `false` | Reserve `sync_pos+1` as a repeated CFO training field. |
| `ofdm_frame.cfo_training_period_samples` | `int` / samples | `16` | Repetition period of the CFO training field; must divide `fft_size`. |
| `ofdm_frame.num_symbols` | `int` | `100` | Number of OFDM symbols per frame. |
| `ofdm_frame.sensing_symbol_num` | `int` | `100` | Number of symbols used in sensing processing. |
| `ofdm_frame.zc_root` | `int` | `29` | Zadoff-Chu root for sync/preamble. |
| `ofdm_frame.pilot_positions` | `int[]` | `[571,...]` | Comb-pilot subcarrier indices. |
| `ofdm_frame.midframe_pilot_symbols` | `int[]` | `[]` | Optional in-frame BPSK pilot symbol indices. |
| `ofdm_frame.midframe_pilot_seed` | `int` | `1296453708` | Deterministic mid-frame BPSK pilot seed; must match TX/RX. |

### `cuda`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `cuda.cuda_mod_pipeline_slots` | `int` | `3` | CUDA modulation pipeline slots; values below `1` are clamped. |

### `ldpc`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `ldpc.fixed_point` | `bool` | `false` | Use the int16/Q16 layered-NMS CPU decoder instead of float32. |
| `ldpc.fixed_point_scale` | `int` | `16` | Power-of-two LLR scale before int16 saturation in fixed-point mode. |

### `downlink`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `downlink.center_freq` | `float` / Hz | `2400000000` | BS downlink RF center frequency. |
| `downlink.tx_gain` | `float` / dB | `60` | BS downlink TX gain. |
| `downlink.tx_channel` | `int` | `0` | BS downlink TX channel index. |
| `downlink.tx_device_args` | `string` | `""` | TX-specific device args; empty uses `usrp_device.device_args`. |
| `downlink.tx_clock_source` | `string` | `""` | TX clock source override. |
| `downlink.tx_time_source` | `string` | `""` | TX time source override. |
| `downlink.wire_format_tx` | `string` | `sc16` | TX wire format, typically `sc16` or `sc8`. |
| `downlink.arq_enabled` | `bool` | `false` | Enable downlink ARQ on the BS transmitter. |
| `downlink.arq_window_packets` | `int` | `256` | Downlink ARQ outstanding packet window; valid range `1`–`32767`. |
| `downlink.arq_retransmit_timeout_ms` | `int` / ms | `100` | Downlink ARQ retransmission timeout. |
| `downlink.arq_max_retries` | `int` | `5` | Max downlink retransmission retries; `0` means unlimited within the window. |

### `downlink_pipeline`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `downlink_pipeline.tx_circular_buffer_size` | `int` | `8` | Capacity of the modulated-frame queue feeding TX. |
| `downlink_pipeline.data_packet_buffer_size` | `int` | `256` | Capacity of the encoded-packet buffer. |

### `uplink`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `uplink.enabled` | `bool` | `false` | Master switch for the UE-to-BS uplink/duplex path. |
| `uplink.duplex_mode` | `string` | `tdd` | `tdd` uses an uplink symbol window; `fdd` uses `uplink.center_freq` and a full-frame uplink. |
| `uplink.center_freq` | `float` / Hz | `2500000000` | FDD-only uplink carrier. TDD uses the downlink center frequency. |
| `uplink.symbol_start` | `int` | `90` | TDD-only first uplink symbol in the downlink frame. |
| `uplink.symbol_count` | `int` | `10` | TDD-only uplink window length; `0` disables TDD uplink. |
| `uplink.guard_symbols` | `int` | `1` | TDD-only leading guard symbols inside the uplink window. |
| `uplink.bs_dl_ul_timing_diff` | `int` / samples | `50` | BS-side uplink RX window offset relative to the downlink TX frame anchor. |
| `uplink.debug_self_channel` | `bool` | `false` | Estimate local TX leakage/self channel from uplink RX windows for `DUTI` debugging. |
| `uplink.ertm_to_enable` | `bool` | `false` | Enable eRTM timing-offset payloads and UE-side TO logs. |
| `uplink.ertm_report_interval_frames` | `int` / frames | `32` | BS eRTM payload/report cadence in downlink TX frames. |
| `uplink.rx_gain` | `float` / dB | `0` | BS uplink RX gain. |
| `uplink.rx_channel` | `int` | `0` | BS uplink RX channel index. |
| `uplink.rx_wire_format` | `string` | `sc16` | BS uplink RX wire format. |
| `uplink.rx_device_args` | `string` | `""` | Uplink RX device args override. |
| `uplink.rx_clock_source` | `string` | `""` | Uplink RX clock source override. |
| `uplink.rx_time_source` | `string` | `""` | Uplink RX time source override. |
| `uplink.rx_agc_enable` | `bool` | `false` | Enable BS uplink hardware RX AGC. |
| `uplink.rx_agc_low_threshold_db` | `float` / dB | `14` | Increase uplink RX gain below this filtered delay-spectrum peak threshold. |
| `uplink.rx_agc_high_threshold_db` | `float` / dB | `16` | Decrease uplink RX gain above this threshold. |
| `uplink.rx_agc_max_step_db` | `float` / dB | `1` | Maximum uplink RX gain step per AGC update. |
| `uplink.rx_agc_update_frames` | `int` | `4` | Minimum processed-uplink-frame interval between AGC updates. |
| `uplink.equalizer_mode` | `string` | `mmse` | BS uplink equalizer inverse mode: `zf` or `mmse`. |
| `uplink.channel_tracking_mode` | `string` | `pilot_phase` | Uplink per-symbol comb-pilot tracking mode: `disabled` or `pilot_phase`. |
| `uplink.equalizer_mag_floor` | `float` | `1e-6` | Lower bound for `|H|^2` in uplink channel inversion. |
| `uplink.channel_tracking_min_pilot_snr` | `float` | `1e-4` | Minimum comb-pilot residual weight before falling back. |
| `uplink.arq_enabled` | `bool` | `false` | Enable uplink ARQ on the BS receiver. |
| `uplink.arq_ordered_delivery` | `bool` | `false` | Buffer accepted uplink packets for in-order UDP delivery. |
| `uplink.arq_window_packets` | `int` | `256` | Uplink ARQ receive/reorder window; valid range `1`–`32767`. |
| `uplink.arq_feedback_interval_ms` | `int` / ms | `10` | Minimum interval between uplink ARQ ACK feedback packets. |

### `sensing`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `sensing.rx_wire_format` | `string` | `sc16` | Default sensing RX wire format. |
| `sensing.rx_device_args` | `string` | `""` | Default sensing RX args. |
| `sensing.rx_clock_source` | `string` | `""` | Default sensing RX clock source override. |
| `sensing.rx_time_source` | `string` | `""` | Default sensing RX time source override. |
| `sensing.rx_channel_count` | `int` | `1` | Number of monostatic sensing RX channels; `0` disables sensing RX. |
| `sensing.rx_channels[]` | `object[]` | see below | Per-channel sensing RX settings. |
| `sensing.range_fft_size` | `int` | `1024` | Range FFT size. |
| `sensing.doppler_fft_size` | `int` | `100` | Doppler FFT size. |
| `sensing.view_range_bins` | `int` | `0` | Backend RD view width; `0` means full `range_fft_size`. |
| `sensing.view_doppler_bins` | `int` | `0` | Backend RD view height; `0` means full `doppler_fft_size`. |
| `sensing.output_mode` | `string` | `dense` | `dense` uses STRD-based full output; `compact_mask` exports selected RE only. |
| `sensing.on_wire_format` | `string` | `complex_float32` | Sensing payload wire format. |
| `sensing.backend_processing_enabled` | `bool` | `false` | Publish backend RD/CFAR/micro-Doppler sidecars when supported. |
| `sensing.symbol_stride` | `int` | `20` | Default dense-mode STRD applied at startup. |
| `sensing.paired_frame_queue_size` | `int` | `64` | Per-channel RX/TX frame-pairing queue capacity. |
| `sensing.mask_blocks` | via `resource_preview.mask_blocks` | optional | Runtime sensing mask derived from resource preview. |

### `sensing.rx_channels[]` fields

| Field | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `usrp_channel` | `int` | `1` | USRP RX channel index for this sensing path. |
| `device_args` | `string` | `""` | Per-channel device args override. |
| `clock_source` | `string` | `""` | Per-channel clock source override. |
| `time_source` | `string` | `""` | Per-channel time source override. |
| `wire_format` | `string` | `""` | Per-channel wire-format override. |
| `rx_gain` | `float` / dB | `30` | Per-channel RX gain. |
| `alignment` | `int` / samples | `63` | Per-channel timing alignment offset. |
| `rx_antenna` | `string` | `RX2` | RX antenna port, such as `RX1`, `RX2`, or `TX/RX`. |
| `enable_system_delay_estimation` | `bool` | `false` | Run periodic ZC-based system-delay estimation and disable normal sensing for this channel. |
| `enable_sensing_output` | `bool` | inherits output switch | Per-channel monostatic output switch. |
| `rx_cpu_core` | `int` | `-1` | CPU core for the channel RX loop. |
| `processing_cpu_core` | `int` | `-1` | CPU core for the channel sensing-processing loop. |

### `resource_preview`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `resource_preview.data_resource_blocks[]` | `object[]` | optional | Payload / sensing-pilot RE rectangles. Each item has `kind`, `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`. |
| `resource_preview.mask_blocks[]` | `object[]` | optional | Compact sensing RE rectangles with `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`. |

### `measurement`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `measurement.measurement_enable` | `bool` | `false` | Enable internal PRBS measurement traffic. |
| `measurement.measurement_mode` | `string` | `internal_prbs` | Measurement generator/checker mode. |
| `measurement.measurement_run_id` | `string` | `""` | Run identifier written into measurement CSV summaries. |
| `measurement.measurement_output_dir` | `string` | `""` | Output directory for measurement CSV summaries. |
| `measurement.measurement_payload_bytes` | `int` / bytes | `1024` | Bytes per generated measurement payload. |
| `measurement.measurement_prbs_seed` | `int` | `0x5A` | Base seed for deterministic PRBS payload contents. |
| `measurement.measurement_packets_per_point` | `int` | `1` | Packets sent for one measurement epoch. |
| `measurement.measurement_max_packets_per_frame` | `int` | `1` | Max measurement packets pulled per frame; `0` means unlimited. |

### `network_output`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `network_output.udp_input_ip` | `string` / IPv4 | `0.0.0.0` | BS downlink payload UDP bind IP. |
| `network_output.udp_input_port` | `int` | `50000` | BS downlink payload UDP bind port. |
| `network_output.udp_output_ip` | `string` / IPv4 | `127.0.0.1` | BS decoded uplink UDP destination IP. |
| `network_output.udp_output_port` | `int` | `50003` | BS decoded uplink UDP destination port. |
| `network_output.udp_egress_pacer_enabled` | `bool` | `false` | Enable queued pacing for decoded UDP egress. |
| `network_output.udp_egress_pacer_target_mbps` | `float` / Mbps | `0` | Egress pacer target rate; `0` auto-estimates from enqueue rate. |
| `network_output.udp_egress_pacer_queue_packets` | `int` | `10240` | Egress pacer queue capacity in datagrams. |
| `network_output.udp_egress_pacer_max_delay_ms` | `float` / ms | `0` | Max queued packet age; `0` disables age drops. |
| `network_output.mono_sensing_output_enabled` | `bool` | `true` | Enable monostatic sensing ZMQ output. |
| `network_output.mono_sensing_ip` | `string` / IPv4 | `0.0.0.0` | Monostatic sensing/control ZMQ bind IP. |
| `network_output.mono_sensing_port` | `int` | `8888` | Monostatic sensing PUB port. |
| `network_output.uplink_channel_ip` | `string` / IPv4 | `0.0.0.0` | BS uplink channel-estimate debug PUB IP. |
| `network_output.uplink_channel_port` | `int` | `12358` | BS uplink channel-estimate debug PUB port. |
| `network_output.uplink_pdf_ip` | `string` / IPv4 | `0.0.0.0` | BS uplink delay-spectrum debug PUB IP. |
| `network_output.uplink_pdf_port` | `int` | `12359` | BS uplink delay-spectrum debug PUB port. |
| `network_output.uplink_constellation_ip` | `string` / IPv4 | `0.0.0.0` | BS uplink constellation debug PUB IP. |
| `network_output.uplink_constellation_port` | `int` | `12356` | BS uplink constellation debug PUB port. |
| `network_output.self_channel_ip` | `string` / IPv4 | `0.0.0.0` | BS self-channel debug PUB IP. |
| `network_output.self_channel_port` | `int` | `12360` | BS self-channel debug PUB port. |
| `network_output.self_pdf_ip` | `string` / IPv4 | `0.0.0.0` | BS self-delay-spectrum debug PUB IP. |
| `network_output.self_pdf_port` | `int` | `12361` | BS self-delay-spectrum debug PUB port. |
| `network_output.ertm_debug_ip` | `string` / IPv4 | `0.0.0.0` | eRTM debug PUB IP. |
| `network_output.ertm_debug_port` | `int` | `12362` | eRTM debug PUB port. |
| `network_output.control_port` | `int` | `9999` | ZMQ ROUTER port for runtime control. |

### `cpu_cores`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `cpu_cores.downlink_cpu_cores` | `int[]` | `[1,2,3,-1]` | BS downlink cores: TX, modulation, LDPC encode, UDP receive. |
| `cpu_cores.uplink_cpu_cores` | `int[]` | `[]` | BS uplink cores: RX ingest, OFDM/LLR processing, LDPC decode + UDP output. |
| `cpu_cores.main_cpu_core` | `int` | `-1` | Main-thread CPU core; `-1` disables binding. |

### `logging`

Profile timing dumps and diagnostic logs are gated by the hierarchical logging filter (not a separate `profiling_modules` key).

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `logging.default_level` | `string` | `warn` | Global verbosity: `off` / `error` / `warn` / `info` / `debug`. |
| `logging.force_error` | `bool` | `true` | Always emit unrecoverable `Error` lines. |
| `logging.timestamps` | `bool` | `false` | Prefix lines with local `HH:MM:SS.uuuuuu`. |
| `logging.modules` | `map` | `{}` | Per-module overrides. **Debug** modules inherit parents (e.g. `mod`, `sensing`, `ertm`). **Performance** modules end with `_profiling` (e.g. `mod_profiling`, `sensing_profiling`) and stay **off** until set explicitly. |
