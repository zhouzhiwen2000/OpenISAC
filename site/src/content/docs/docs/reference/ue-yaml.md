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
| `radio.radio_backend` | `string` | `uhd` | Radio I/O backend. Use `uhd` for USRPs, `sim` for the simulator, or `xr1710g` for the CPU receive-only UE. |
| `radio.xr1710g_shm` | `string` | `/xr1710g-live` | POSIX shared-memory name of the external single-channel host receiver. |
| `radio.xr1710g_center_freq` | `float` / Hz | `2442000000` | Required actual source LO, read from the XR1710G controller; not an RF tuning command. |
| `radio.xr1710g_attach_timeout_s` | `float` / s | `5` | Wait for a live host receiver, within 0–3600 seconds. |
| `radio.xr1710g_pll_command` | `string` | `""` | Optional persistent PLL helper command, executed on the receiver host. Empty uses software NCO; configured uses hardware LO fine correction within ±100 kHz. See setup below. |

See [XR1710G setup](../../getting-started/hardware/#xr1710g-as-a-receive-only-ue) for supported rates, source ownership and timing limits.

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
| `usrp_device.time_source` | `string` | `internal` | UE time/PPS source. Empty resolves to `internal`; `pps_phase` requires this and `clock_source` to both be `external`. |

Whenever `usrp_device.clock_source` is non-empty, CPU, CUDA, and HIP UE perform
strict device-time alignment before any stream starts, independently of the
selected `sync_tracking.ocxo_mode`. This includes `time_source: internal`;
`time_source` selects the PPS input but does not disable startup alignment.
Startup observes one PPS transition, arms `set_time_next_pps(0)`, and waits until
the device time counter actually resets and `time_last_pps=0`. A missing PPS,
unsupported backend, or failed verification aborts startup. With `gpsdo`, alignment intentionally
replaces any UTC-like device-time epoch with zero. It aligns the software time
origin but retains the X310's 5 ns PPS capture quantization at a 200 MHz master
clock.

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
| `cuda.cuda_precision_profile` | `string` | `fp32` | `fp32` preserves the existing path. `mixed_fp16` uses FP16 for bulk OFDM FFT/IFFT and GPU LDPC messages while retaining FP32 for channel estimation, tracking, equalization, sensing, and LLR generation. Unsupported FP16 FFT shapes fall back to FP32. |
| `cuda.cuda_demod_pipeline_slots` | `int` | `3` | CUDA demodulation pipeline slots. |
| `cuda.cuda_ldpc_decoder_backend` | `string` | `gpu` | CUDA demod LDPC decoder backend: `gpu` or `cpu`. |
| `cuda.cuda_ldpc_worker_buffers` | `int` | `3` | CUDA LDPC async worker batch buffers. |
| `cuda.cuda_ldpc_cross_frame_flush_frames` | `int` | `2` | Max frames accumulated before CUDA LDPC batch decode. |
| `cuda.cuda_ldpc_cross_frame_flush_us` | `float` / us | `1000` | Max CUDA LDPC cross-frame batch wait time. |

### `ldpc`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `ldpc.decoder_iterations` | `int` | `12` | Maximum CPU LDPC iterations, at least 1; syndrome early stopping stays enabled. |
| `ldpc.nms_normalization_factor` | `float` | `0.75` | CPU NMS normalization factor, finite and in `(0, 1]`; Q16 requires multiples of `0.125`. |
| `ldpc.fixed_point` | `bool` | `false` | Use int16/Q16 CPU LDPC decode path. |
| `ldpc.fixed_point_scale` | `int` | `16` | LLR scale before int16 saturation. |

These fields configure CPU decoding on the UE downlink, for both float32 and Q16. Omitting them selects 12 iterations and a factor of 0.75. Explicit YAML values remain in effect; update both fields to use these defaults in an existing config. Edit them under **LDPC decode** in the [Web Config Console](/docs/tools-workflows/web-config-console/), save, and restart the backend. Q16 supports `0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0`; the LLR scale applies only when `fixed_point: true`. More iterations may improve error correction while increasing computation for weak signals.

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
| `downlink.arq_window_packets` | `int` | `256` | Downlink ARQ receive/reorder window; valid range `1`–`32767`. |
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
| `uplink.ertm_timing_metric` | `string` | `delay_magnitude` | Differential-TO metric: phase-robust `delay_magnitude` or coherent `maximum_likelihood`. |
| `uplink.ertm_delay_oversample_factor` | `int` | `10` | eRTM delay-spectrum IFFT oversampling factor. |
| `uplink.ertm_dl_rf_delay_samples` | `float` / samples | `67.0` | Calibrated downlink RF-chain delay for eRTM equations. |
| `uplink.ertm_ul_rf_delay_samples` | `float` / samples | `67.0` | Calibrated uplink RF-chain delay for eRTM equations. |
| `uplink.ertm_debug_output_enabled` | `bool` | `false` | Enable UE-side eRTM debug ZMQ spectra. |
| `uplink.ertm_report_interval_frames` | `int` / frames | `32` | BS eRTM report cadence; keep matched with BS. |
| `uplink.arq_enabled` | `bool` | `false` | Enable uplink ARQ on the UE transmitter. |
| `uplink.arq_window_packets` | `int` | `256` | Uplink ARQ outstanding packet window; valid range `1`–`32767`. |
| `uplink.arq_retransmit_timeout_ms` | `int` / ms | `100` | Uplink ARQ retransmission timeout. |
| `uplink.arq_max_retries` | `int` | `5` | Max uplink retransmission retries; `0` means unlimited within the window. |

With timed uplink enabled on a real radio, CPU, CUDA, and HIP UE startup queries
the device's actual RX and TX sample rates. A difference greater than 1 Hz aborts
startup before worker threads are launched because exact uplink identity mapping
requires both directions to use the same sample-time grid.

### `sync_tracking`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `sync_tracking.sync_queue_size` | `int` | `32` | Sync-search batch queue capacity. |
| `sync_tracking.sync_cfo_alias_search_range_hz` | `float` / Hz | `800000` | CFO range covered by the sync alias resolver. |
| `sync_tracking.reset_hold_s` | `float` / s | `0.5` | Consecutive invalid-delay or preamble-mismatch duration before a hard reset to sync search. |
| `sync_tracking.software_sync` | `bool` | `true` | Enable software sync tracking. |
| `sync_tracking.predictive_delay` | `bool` | `true` | Use CFO-based predictive delay compensation. |
| `sync_tracking.hardware_sync` | `bool` | `false` | Enable hardware OCXO discipline. |
| `sync_tracking.hardware_sync_tty` | `string` | `/dev/ttyUSB0` | Serial device for the OCXO DAC controller. |
| `sync_tracking.ocxo_mode` | `string` | `auto` | `auto`, `freq`, and `phase` retain the legacy TO-based behavior; `pps_phase` explicitly selects PPS phase locking. |
| `sync_tracking.ocxo_dac_ppm_per_lsb` | `float` | `9.71e-7` | Calibrated DAC ppm per LSB (MV197 map). |
| `sync_tracking.ocxo_dac_zero_word` | `int` | `493367` | DAC word for 0 ppm. |
| `sync_tracking.ocxo_dac_state_file` | `string` | `.openisac_ocxo_dac_state.yaml` | Persist the last SF word acknowledged by the DAC. A relative path is resolved from the UE working directory; an empty value disables restart persistence. |
| `sync_tracking.ocxo_max_abs_ppm` | `float` / ppm | `0.45` | Absolute OCXO command clamp. |
| `sync_tracking.ocxo_actuator_max_slew_ppm_per_s` | `float` / ppm/s | `0.01` | Hard actuator slew limit; applied steps are fed back to the controller. |
| `sync_tracking.ocxo_coarse_*` | various | see presets | One-target startup correction: Mode A averages residual CFO; Mode B regresses Alignment-continuous `TO_clock` against absolute frame sequence over the coarse window. |
| `sync_tracking.ocxo_fll_fast_*`, `ocxo_fll_slow_*` | various | see presets | Fast/slow FLL on residual CFO in Mode A or rolling `TO_clock` regression SFO in Mode B, with threshold hysteresis and dwell. |
| `sync_tracking.ocxo_fll_slow_kp` | `float` | `0.1` | Slow FLL proportional gain. |
| `sync_tracking.ocxo_fll_slow_ki` | `float` | `0.007` | Slow FLL integral gain. |
| `sync_tracking.ocxo_fll_slow_max_slew_ppm_per_s` | `float` / ppm/s | `0.005` | Slow FLL command slew limit. |
| `sync_tracking.ocxo_fll_slow_enter_abs_ppm` | `float` / ppm | `0.0005` | Residual threshold for entering Slow FLL from Fast FLL. |
| `sync_tracking.ocxo_phase_setpoint_mode` | `string` | `freeze_on_lock` | Legacy `auto`/`phase` target: freeze `TO_clock` on Phase entry, use zero, or use a fixed target. |
| `sync_tracking.ocxo_phase_setpoint_fixed_samples` | `float` / samples | `0.0` | Legacy fixed `TO_clock` target when the setpoint mode is `fixed`. |
| `sync_tracking.ocxo_phase_*` | various | see presets | Velocity-form Phase PI. Its error is `setpoint-TO_clock` in legacy `auto`/`phase`, or wrapped `P_U-P_B` in `pps_phase`; entry/exit still uses `TO_clock` SFO regression. |
| `sync_tracking.ocxo_pps_initial_ap_enable` | `bool` | `true` | Legacy key name. In `pps_phase`, start one PPSOUT alignment transaction on every `FllSlow -> Phase` entry before releasing SF phase tracking. Disabling it emits a startup warning because a large residual may converge very slowly through the slew-limited PI. |
| `sync_tracking.ocxo_pps_initial_ap_max_abs_ns` | `float` / ns | `500000000` | Absolute AP clamp, matching the PPSDO hardware range. |
| `sync_tracking.ocxo_pps_initial_ap_settle_s` | `float` / s | `2.0` | Minimum AP settling time; a newer eRTM/PPS observation is also required. |
| `sync_tracking.ocxo_pps_initial_ap_command_timeout_s` | `float` / s | `3.0` | Total AP worker wall-clock budget, including all `JAMSYNCING` retries. The aligner waits a 0.5 s terminal-status grace; no AP copy is sent after the worker deadline. |
| `sync_tracking.ocxo_pps_initial_ap_max_retries` | `int` | `1` | Bounded retries (`0..3`) when AP was accepted but no fresh post-AP publication arrived by `settle_s + command_timeout_s`. A retry waits for a new valid publication and recomputes AP from its residual. |
| `sync_tracking.ocxo_pps_initial_ap_deadzone_ns` | `float` / ns | `10.0` | Skip AP and the settling wait when the PPS error at Phase entry is already within this range. |
| `sync_tracking.ocxo_pps_initial_ap_polarity` | `int` | `1` | AP command sign for a positive `P_U-P_B` error. The default is the polarity confirmed by the PPSOUT bench measurement; `-1` remains available for different hardware/firmware polarity. |
| `sync_tracking.cfo_ewma_enable` | `bool` | `true` | EWMA pre-filter on residual CFO ppm for Mode A Coarse/FLL. |
| `sync_tracking.cfo_ewma_alpha` | `float` | `0.02` | EWMA weight on new residual samples (`0..1`; higher = faster, noisier). |
| `sync_tracking.desired_peak_pos` | `int` | `20` | Target delay-peak position for alignment logic. |

The hardware presets use `ocxo_fll_slow_kp=0.1`, `ocxo_fll_slow_ki=0.007`,
`ocxo_fll_slow_max_slew_ppm_per_s=0.005`, and
`ocxo_fll_slow_enter_abs_ppm=0.0005`. They retain
`ocxo_fll_slow_deadzone_ppm=5.0e-5`, `ocxo_phase_kp=2.0e-4`,
`ocxo_phase_ki=1.0e-6`, and `ocxo_phase_enter_sfo_abs_ppm=2.0e-4`.
The PPSOUT AP correction is enabled on every Phase entry and uses the
bench-confirmed positive command polarity by default.

Normal synchronization reacquisition resets OCXO observations, dwell timers,
and loop stages without writing the DAC zero word. The discipliner restarts at
the actuator's retained command. The actuator rebases its slew timer at the
first actual nonzero post-lock correction, so time spent searching cannot
inflate that correction. After an `SF` command receives `OK`, the non-real-time
serial worker durably updates
`ocxo_dac_state_file`. On the next UE/CUDAUE process start, that acknowledged
raw word initializes both the DAC and controller instead of starting at zero.
During graceful shutdown, the serial worker is stopped first and the last SF
word that actually received `OK` is saved again; a newer unacknowledged queued
word is never promoted to the steady-state restart value.
Restore also requires the saved serial device, `zero_word`, and `ppm_per_lsb`
to match the active configuration. A missing file starts from the calibrated
zero word and is created after an acknowledged command. A corrupt, mismatched,
or out-of-range file also falls back to zero, but remains unchanged and disables
persistence for that process run; fix or remove it before restarting. This
assumes the external DAC itself remains powered and retains its word while the
UE process restarts.

`auto` and `phase` preserve the existing Phase anchor: the PI controls
`setpoint - TO_clock`, including `freeze_on_lock`, `zero`, and `fixed`. They do
not read `time_last_pps()` or run the AP gate.

`pps_phase` is a separate explicit mode. For downlink frame `N`, eRTM
reconstructs `r_B=B_N-P_B` from the BS anchor refreshed immediately before each
ERTM5 report is packed, and obtains
`r_U=U_N-P_U` from the UE frame-boundary and PPS timestamps. Its Phase error is
`wrap_1s(tau_DL_RF - TO_UE + r_B - r_U) = wrap_1s(P_U-P_B)`, with target zero.
This mode is rejected unless `sync_tracking.hardware_sync=true`,
`uplink.ertm_to_enable=true`, and both `usrp_device.clock_source` and
`usrp_device.time_source` are explicitly `external`; an empty `time_source`
means `internal`. The BS must likewise use the PPSDO-connected external clock
and time sources when producing the PPS anchor. The BS anchor refresh runs at
the eRTM report cadence configured by
`uplink.ertm_report_interval_frames`, outside the TX/modulation real-time path.
On every `FllSlow -> Phase` transition, the AP
gate is rearmed and computes one correction from the latest valid PPS residual.
It remains done throughout that Phase residence, so normal Phase updates and
Freeze do not repeat AP. If Phase falls back through FLL and later re-enters,
one new correction is allowed. `AP` changes PPSDO PPSOUT only and never resets
USRP time, reanchors the air-frame clock, clears eRTM state, or rebuilds TO. If
the previous AP is still queued or in flight, the new entry waits for hardware
`Idle` instead of treating it as a terminal queue failure. `JAMSYNCING` retries
reuse the same AP request.

### `sensing`

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `sensing.sensing_delay_correction_mode` | `string` | `los_tracking` | Bistatic sensing delay correction source: `los_tracking` or `ertm_absolute`. |
| `sensing.bi_enabled` | `bool` | `true` | Enable bistatic sensing processing. |
| `sensing.bistatic_phase_reference` | `bool` | `false` | LoS phase reference; improves sensing with less stable clocks. Requires LoS and a transmitter/receiver pair stationary relative to each other; restart UE to apply. |
| `sensing.phase_reference_min_coherence` | `float` | `0.05` | Minimum reference coherence in `(0, 1]`; effective when phase reference is enabled. |
| `sensing.range_fft_size` | `int` | `1024` | Range FFT size. |
| `sensing.doppler_fft_size` | `int` | `100` | Doppler FFT size. |
| `sensing.view_range_bins` | `int` | `0` | Backend RD view width; `0` means full range. |
| `sensing.view_doppler_bins` | `int` | `0` | Backend RD view height; `0` means full Doppler size. |
| `sensing.output_mode` | `string` | `dense` | `dense` full-buffer output or `compact_mask` selected-RE output. |
| `sensing.on_wire_format` | `string` | `complex_float32` | Sensing payload wire format. |
| `sensing.backend_processing_enabled` | `bool` | `false` | Publish backend RD/CFAR/micro-Doppler sidecars when supported. |
| `sensing.symbol_stride` | `int` | `20` | Default dense-mode STRD applied at startup. |


#### LoS phase reference

This option can improve bistatic sensing coherence and Doppler stability when independent or less stable transmitter/receiver clocks leave residual common phase drift after synchronization. **It requires a stable line-of-sight (LoS) path between the transmitter and receiver, and the two radios must remain stationary relative to each other.** Sensing targets may still move. Do not enable it for NLoS-only links or when the transmitter and receiver move relative to each other: motion of the reference path would also be subtracted from the measured Doppler.

On the UE Linux host, merge these fields into the active `UE.yaml`:

```yaml
sensing:
  bi_enabled: true
  output_mode: dense
  sensing_delay_correction_mode: los_tracking
  bistatic_phase_reference: true
  phase_reference_min_coherence: 0.05
```

This is a **startup YAML option**, also editable in the Web Config Console. Save the YAML and restart the UE to apply it; there is currently no online enable/disable command for this option.

The correction removes residual common phase from bistatic CSI in `dense` output mode after CFO/SFO/delay and system-response correction, before MTI.

Each slow-time row uses a Hamming-weighted LoS reference and a unit-magnitude conjugate rotation on all subcarriers. The reference follows the known `ALGN` display translation: it cancels that phase ramp only while measuring the reference, and retains the requested delay shift in the output. A translation clears the incomplete sensing batch and MTI/micro-Doppler history so the two display coordinates are not mixed. Amplitudes are preserved. The direct path must be static and dominant: the displayed Doppler is relative to this path; a moving reference also subtracts its own motion. This does not correct the radio sample clock.

`sensing.phase_reference_min_coherence` defaults to `0.05`, with range `(0, 1]`. Coherence is `|sum(w H)|² / (sum(w) sum(w |H|²))`, where `w` is the Hamming weight and `H` is evaluated after cancelling the known display phase ramp. A weak, near-zero or non-finite reference rejects the entire sensing batch and clears MTI/micro-Doppler history. Communication samples are unaffected. Missing slow-time samples reset the sensing window and history to keep Doppler processing on a uniform timeline.

Restart the UE after changing these YAML fields. On the UE Linux host, check the startup log for `static LoS phase reference enabled before MTI`. Frequent `weak/invalid LoS phase reference` messages indicate that this path is unsuitable as a reference; disable the feature or improve the direct-path signal.

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
| `network_output.ocxo_debug_ip` | `string` / IPv4 | `0.0.0.0` | OCXO debug ZMQ PUB bind IP. |
| `network_output.ocxo_debug_port` | `int` | `12347` | OCXO debug ZMQ PUB port. |
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
| `runtime.default_out_ip` | `string` / IPv4 | `127.0.0.1` | Default destination IP for UDP outputs when a specific IP field is empty. |

When hardware sync is enabled, the `OCXODBG3` ZeroMQ packet retains these 16 fixed channels:

| Channel | Signal | Channel | Signal |
| :--- | :--- | :--- | :--- |
| 0 | modulo CFO (Hz) | 8 | active-window `TO_clock` regression SFO (ppm) |
| 1 | de-aliased CFO (Hz) | 9 | cumulative Alignment (samples) |
| 2 | raw residual CFO (ppm) | 10 | Alignment feedforward (samples) |
| 3 | filtered control CFO (ppm) | 11 | desired OCXO delta (ppm) |
| 4 | OFDM delay-track SFO (ppm; channel compensation only) | 12 | applied OCXO delta (ppm) |
| 5 | raw `TO_UE` (samples) | 13 | applied OCXO command (ppm) |
| 6 | continuous `TO_clock` (samples) | 14 | coarse target outstanding (ppm) |
| 7 | active Phase error: legacy `setpoint-TO_clock` or `pps_phase` `P_U-P_B` (samples) | 15 | stage (`0..4`: Coarse, Fast, Slow, Phase, Freeze) |

The packet is a fixed 152-byte little-endian frame: magic, version, channel count,
sequence, Unix-nanosecond source timestamp, the original 16 `float32` values, and
a typed PPS diagnostic trailer. The trailer and CSV append `r_bs_samples` and
`r_ue_samples` (`float64`), `bs_last_pps_ticks`, `bs_t0_to_pps_ticks`, and
`ue_last_pps_ns` (exact `int64`), plus `bs_tick_rate` and `sample_rate` (`float64`).
These fields distinguish BS-anchor, UE-PPS-latch, and TO discontinuities without
passing absolute timestamps through `float32`. The viewer still accepts legacy
`OCXODBG2` packets. Launch the
rolling viewer with `python3 scripts/plot_ocxo_debug.py`. Use `--channels 6,7,8,13,15`
to choose initial curves and `--record capture/ocxo.csv` to start CSV recording;
automatic names use `ocxo_debug_YYYYMMDD_01.csv` and increment to `_02`, `_03`,
and so on for later captures on the same day. Recording always stores all channels
even when only a subset is visible. Periodic
state packets are supplemented by immediate control, stage-transition, TO-jump,
and Alignment event packets so one-frame delta channels are not lost.

### `logging`

Same hierarchical filter as BS (`logging.*`). **Debug** examples: `ertm: debug`, `recovery: info`, `demod.snr: info`. **Performance** (paths end with `_profiling`, off until set): `demod_profiling: info`, `demod_eq_profiling: info`.

Resource-map notes:
* `resource_preview.data_resource_blocks` should normally match between BS and UE, including `kind`.
* Built-in ZC sync symbols, the optional CFO training field, comb pilots, and mid-frame BPSK pilots take precedence over configured resource blocks.
* `resource_preview.mask_blocks` controls compact sensing export only. In `output_mode=compact_mask`, runtime `STRD` is ignored because the mask defines the sampling pattern.
* Compact sensing payloads begin with `CompactSensingFrameHeader`, followed by fixed-order raw `complex<float>` samples.
