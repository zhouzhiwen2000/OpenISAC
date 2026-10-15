---
title: Hardware Setup
description: Hardware roles and basic preparation for OpenISAC experiments.
---

## Backend (C++)

To set up the complete system, you will need the following hardware:

*   **USRP Devices**: 2 units (e.g., USRP X310, B210, etc.)
*   **Computers**: 2 units (High performance recommended for signal processing)
*   **Antennas**: 3 for downlink-only operation; 5 if duplex/uplink is enabled. The BS uplink RX path currently cannot share the same RF channel as the sensing RX path.
*   **OCXO/GPSDO**: 2 units (Required for both USRPs)

### Connection Setup

The system consists of two main nodes:

1.  **BS Node**
    *   **Hardware**: 1x Computer, 1x USRP.
    *   **Antennas**: Connect 2 antennas for downlink-only operation (1 for downlink TX, 1 for sensing RX). If duplex/uplink is enabled, add a separate BS uplink RX antenna/RF chain; it cannot currently share the sensing RX channel.
    *   **Clock**: Connect an OCXO or GPSDO to the REFIN port of the USRP.
    *   **Function**: Transmits the OFDM signal and receives the radar echo.

2.  **UE Node**
    *   **Hardware**: 1x Computer, 1x USRP.
    *   **Antennas**: Connect 1 antenna to the RX port for downlink-only operation. If duplex/uplink is enabled, also connect the UE TX antenna/RF chain; in FDD mode, ensure the configured uplink carrier is supported.
    *   **Clock**: Connect an OCXO or GPSDO to the REFIN port of the USRP.
    *   **High-precision DAC (Optional)**: Use a high-precision DAC to enable finetuning the OCXO.
    *   **Function**: Receives the OFDM signal for communication and bistatic sensing; when duplex/uplink is enabled, also transmits UE->BS uplink payloads.

### Interface Requirements
To support high bandwidth and sample rates, ensure the connection between the Computers and USRPs uses:
*   **>= 10 Gigabit Ethernet (10GbE)** (For X-series)
*   **USB 3.0** (For B-series)

## Frontend (Python)
*   **Computer**: 1x Computer (Windows or Linux).
    *   Can be one of the backend computers or a separate machine.
*   **CPU**: High performance CPU (i7 10700 or better) if no GPU is available.
*   **GPU**: An Nvidia GPU is recommended for acceleration.

## XR1710G as a receive-only UE

The CPU `UE` can use an XR1710G running
[`xr1710g-sdr`](https://github.com/zhouzhiwen2000/xr1710g-sdr) through
`radio.radio_backend: xr1710g`. It reads the existing `w1700k-stream` POSIX
shared-memory ring on the **same Linux computer**. The external receiver handles
10 GbE ingest, reordering and unpacking; the OpenISAC backend converts the signed
14-bit IQ to FC32 and defaults to software frequency correction. Native builds on
AVX2/FMA CPUs fuse conversion, normalization and NCO mixing in one SIMD loop;
other CPUs use the portable implementation. The standard UE
synchronization, channel estimation, LDPC decoding and UDP output are reused.

This backend accepts one channel at **40, 80, 160 or 320 MS/s**. It does not
resample: the source and BS sample rates must match `rf_sampling.sample_rate`.
The experimental dual-path interleaved capture format is not supported.
Initial RF LO, analog filter, gain and stream production are configured with the
XR1710G tools. UE start/stop affects only its own reader, so a spectrum viewer
can continue reading the same single-channel stream.

Carrier correction defaults to the software NCO. The optional PLL helper below
implements the same backend tuning interface with hardware LO fine adjustment;
the UE's existing correction ordering and stale-observation gates remain in use.
The hardware path only converts IQ to FC32, without software mixing.

Automatic crystal steering is not supported; use software NCO or PLL carrier
correction with the existing UE synchronization tracking.

### Hardware carrier correction

On the **Linux receiver host**, set this command in the UE YAML, replacing all
paths with absolute paths to the local checkouts and existing router config:

```yaml
radio:
  radio_backend: xr1710g
  xr1710g_pll_command: >-
    python3 /home/user/OpenISAC/scripts/xr1710g_pll_control.py
    --sdr-repo /home/user/xr1710g-sdr
    --config /home/user/xr1710g-config.json
```

`xr1710g_center_freq` remains the initial actual source LO. The backend starts
one persistent helper and applies synchronous, total offsets within ±100 kHz.
A positive physical correction raises the LO. UHD RX tuning uses effective
center = RF − DSP, so a manual DSP request has the opposite sign. The backend
returns the negative of the quantized physical offset through
`TuneResult.actual_dsp_freq`, keeping `actual_rf_freq` at the initial LO to
preserve the UE's existing accumulation and reset behavior. This logical split
also applies when the physical correction is implemented in the synthesizer.
The divider step is about 2.98 Hz on band 0 and 11.92 or 14.31 Hz on bands 1/2,
depending on the LO. Corrections cannot cross the synthesizer ratio boundary.

The helper checks source identity, register readback and PLL lock. Keep RF
configuration under this backend's sole control while it runs; a spectrum viewer
may continue reading IQ, but must not retune or reconfigure the source. Detected
external changes or control errors fail the tuning call. Normal helper shutdown
restores the saved divider; restoration errors are reported on stderr.

Tuning preserves the reader cursor and buffered samples: it does not flush IQ,
inject a stream discontinuity or reset timestamps. Phase-continuous hardware
retuning is not guaranteed. PLL correction does not adjust the sample clock.
An empty command retains software correction.

### Start reception

1. On the XR1710G receiving computer, prepare a working source stream using the
   XR1710G repository's installation instructions. Reuse the running host receiver
   if it already has the required shared-memory name. Otherwise, build and run
   it in a separate terminal:

   ```bash
   cd ~/xr1710g-sdr
   make -C host/stream
   host/stream/build/w1700k-stream --name /xr1710g-live --port 7700
   ```

2. Configure the radio through its existing controller. Use one controller owner;
   stop a spectrum controller that owns RF configuration before using the CLI.
   For a 40 MS/s, 2.442 GHz source:

   ```bash
   cd ~/xr1710g-sdr
   host/bin/w1700k --config /path/to/unit-config.json stop
   host/bin/w1700k --config /path/to/unit-config.json band 0 --rate 40M --lo 2442M
   host/bin/w1700k --config /path/to/unit-config.json filter 40
   host/bin/w1700k --config /path/to/unit-config.json gain --agc
   host/bin/w1700k --config /path/to/unit-config.json start
   host/bin/w1700k --config /path/to/unit-config.json status
   ```

   Record the returned **actual** `radio.lo_hz`. While UE runs, keep the source
   rate fixed and let the backend own any configured PLL correction.
   The shared-memory ABI contains no LO or antenna-layout identifier;
   the configured source LO is an explicit declaration. A nonzero source-rate
   field is checked at attachment and during reception; if it is zero, the
   backend relies on the declared sample rate.

3. On the same computer, build the CPU receiver and copy the preset into its
   runtime directory:

   ```bash
   cd ~/OpenISAC
   cmake -S . -B build-xr1710g
   cmake --build build-xr1710g --target UE Xr1710gRxProbe -j4
   cp config/UE_XR1710G.yaml build-xr1710g/UE.yaml
   ```

   Set `radio.xr1710g_center_freq` to the actual source LO and
   `downlink.center_freq` to the BS carrier. Match sample rate, FFT/CP, preamble,
   pilots and resource allocation to the BS. The preset uses 40 MS/s; an existing
   50 or 100 MS/s BS profile must be changed to a supported rate. A USRP BS can
   continue using its existing UHD backend.

4. Check the transport, then start the UE from its runtime directory. Substitute
   the actual source LO in the probe command:

   ```bash
   cd ~/OpenISAC/build-xr1710g
   ./Xr1710gRxProbe /xr1710g-live 40000000 2442000000 5
   ./UE
   ```

   The probe should report nonzero samples and zero errors/timeouts. UE should
   then report synchronization and deliver decoded payloads to the configured
   UDP destination (preset: `127.0.0.1:50001`). The probe proves sample transport;
   decoded payloads from a matching BS establish the communication path.

### Timing and recovery

Timestamps count samples on a **local receiver timeline**, without hardware
PPS or BS time synchronization. Firmware events consume no sample time. Lost
or overwritten data generates overflow metadata; source run changes or service
replacement generate broken-chain metadata. Reads spanning a gap are discarded,
and resumed data uses the UE's existing receive-only reacquisition path.
Unobserved events during an overrun make the duration of that gap uncertain;
these timestamps are not suitable for absolute ranging or eRTM calibration.

The preset disables uplink/ARQ feedback, hardware clock control and predictive
delay. CFO tracking remains enabled. The XR1710G RF/sample-clock
relationship has not been calibrated for CFO-derived SFO prediction. Hardware
gain is configured externally rather than interpreted as a USRP gain in dB.
A live sample-rate mismatch holds reception until the declared rate is restored;
change both source and YAML, then restart UE for a new rate.
