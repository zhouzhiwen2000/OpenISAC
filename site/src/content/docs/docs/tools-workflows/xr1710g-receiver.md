---
title: XR1710G Receive-only UE
description: Receive OpenISAC downlink signals with XR1710G, configure the source and UE, and optionally enable hardware carrier correction.
---

An XR1710G can replace the USRP on the UE side to receive OFDM signals and deliver decoded payloads. The BS continues transmitting through its USRP.

**XR1710G support is experimental and is not recommended for formal experiments.** This page walks through a single-channel setup at 40 MS/s and 2.442 GHz, from configuring the source to receiving data with UE.

## Requirements

- Prepare the device using the [XR1710G installation guide](https://github.com/zhouzhiwen2000/xr1710g-sdr/blob/main/platform/xr1710g/README.md). The receiving computer must be able to control the device and receive its IQ stream.
- Connect XR1710G to a Linux receiving computer over 10 GbE. `w1700k-stream` and OpenISAC `UE` must run on **the same computer**, where they exchange samples through shared memory.
- Complete the OpenISAC [software installation](/docs/getting-started/installation/) on that computer. The examples assume the repositories are at `~/xr1710g-sdr` and `~/OpenISAC`.
- This integration currently uses the CPU `UE` executable and supports one receive channel. UE uplink transmission, ARQ feedback, and dual-path interleaved input are not supported.

Supported source sample rates are **40, 80, 160, and 320 MS/s**. XR1710G, the BS, and `UE.yaml` must use the same rate; OpenISAC does not resample automatically. Use 40 MS/s for the setup shown here. Existing 50 or 100 MS/s configurations need to be adjusted, and the BS hardware must also support the selected rate.

## Data path

`USRP BS → radio signal → XR1710G → 10 GbE → w1700k-stream → OpenISAC UE → UDP receiving application`

The three programs have separate roles:

| Program | Runs on | Purpose |
| :--- | :--- | :--- |
| `w1700k` | Linux receiving computer | Configures XR1710G frequency, sample rate, filter, and gain; starts or stops the device stream |
| `w1700k-stream` | Linux receiving computer | Receives IQ from the device for local UE and spectrum applications |
| `UE` | The same Linux computer | Synchronizes, demodulates, decodes, and sends payloads to the configured UDP destination |

Keep separate terminals for `w1700k-stream`, device control and configuration, and `UE`.

## 1. Start the host receiver

**Run on the Linux receiving computer, terminal 1.**

If `w1700k-stream` is already running with shared-memory name `/xr1710g-live`, reuse it. Otherwise, run:

```bash
cd ~/xr1710g-sdr
make -C host/stream
host/stream/build/w1700k-stream --name /xr1710g-live --port 7700
```

Leave this terminal running. `--name` identifies the shared memory that UE will read; `--port` must match the device's stream destination configuration.

## 2. Configure XR1710G and start its stream

**Run on the Linux receiving computer, terminal 2.**

Replace `/path/to/unit-config.json` below with the device configuration created during XR1710G installation. `w1700k` uses it to connect to and control the device. It is separate from the OpenISAC `UE.yaml` configured later.

If a spectrum controller is managing the radio, stop its RF control before running these commands:

```bash
cd ~/xr1710g-sdr
host/bin/w1700k --config /path/to/unit-config.json stop
host/bin/w1700k --config /path/to/unit-config.json band 0 --rate 40M --lo 2442M
host/bin/w1700k --config /path/to/unit-config.json filter 40
host/bin/w1700k --config /path/to/unit-config.json gain --agc
host/bin/w1700k --config /path/to/unit-config.json start
host/bin/w1700k --config /path/to/unit-config.json status
```

This selects band 0, 40 MS/s, a receive center frequency near 2.442 GHz, a 40 MHz analog filter, and automatic gain. Record the actual local-oscillator frequency, `radio.lo_hz`, returned by `status`. Use this readback in the UE configuration; hardware tuning steps can make it slightly different from the requested frequency.

Confirm that terminal 1 receives samples continuously before continuing. Keep the source sample rate fixed during reception. Configure hardware gain and filtering through the XR1710G tools rather than through USRP gain settings.

## 3. Build UE and copy the preset

**Run on the Linux receiving computer, terminal 2.**

```bash
cd ~/OpenISAC
cmake -S . -B build-xr1710g -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=NOTFOUND
cmake --build build-xr1710g --target UE Xr1710gRxProbe -j4
cp config/UE_XR1710G.yaml build-xr1710g/UE.yaml
```

Edit `build-xr1710g/UE.yaml`. Keep the complete preset and check these fields:

| Field | Example | How to set it |
| :--- | :--- | :--- |
| `radio.radio_backend` | `xr1710g` | Selects XR1710G reception |
| `radio.xr1710g_shm` | `/xr1710g-live` | Must match `--name` in step 1 |
| `radio.xr1710g_center_freq` | `2442000000` | Actual `radio.lo_hz` read back in step 2, in Hz; this field does not tune the hardware |
| `radio.xr1710g_pll_command` | `""` | Leave empty for the initial setup to use software carrier correction |
| `rf_sampling.sample_rate` | `40000000` | Must match XR1710G and BS, in samples/s |
| `rf_sampling.bandwidth` | `40000000` | Describes the example receive bandwidth; configure the hardware filter in step 2 |
| `downlink.center_freq` | `2442000000` | BS transmit center frequency, in Hz |
| `network_output.udp_output_ip` | `127.0.0.1` | IP address of the payload receiving application |
| `network_output.udp_output_port` | `50001` | UDP port of the payload receiving application |

`xr1710g_center_freq` describes where the receiver is actually tuned; `downlink.center_freq` describes where the BS transmits. Set XR1710G close to the BS carrier first, then let UE track and correct the remaining frequency offset.

The preset disables uplink, ARQ feedback, hardware synchronization, and predictive delay. Keep these settings: this integration provides neither hardware time aligned with the BS nor automatic sample-clock correction. Carrier-frequency tracking remains enabled.

## 4. Configure and start BS

**Run on the BS computer.**

In `BS.yaml` in the BS runtime directory, set the sample rate to `40000000` and the transmit center frequency to `2442000000`. Keep the UHD backend for the BS.

Compare the two YAML files and match:

- Sample rate, FFT size, CP length, and symbols per frame.
- Synchronization-symbol position, ZC root, duplicate synchronization symbol, and CFO training-field settings.
- Pilot positions, mid-frame pilots, and data-resource allocation.

`UE_XR1710G.yaml` is a starting preset; it does not automatically match an existing BS configuration. Once the parameters agree, start BS from the directory containing the edited `BS.yaml` and keep it transmitting. This example assumes `~/OpenISAC/build` is the runtime directory:

```bash
cd ~/OpenISAC/build
sudo ../scripts/isolate_cpus.py run ./BS
```

## 5. Check samples and start UE

**Run on the Linux receiving computer, terminal 3.**

Run the receive check first. Its arguments are the shared-memory name, sample rate, actual receive LO frequency, and duration in seconds. Replace `2442000000` with the readback from step 2.

```bash
cd ~/OpenISAC/build-xr1710g
./Xr1710gRxProbe /xr1710g-live 40000000 2442000000 5
```

Normally, `samples` and `nonzero_probes` are greater than zero, while `errors`, `timeouts`, and `overflows` are zero. This checks continuous sample delivery to UE. To check communication, receive the payload sent by BS in the next step.

```bash
./UE
```

Leave UE running, wait for synchronization, and send payload data from BS. By default, UE delivers decoded data to local UDP port `127.0.0.1:50001`. For video, follow [Video and UDP Workflows](/docs/tools-workflows/video-udp-workflows/) and start with a low video bitrate.

## 6. Stop reception or change parameters

Press **Ctrl+C** in the UE terminal and wait for it to exit. Stopping UE leaves `w1700k-stream` and the XR1710G stream running.

To stop the device stream, run in the control terminal:

```bash
cd ~/xr1710g-sdr
host/bin/w1700k --config /path/to/unit-config.json stop
```

Then press Ctrl+C in terminal 1 to stop `w1700k-stream`. To change the sample rate or initial receive frequency, stop UE first, configure the device, update both YAML files, and restart in the order above.

## Optional: hardware carrier correction

By default, software carrier correction processes IQ on the receiving computer. To adjust the XR1710G LO through its PLL instead, first establish reception, stop UE, and set the following fields in the existing `radio` section:

```yaml
radio:
  radio_backend: xr1710g
  xr1710g_pll_command: >-
    python3 /home/user/OpenISAC/scripts/xr1710g_pll_control.py
    --sdr-repo /home/user/xr1710g-sdr
    --config /home/user/xr1710g-config.json
```

Replace all three paths with actual absolute paths on the receiving computer. Use the same device configuration as in step 2 for `--config`. Keep `radio.xr1710g_center_freq` set to the actual LO read back before startup. Save the YAML and restart UE; the control program starts with it.

Hardware correction is limited to **±100 kHz** around the initial LO and cannot cross a PLL divider boundary. It corrects the carrier frequency, not the sample clock. While UE controls the LO, other applications may read the same IQ stream but must not change the frequency, sample rate, or filter.

On normal exit, the control program restores the initial PLL settings. If the log reports a control, lock, or restoration failure, stop UE, inspect the device with `w1700k status`, and reconfigure it before restarting. Clear `xr1710g_pll_command` and restart UE to return to software correction.

## Troubleshooting and limitations

| Symptom or requirement | Check or action |
| :--- | :--- |
| UE cannot find the source or times out | Check that `w1700k-stream` remains running, its shared-memory name matches, and it runs on the same computer as UE |
| The receive check reports no samples | Check that the XR1710G stream is started and the 10 GbE link and data port are configured correctly |
| Samples arrive, but UE cannot synchronize or decode | Match actual receive frequency, BS carrier, sample rate, frame structure, pilots, and resource allocation; then check received signal strength |
| Repeated `overflow` messages | Check the host receiver's `lost` counter, 10 GbE link, and processing load; if reducing the rate, update the source and both configurations together |
| Reception stops after a rate change | A source/YAML rate mismatch pauses sample delivery; align the settings and restart UE |
| Synchronization is briefly lost after a source restart | UE reacquires synchronization; if it does not recover, confirm the source is delivering data again, then restart UE |
| Uplink or ARQ feedback is required | This integration is receive-only; use a USRP UE configuration with transmit support |
| Absolute ranging or eRTM calibration is required | XR1710G timestamps count local samples without hardware PPS alignment to BS and are not suitable for these uses |

See the [UE YAML Reference](/docs/reference/ue-yaml/) for the complete parameter descriptions.
