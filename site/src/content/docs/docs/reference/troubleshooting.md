---
title: Troubleshooting
description: Symptom-oriented checks for OpenISAC builds, configuration, USRPs, communication, sensing, and frontends.
---

Preserve the first concrete error instead of copying only the final log line. Record the launch command, working directory, active `BS.yaml` / `UE.yaml`, and USRP model; these details usually separate configuration, environment, and link problems quickly.

## Four checks to start with

1. Confirm that the process working directory contains the intended `BS.yaml` or `UE.yaml`.
2. Match `radio.radio_backend` to the scenario: `uhd` for a real USRP or `sim` for simulation.
3. Compare the BS and UE sample rate, FFT/CP, frame length, pilots, frequencies, duplex mode, and resource mapping.
4. Temporarily set the relevant `logging.modules` entry to `info` or `debug`, reproduce once, and inspect the first Warn/Error.

## Build fails

Reproduce with the standard build first:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

- During CMake configuration, inspect the first missing package: UHD, Boost, FFTW3, yaml-cpp, ZeroMQ, OpenMP, or Aff3ct.
- During compilation, keep the first compiler error; later messages are often cascading failures.
- After changing compilers, CUDA, or dependency paths, configure a new empty build directory instead of reusing stale CMake state.

## Runtime cannot find YAML

`BS` and `UE` read fixed filenames from the current working directory; they do not automatically load presets from `config/`.

```bash
cp config/BS_X310.yaml build/BS.yaml
cp config/UE_X310.yaml build/UE.yaml
cd build
./BS
```

The same working-directory rule applies through `sudo` and launch helpers. Restart the backend after editing YAML.

## USRP is not found or cannot initialize

1. Run `uhd_find_devices` and confirm that UHD discovers the device.
2. Check `usrp_device.device_args` and any TX/RX-specific device arguments.
3. For X310, check host NIC addressing, routing, and MTU. For B210, check USB 3, cable quality, and power.
4. If an external clock or PPS is configured, verify that the source is physically present; otherwise validate basic connectivity with `internal` first.

## UHD overflow, underflow, or late packets

- X310: inspect link speed, MTU, packet loss, and competing traffic on the NIC.
- B210: use USB 3 directly and avoid hubs or a shared high-throughput controller.
- For either device, check sustained CPU load and YAML CPU assignments. Lower `rf_sampling.sample_rate` temporarily to test whether the failure is throughput-related.

## UE cannot decode the downlink

Follow the signal chain instead of starting with LDPC tuning:

1. Confirm that the UE finds a synchronization peak and remains locked.
2. Check whether CFO/SFO converges or repeatedly triggers reacquisition.
3. Compare `ofdm_frame.*`, pilot locations, mid-frame pilots, and resource blocks on BS and UE.
4. Inspect channel estimates and constellation quality before LLR and LDPC diagnostics.
5. On RF hardware, check center frequency, gains, and antenna ports for weak or saturated input.

## Uplink does not work

Set `uplink.enabled: true` on both BS and UE, and match `uplink.duplex_mode`, the TDD symbol window, or the FDD center frequency. With `ChannelSimulator`, also enable `simulation.enable_uplink`. If samples arrive but decoding fails, inspect timing windows, channel estimation, equalization, and LDPC in that order.

## Sensing output is unstable or incorrect

- Establish stable communication/synchronization first; sensing cannot compensate for persistent frame errors or loss of lock.
- Verify `alignment`, RF-chain delay, channel calibration, FFT dimensions, and frontend parsing settings.
- Check backend warnings for frame-pair queues, sensing-output queues, or ZMQ high-water-mark drops.
- For bistatic sensing, match `sensing.sensing_delay_correction_mode` to the active LoS-tracking or eRTM configuration.

## Visualization window does not update

1. Enable the corresponding `network_output.*_enabled` switch.
2. Match the viewer IP/port to YAML and allow the TCP port through host firewalls.
3. Run `ss -ltnp` on the backend host and confirm that the expected endpoint is listening.
4. Check that the backend is producing data and is not continuously dropping frames because of full queues or missing subscribers.

If the problem remains, retain the complete YAML files, launch commands, logs from startup through the first failure, USRP connection details, and host environment information.
