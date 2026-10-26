---
title: Scripts and Tools
description: User-facing helper scripts, launch commands, and prerequisites grouped by task.
---

This page lists tools that users launch directly. Unless stated otherwise, run commands from the repository root after installing the Python dependencies in `requirements.txt`.

## Sensing visualization

See the [Sensing Viewer Guide](/docs/tools-workflows/sensing-viewers/) for the purpose of each viewer, screenshots, controls, and experiment workflows.

After the BS and UE backends are running and the corresponding sensing outputs are enabled, launch a viewer:

```bash
# BS multichannel monostatic sensing
python3 scripts/plot_sensing_fast.py

# UE bistatic sensing
python3 scripts/plot_bi_sensing_fast.py
```

If a window opens without data, check the sensing-output switch, IP/port settings, and backend logs before restarting the viewer.

## OCXO debug

```bash
python3 scripts/plot_ocxo_debug.py --host 127.0.0.1 --port 12347
python3 scripts/plot_ocxo_debug.py --channels 6,7,8,13,15 --record capture/ocxo.csv
```

The viewer groups channels by unit, keeps a configurable rolling history, and
can pause rendering without pausing reception or CSV recording. Channel selection
only affects the plots; CSV output always contains the full 16-channel packet.

## Uplink timing control

`uplink_timing_control.py` reads and adjusts the BS `DUTI` and UE `TADV` values through their ZMQ control ports:

```bash
python3 scripts/uplink_timing_control.py \
  --bs-host 127.0.0.1 --bs-port 9999 \
  --ue-host 127.0.0.1 --ue-port 10001
```

For remote hosts, use the actual BS and UE addresses and keep the ports aligned with `network_output.control_port`.

## CPU isolation and backend launch

`isolate_cpus.py` reads critical CPU assignments from `BS.yaml` and `UE.yaml`. Apply isolation before launching a backend through `run`:

```bash
cd build
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS
```

The default command asks whether the host runs BS, UE, or both. When any kernel-isolated CPU is available, a second menu can isolate NIC IRQs; RX/TX splitting is shown only with at least two eligible isolated physical cores. `run` lets YAML-pinned critical threads use their assigned CPUs while excluding any dedicated NIC IRQ cores from the process.

## Web Config Console

```bash
python3 scripts/config_web_editor.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/`. The console edits the current `build/BS.yaml` and `build/UE.yaml` and provides backend launch and CPU-isolation controls. See [Web Config Console](/docs/tools-workflows/web-config-console/) for the full workflow.

`config_web_editor_schema.yaml`, `config_web_editor.js`, and the related HTML/CSS files are internal parts of the console and are not launched separately.

## Documentation site

```bash
# Local preview
cd site
npm run dev -- --host 0.0.0.0

# Build and publish into repository-root docs/
npm run build
```

`npm run build` creates `site/dist/` and then runs `scripts/publish_docs_site.py` to update `docs/`. Do not edit generated HTML under `docs/` directly.

## Physical sensing coordinates

Both the fast viewers and backend sensing viewers display Doppler frequency in **Hz**. Monostatic spectra display **range in metres**, while bistatic spectra display **relative delay in nanoseconds (ns)**. Micro-Doppler time is displayed in seconds. Detector configuration and bin-selection controls still use bin indices.

The backend and viewer must be updated together: sensing `PARM` **V10 is required**, and older or unknown layouts are rejected. The packet supplies the original OFDM FFT size, CP length, active sensing symbol stride and uniform subcarrier stride; the viewer uses the actual processing FFT lengths, including zero padding. Cropping or display downsampling does not change the physical calibration. A nonuniform compact subcarrier grid has no single FFT-bin delay scale and is marked unavailable instead of being assigned an incorrect distance.

With sample rate $F_s$, OFDM FFT size $N$, CP length $N_{CP}$, symbol stride $S_t$, subcarrier stride $S_f$, and processing FFT sizes $N_R,N_D$:

$$
f_{slow}=\frac{F_s}{(N+N_{CP})S_t},\quad \Delta f_D=\frac{f_{slow}}{N_D},\quad \Delta\tau=\frac{N}{F_s S_f N_R}.
$$

Monostatic range uses $R=c\tau/2$; bistatic delay uses $10^9\tau$ ns and is **relative to the current synchronization/alignment reference**. It is not an absolute transmitter-to-target or receiver-to-target distance. Monostatic zero range likewise depends on the system-delay alignment calibration. Changing the processing FFT length changes the displayed bin spacing, not the waveform's physical resolution.


## Calibration tools

Run `calibrate_hsys.py capture` to request and verify a saved response, or `calibrate_hsys.py inspect` for an offline file check. `calibrate_system_delay.py` summarizes recent per-channel alignment estimates from backend logs. See [Calibration Workflows](/docs/tools-workflows/calibration-workflows/) for setup, commands and validation limits.

### Optional radial velocity in the monostatic viewer

In `plot_sensing_fast.py`, open **Display & View**, enter the actual radiated carrier in **Velocity RF Carrier (GHz)**, and click **Set** (or press Enter). The range-spectrum horizontal axis, micro-Doppler vertical axis, and detected-target list then show radial velocity in **m/s**. Clear the field and apply it to return to Doppler frequency in **Hz**. The applied value is saved for the next launch.

The conversion is $v_r = f_D c / (2 f_c)$, with the sign inherited from the displayed Doppler frequency. For an external millimetre-wave converter, enter its RF output carrier rather than the USRP intermediate frequency; for example, enter `77` only when the actual RF carrier is 77 GHz. This display setting does not tune the radio or change FFT processing, detector thresholds, or bin coordinates. Positive/negative motion direction must be checked against the converter and experiment's frequency-sign convention. The bistatic viewer keeps its Doppler-frequency display because radial speed cannot be inferred with the monostatic formula.

A separate yellow annotation at the top right of the main spectrum shows the strongest detected return's range and velocity to one decimal place (or Doppler frequency when the carrier is blank). The annotation keeps a fixed position and size; missing target values are shown as **N/A**.

The monostatic fast viewer keeps its rolling micro-Doppler history across dropped or decimated frames and inserts zeros at missing slow-time positions. This preserves elapsed time; it does not reconstruct missing echoes, and gaps can weaken peaks or introduce sidelobes. History restarts when the range cell or sampling geometry changes, or the timestamp moves backwards.

### Strongest target in the bistatic viewer

`plot_bi_sensing_fast.py` shows the strongest detected target's relative delay (**ns**) and Doppler frequency (**Hz**) in yellow at the top right of the main spectrum, both to two decimal places. Missing targets are shown as **N/A**; the fixed annotation does not shift the plot layout. Delay is relative to the current receive-alignment reference and matches the main plot's vertical axis.

The bistatic defaults are **Delay FFT = 20480** and **Doppler FFT = 1000**. Existing saved settings take precedence. Target values are converted using the current FFT and sampling parameters.
