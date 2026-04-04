<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/logo.svg">
    <img src="docs/images/logo.svg" alt="OpenISAC Logo" width="400">
  </picture>
</p>

# OpenISAC

[中文版本](README_zh.md)

OpenISAC is a real-time OFDM platform for integrated sensing and communication (ISAC), built for academic experiments and rapid PHY iteration.

It is designed for the common gap between simulation code and full standard stacks: simple enough to modify quickly, but complete enough to run over the air with USRP hardware.

If your goal is "idea -> OTA experiment" with a minimal and readable codebase, this repository is a good fit. If you need Wi-Fi/5G NR interoperability or a production-grade protocol stack, it is not.

## Highlights

- Real-time OFDM communication plus monostatic and bistatic sensing
- Over-the-air synchronization support for bistatic experiments
- C++ backend for the radio path and Python frontend tools for visualization
- YAML-based runtime configuration with sample presets for X310/B210, adaptable to other UHD-supported USRPs
- Included utilities for CPU isolation, plotting, web-based config editing, and process control

## At a Glance

| Component | Main entry points | Purpose |
| :--- | :--- | :--- |
| BS backend | `OFDMModulator`, `config/Modulator_*.yaml` | Transmit OFDM frames, ingest UDP payloads, and output monostatic sensing data |
| UE backend | `OFDMDemodulator`, `config/Demodulator_*.yaml` | Receive and decode frames, output payload data, and run bistatic sensing |
| Frontend tools | `scripts/plot_*.py`, `scripts/config_web_editor.py` | Visualize sensing/channel results and edit runtime configs |

## Quick Navigation

- Setup and installation: [Hardware Setup](#hardware-setup), [Software Installation](#software-installation)
- First end-to-end run: [Typical Usage Example](#typical-usage-example)
- Runtime configuration: [OFDM Modulator](#ofdm-modulator), [OFDM Demodulator](#ofdm-demodulator)
- Web control UI: [Web Config Console](#7-web-config-console)

## Repository Layout

| Path | Description |
| :--- | :--- |
| `src/`, `include/` | Core C++ PHY, sensing, threading, and runtime logic |
| `config/` | Sample YAML presets for different roles, including X310/B210 examples that can be adapted to other USRPs |
| `scripts/` | Python frontends, web config console, and Linux performance helpers |
| `capture/` | Offline plotting helpers for saved sensing results |
| `docs/` | Static project site and architecture/signal-processing pages |

## Common Workflows

| Goal | Backend program | Typical config | Typical frontend |
| :--- | :--- | :--- | :--- |
| Run the BS side | `OFDMModulator` | `config/Modulator_X310.yaml` or `config/Modulator_B210.yaml` | `plot_sensing.py` / `plot_sensing_fast.py` |
| Run the UE side | `OFDMDemodulator` | `config/Demodulator_X310.yaml` or `config/Demodulator_B210.yaml` | `plot_bi_sensing.py` / `plot_bi_sensing_fast.py` |
| Tune parameters from a browser | `scripts/config_web_editor.py` | Reads `build/Modulator.yaml` and `build/Demodulator.yaml` | Browser at `http://<host>:8765` |

## Before the First OTA Run

- Prepare two backend nodes: BS uses one TX plus one RX antenna path; UE uses one RX path.
- Start from the sample YAML that is closest to your hardware. The repository includes X310/B210 examples, but it is not limited to those USRP models.
- Copy runtime YAMLs into `build/`, because both binaries read `Modulator.yaml` or `Demodulator.yaml` from their working directory.
- If the frontend runs on another machine, set `default_ip` in the runtime YAML to that machine's IP.
- If you care about stable real-time behavior, run `scripts/set_performance.bash`, then apply CPU isolation with `sudo ../scripts/isolate_cpus.bash` (or a custom CPU set), and only then launch via `sudo ../scripts/isolate_cpus.bash run ...`.

## What it is - and what it is not

### What OpenISAC is

- A minimal OFDM-based PHY for joint communication and sensing research
- Designed for prototyping, academic experiments, and rapid algorithm validation
- Focused on readability and modification speed rather than full-stack completeness

### What OpenISAC is not

- A standard-compliant implementation; it does not aim to match Wi-Fi or 5G NR
- A replacement for full-stack systems such as openwifi or OpenAirInterface
- A production-ready communications stack

### When to use it

- Prototyping new OFDM/ISAC algorithms
- Rapidly validating PHY, synchronization, or sensing ideas over the air
- Research setups where interoperability is not required

### When not to use it

- Building a Wi-Fi/NR-compatible system
- Requiring full MAC/stack behavior, interoperability, or certification-oriented behavior

## Citation

If you find this repository useful, please cite our paper:

> Z. Zhou, C. Zhang, X. Xu, and Y. Zeng, "OpenISAC: An Open-Source Real-Time Experimentation Platform for OFDM-ISAC with Over-the-Air Synchronization," *arXiv preprint* arXiv:2601.03535, Jan. 2026.
>
> [[arXiv](https://arxiv.org/pdf/2601.03535)]

## Authors

- Zhiwen Zhou (zhiwen_zhou@seu.edu.cn)
- Chaoyue Zhang (chaoyue_zhang@seu.edu.cn)
- Xiaoli Xu (Member, IEEE) (xiaolixu@seu.edu.cn)
- Yong Zeng (Fellow, IEEE) (yong_zeng@seu.edu.cn)

## Affiliation

<img src="docs/images/SEUlogo.png" height="80" alt="SEU Logo" style="border:none; box-shadow:none;"> &nbsp;&nbsp; <img src="docs/images/PML.png" height="80" alt="PML Logo" style="border:none; box-shadow:none;">

**Yong Zeng Group at the National Mobile Communications Research Laboratory, Southeast University and the Purple Mountain Laboratories**

## Community

- [Join our QQ Group](https://qm.qq.com/q/NIQRNGb0kY)
- [Bilibili Channel (Yong Zeng Group)](https://space.bilibili.com/627920129)
- WeChat Official Account:

  <img src="docs/images/WeChat.jpg" width="150" alt="WeChat QR Code">

## Hardware Setup
 
### Backend (C++)
 
To set up the complete system, you will need the following hardware:
 
*   **USRP Devices**: 2 units (e.g., USRP X310, B210, etc.)
*   **Computers**: 2 units (High performance recommended for signal processing)
*   **Antennas**: 3
*   **OCXO/GPSDO**: 2 units (Required for both USRPs)
 
#### Connection Setup
 
The system consists of two main nodes:
 
1.  **BS Node**
    *   **Hardware**: 1x Computer, 1x USRP.
    *   **Antennas**: Connect 2 antennas to this USRP (1 for TX, 1 for RX).
    *   **Clock**: Connect an OCXO or GPSDO to the REFIN port of the USRP.
    *   **Function**: Transmits the OFDM signal and receives the radar echo.
 
2.  **UE Node**
    *   **Hardware**: 1x Computer, 1x USRP.
    *   **Antennas**: Connect 1 antenna to the RX port of this USRP.
    *   **Clock**: Connect an OCXO or GPSDO to the REFIN port of the USRP.
    *   **High-precision DAC (Optional)**: Use a high-precision DAC to enable finetuning the OCXO.
    *   **Function**: Receives the OFDM signal for communication and bistatic sensing.
 
#### Interface Requirements
To support high bandwidth and sample rates, ensure the connection between the Computers and USRPs uses:
*   **>= 10 Gigabit Ethernet (10GbE)** (For X-series)
*   **USB 3.0** (For B-series)
 
### Frontend (Python)
*   **Computer**: 1x Computer (Windows or Linux).
    *   Can be one of the backend computers or a separate machine.
*   **CPU**: High performance CPU (i7 10700 or better) if no GPU is available.
*   **GPU**: An Nvidia GPU is recommended for acceleration.


## Software Installation
 
### Backend (C++)
 
#### Operating System
*   **Ubuntu 24.04 LTS**
    *   Download: [http://www.ubuntu.com/download/desktop](http://www.ubuntu.com/download/desktop)
*   **macOS (Apple Silicon, local development / demo only)**
    *   Guide: [Separate macOS build guide](https://github.com/zhouzhiwen2000/UHD_OFDM/blob/OpenISAC_MultiChannel/docs/macos_build.md)
 
#### Dependencies & Installation
 
#### 1. UHD (USRP Hardware Driver)
Install the UHD toolchain by following the official Ettus guide (Please follow the tutorial for Ubuntu 24.04):
*   [Building and Installing the USRP Open-Source Toolchain on Linux](https://kb.ettus.com/Building_and_Installing_the_USRP_Open-Source_Toolchain_(UHD_and_GNU_Radio)_on_Linux#Update_and_Install_dependencies)
 
> **Note:** This code has been tested on UHD v4.9.0.1. You can checkout this version using `git checkout v4.9.0.1`.
 
#### 2. Install Aff3ct
This project uses the Aff3ct library for Forward Error Correction (FEC). Install it from source:
 
```bash
sudo apt-get install nlohmann-json3-dev
git clone https://github.com/aff3ct/aff3ct.git
cd aff3ct
git submodule update --init --recursive
mkdir build
cd build
cmake .. -G"Unix Makefiles" -DCMAKE_CXX_COMPILER="g++" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_CXX_FLAGS="-funroll-loops -march=native" -DAFF3CT_COMPILE_EXE="OFF" -DAFF3CT_COMPILE_SHARED_LIB="ON" -DSPU_STACKTRACE="OFF" -DSPU_STACKTRACE_SEGFAULT="OFF" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -faligned-new"
make -j$(nproc)
sudo make install
```
 
#### 3. Clone Repository
Clone the OpenISAC repository:
 
```bash
cd ~
git clone https://github.com/zhouzhiwen2000/OpenISAC.git
```
 
#### 4. Build OpenISAC
Build the project using CMake:
 
```bash
sudo apt-get install libyaml-cpp-dev
cd OpenISAC
mkdir build
cd build
cmake ..
make -j$(nproc)
```
 
#### 5. System Performance Tuning
Run the provided script to optimize your system settings for real-time processing:
 
```bash
cd ~/OpenISAC
chmod +x scripts/set_performance.bash
./scripts/set_performance.bash
```
 
> **Note:** `secure_boot` needs to be turned off in your BIOS settings to enable `RT_RUNTIME_SHARE` functionality.
 
When UHD spawns a new thread, it may try to boost the thread's scheduling priority. If setting the new priority fails, the UHD software prints a warning to the console, as shown below.
 
```text
[WARNING] [UHD] Failed to set desired affinity for thread
```
 
To address this issue, non-privileged (non-root) users need to be given special permission to change the scheduling priority. This can be enabled by creating a group `usrp`, adding your user to it, and then appending the line `@usrp - rtprio 99` to the file `/etc/security/limits.conf`.
 
```bash
sudo groupadd usrp
sudo usermod -aG usrp $USER
```
 
Then add the line below to end of the file `/etc/security/limits.conf`:
 
```text
@usrp - rtprio  99
```
 
You must log out and log back into the account for the settings to take effect. In most Linux distributions, a list of groups and group members can be found in the `/etc/group` file.
 
#### 6. CPU Isolation and Execution
To ensure stable real-time performance, use `scripts/isolate_cpus.bash` to constrain system services (`system.slice`, `user.slice`, `init.scope`) to selected CPUs and reserve other CPUs for your workload.

All commands require root privileges (`sudo`):

```bash
cd ~/OpenISAC
chmod +x scripts/isolate_cpus.bash
sudo ./scripts/isolate_cpus.bash --help
```

**Default isolation policy**

```bash
sudo ./scripts/isolate_cpus.bash
```

- Default reserved cores for application: first 8 cores (`0-7`).
- System services are restricted to the remaining cores.
- If total CPU cores are `<= 8`, isolation cannot be applied effectively and both app/system use all cores.

**Custom application CPU set**

```bash
sudo ./scripts/isolate_cpus.bash 4          # App uses 0-3
sudo ./scripts/isolate_cpus.bash 8-15       # App uses 8-15
sudo ./scripts/isolate_cpus.bash 0,2,4,6    # App uses explicit core list
```

The selected app CPU set is saved to `/tmp/isolate_cpus_app.conf`.

**CPU binding priority when cores are limited**

- Reserve one dedicated core for the main thread first.
- Then prioritize the TX/RX real-time threads.
- Finally allocate cores to modulation/demodulation and sensing/signal-processing threads, because these compute-heavy stages typically have larger buffers and can absorb moderate scheduling jitter.

In the web CPU-binding editor, this usually means prioritizing `main thread affinity` first, then `_tx_proc` / `rx_proc` and per-channel RX loops, and only after that the modulation/demodulation and sensing-processing workers.

**Run application on reserved cores**

```bash
cd build
sudo ../scripts/isolate_cpus.bash run ./OFDMModulator
```

- `run` reads saved app CPUs from `/tmp/isolate_cpus_app.conf`.
- If no saved config exists, `run` falls back to the default app CPU set.

> **Note:** Always use `sudo ../scripts/isolate_cpus.bash run ...` to launch applications after isolation is set. Direct execution or manual `taskset` may fail due to slice affinity constraints.

**Reset configuration (optional)**

```bash
sudo ./scripts/isolate_cpus.bash reset
```

This removes the isolation settings and restores system slices to all CPUs.

#### 7. Configuration
The system uses YAML files for runtime configuration.

*   **Config filenames**:
    `OFDMModulator` reads `Modulator.yaml`, and `OFDMDemodulator` reads `Demodulator.yaml` from the working directory.
*   **First run**:
    Template YAML files live in `config/`. Copy `config/Modulator_X310.yaml` / `config/Modulator_B210.yaml` or
    `config/Demodulator_X310.yaml` / `config/Demodulator_B210.yaml` to `Modulator.yaml` / `Demodulator.yaml`, then edit them in place.
 
### Frontend (Python)

It is recommended to use a `conda` or `venv` environment with **Python 3.13**.

**Miniconda Installation Guide:**
*   **Windows:** [Miniconda Installation for Windows](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-powershell)
*   **Linux:** [Miniconda Installation for Linux](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2)

**Create a new conda environment:**
```bash
conda create -n OpenISAC python=3.13
conda activate OpenISAC
```
 
#### Install Dependencies
 
```bash
pip install -r requirements.txt
```

**Note:** `ffmpeg` is required for video streaming demonstrations.
*   **Ubuntu:** `sudo apt install ffmpeg`
*   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or place executable in the working directory.
 
#### Enable GPU Acceleration (Optional)

Some fast plotting scripts can use an optional GPU array backend when it is available in the active Python environment. If no optional backend is installed, the scripts automatically fall back to CPU execution.

#### Enable Intel Integrated GPU Acceleration (Optional)

If your device has an Intel integrated GPU (e.g., Intel UHD Graphics, Intel Iris Xe, etc.), you can enable GPU acceleration via `dpctl` and `dpnp`. This is especially useful for laptops and desktops without a dedicated Nvidia GPU.

##### 1. Install Intel GPU Official Driver

First, ensure your system has the latest Intel GPU driver installed:
*   **Windows:** Download and install the latest driver from Intel Download Center: [https://www.intel.com/content/www/us/en/download-center/home.html](https://www.intel.com/content/www/us/en/download-center/home.html)
*   **Ubuntu:** Install Intel compute-runtime:
    ```bash
    sudo apt install intel-opencl-icd libze-intel-gpu1 libze1 intel-media-va-driver-non-free
    ```

##### 2. Install Python Dependencies

```bash
pip install dpctl dpnp
```

##### 3. Verify Installation

Run the following command to check if the Intel GPU is correctly recognized:

```bash
python -c "import dpctl; print(dpctl.get_devices())"
```

If the installation is successful, you should see output similar to:
```
[<dpctl.SyclDevice [backend_type.level_zero, device_type.gpu, Intel(R) UHD Graphics] at 0x...>]
```

##### 4. Usage Notes

The OpenISAC frontend automatically detects available array backends. The priority order is:
1. CuPy backend
2. dpnp backend
3. CPU (fallback)

No code modifications are required; the system will automatically select the best available backend.

## Typical Usage Example

### 1. Startup of the BS
```bash
sudo -s
cd build
# For X310:
cp ../config/Modulator_X310.yaml Modulator.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./OFDMModulator

# For B210:
cp ../config/Modulator_B210.yaml Modulator.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./OFDMModulator
```
*If you are using a separate frontend computer, set `default_ip` in `Modulator.yaml` to that frontend IP.*

### 2. Startup of the UE
```bash
sudo -s
cd build
# For X310:
cp ../config/Demodulator_X310.yaml Demodulator.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./OFDMDemodulator

# For B210:
cp ../config/Demodulator_B210.yaml Demodulator.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./OFDMDemodulator
```
*If you are using a separate frontend computer, set `default_ip` in `Demodulator.yaml` to that frontend IP.*

### 3. Stream video to the BS
```bash
ffmpeg -re -stream_loop -1 -fflags +genpts -i video.mp4 -an -c:v libx264 -x264-params keyint=5:min-keyint=1 -b:v 3000k -minrate 3000k -maxrate 3000k -bufsize 1M -f rtp -sdp_file video.sdp "rtp://<your IP of the BS>:50000"
```
*If you are streaming locally, your IP of the BS can be set to 127.0.0.1.*

### 4. Play video from the UE
Copy `video.sdp` to the video receiver, modify `m=video 50000 RTP/AVP 96` to `m=video 50001 RTP/AVP 96`.
```bash
ffplay -protocol_whitelist file,rtp,udp -i video1.sdp
```
*Note that this command should be run on the frontend.*

### 5. Run monostatic frontend
```bash
python3 ./scripts/plot_sensing.py
```
For better real-time performance (if an NVIDIA GPU is available):
```bash
python3 ./scripts/plot_sensing_fast.py
```

### 6. Run bistatic frontend
```bash
python3 ./scripts/plot_bi_sensing.py
```
For better real-time performance (if an NVIDIA GPU is available):
```bash
python3 ./scripts/plot_bi_sensing_fast.py
```

### 7. Web Config Console
For remote-friendly configuration editing and process control, run:
```bash
python3 scripts/config_web_editor.py --host 0.0.0.0 --port 8765
```

Then open `http://<your-host>:8765` in a browser.

What it does:
* Provides separate Modulator / Demodulator tabs, plus a dedicated `Resource Planner` tab for `data_resource_blocks`.
* Edits `build/Modulator.yaml` and `build/Demodulator.yaml` as parameter/value forms instead of a raw YAML text area.
* Provides a dedicated CPU-binding editor for `cpu_cores`, with thread names, generated comments, and per-thread CPU selection.
* Saves the current form back to YAML and starts/stops modulator and demodulator processes from the `build/` directory.
* Includes launch options such as enabling/disabling CPU isolation and overriding the isolate CPU list.
* Includes command presets and a custom command field for each tab.
* Lets you draw payload rectangles on a large time-frequency planner canvas, snap the block boundaries to integer RE grid points, and apply the result independently to the transmitter or receiver YAML.
* Includes a `Guard Band Grid` preset that follows `scripts/plot_const.py`, i.e. it keeps only subcarriers `1..489` and `535..N-1` before the normal sync/pilot stripping rules are applied.

Notes:
* Default commands are `./OFDMModulator` and `./OFDMDemodulator`.
* The editor currently targets the runtime YAML files in `build/`, because the binaries read `Modulator.yaml` / `Demodulator.yaml` from their working directory.
* The `Resource Planner` tab edits `data_resource_blocks` only. Use `Apply to Transmitter` to write the current planner state into `build/Modulator.yaml`, and `Apply to Receiver` to write it into `build/Demodulator.yaml`.
* The planner intentionally allows TX and RX to differ while you are experimenting, but normal operation still expects matching `data_resource_blocks` on both sides.
* If CPU cores are limited, reserve a dedicated core for `main thread affinity` first, then prioritize TX/RX threads, and finally modulation/demodulation plus sensing/signal-processing threads; these compute-heavy stages typically have larger buffers and tolerate transient jitter better.
* If `Enable runtime CPU isolation` is on, the console uses the current `cpu_cores` list to derive the default isolated CPU set and calls `scripts/isolate_cpus.bash` before launch.
* If `Override CPU isolation list` is enabled, the runtime isolation text box is seeded from the default isolate list and you may edit it manually for this launch.
* If `Enable runtime CPU isolation` is off, the console still launches the selected command through the privileged runtime path, but it does not call `scripts/isolate_cpus.bash`.
* The runtime panel also provides an optional sudo-password field and a `Reset CPU isolation` action.
* Because the console can launch arbitrary commands entered in the web UI, bind it only to trusted networks or keep the default `127.0.0.1`.

### OFDM Modulator

`OFDMModulator` is configured through `Modulator.yaml`.
Use `config/Modulator_X310.yaml` or `config/Modulator_B210.yaml` as a starting template.

`Modulator.yaml` parameter reference:

| Key | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `fft_size` | `int` | `1024` | OFDM FFT size. |
| `cp_length` | `int` | `128` | Cyclic prefix length (samples). |
| `sync_pos` | `int` | `1` | Sync symbol index in one frame. |
| `sample_rate` | `float` / Hz | `50000000` | Baseband sample rate. |
| `bandwidth` | `float` / Hz | `50000000` | Analog bandwidth, usually same as `sample_rate`. |
| `center_freq` | `float` / Hz | `2400000000` | RF center frequency. |
| `tx_gain` | `float` / dB | `30` | TX gain. |
| `tx_channel` | `int` | `0` | TX channel index. |
| `zc_root` | `int` | `29` | Zadoff-Chu root index. |
| `num_symbols` | `int` | `100` | Number of OFDM symbols per frame. |
| `sensing_output_mode` | `string` | `dense` | Sensing UDP output mode. `dense` keeps the legacy STRD-based full-buffer output. `compact_mask` switches sensing to per-frame compact RE extraction. |
| `pilot_positions` | `int[]` | `[571,...,451]` | Pilot subcarrier indices. |
| `data_resource_blocks` | `object[]` | omitted | Optional payload resource rectangles. Omit the key to keep legacy full-grid payload mapping; set `[]` to disable payload RE entirely. Each block uses `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`; only selected non-sync, non-pilot RE carry real payload, and all other non-sync, non-pilot RE transmit pre-generated QPSK. |
| `sensing_mask_blocks` | `object[]` | omitted | Optional compact sensing RE rectangles. Used only when `sensing_output_mode=compact_mask`. Blocks use absolute frame symbol indices and raw FFT-bin indices, allow sync/pilot RE, are unioned on overlap, and are emitted in symbol-major then subcarrier-major order. If every selected symbol uses the same subcarrier set and the selected symbols are equally spaced on the frame ring, compact sensing can also enable runtime MTI and local Delay-Doppler processing. |
| `device_args` | `string` | `""` | Shared USRP args fallback for TX/RX. |
| `tx_device_args` | `string` | `""` | TX-specific USRP args. |
| `rx_device_args` | `string` | `""` | Default sensing RX USRP args. |
| `clock_source` | `string` | `internal/external/gpsdo` | Global clock source. |
| `time_source` | `string` | `""` | Global time/PPS source; empty means follow `clock_source`. |
| `tx_clock_source` | `string` | `""` | TX clock source override. |
| `tx_time_source` | `string` | `""` | TX time source override. |
| `rx_clock_source` | `string` | `""` | Default sensing RX clock source override. |
| `rx_time_source` | `string` | `""` | Default sensing RX time source override. |
| `wire_format_tx` | `string` | `sc16` | TX wire format, typically `sc16` or `sc8`. |
| `wire_format_rx` | `string` | `sc16` | RX wire format, typically `sc16` or `sc8`. |
| `udp_input_ip` | `string` / IPv4 | `0.0.0.0` | Bind IP for incoming payload UDP packets. |
| `udp_input_port` | `int` | `50000` | Bind port for incoming payload UDP packets. |
| `mono_sensing_ip` | `string` / IPv4 | `127.0.0.1` | Destination IP for monostatic sensing output. |
| `mono_sensing_port` | `int` | `8888` | Destination port for monostatic sensing output. |
| `sensing_rx_channel_count` | `int` | `1` | Number of sensing RX channels (`0` disables sensing RX). |
| `sensing_rx_channels` | `object[]` | `[]` | Per-channel sensing RX settings (see table below). |
| `default_ip` | `string` / IPv4 | `127.0.0.1` | Default destination IP for outputs that are not explicitly set. |
| `control_port` | `int` | `9999` | UDP port for control commands (heartbeat/MTI/etc.). |
| `measurement_enable` | `bool` | `false` | Enable CPU internal measurement mode. When enabled, `OFDMModulator` generates deterministic PRBS payloads instead of listening on `udp_input_*`, and `OFDMDemodulator` switches decoded measurement payloads into BER/BLER/EVM accounting. |
| `measurement_mode` | `string` | `internal_prbs` | Measurement mode selector. Only `internal_prbs` is supported. Unsupported values disable measurement mode during config normalization. |
| `measurement_run_id` | `string` | `""` | Run identifier written into measurement CSV summaries. |
| `measurement_output_dir` | `string` | `""` | Output directory used by the CPU measurement summaries. |
| `measurement_payload_bytes` | `int` | `1024` | Bytes per internally generated measurement payload. Values below the internal header size are clamped up. |
| `measurement_prbs_seed` | `int` | `0x5A` | Base seed used to derive deterministic PRBS payload contents. |
| `measurement_packets_per_point` | `int` | `1` | Number of measurement payloads sent for each online `MRST` epoch. Values below `1` are clamped to `1`. |
| `profiling_modules` | `string` | `""` | Profiling module list, comma-separated. Common values include `modulation`, `latency`, `data_ingest`, `sensing_proc`, and `sensing_process`; `all` enables every module. Modulator end-to-end latency profiling is enabled only when both `modulation` and `latency` are included. |
| `cpu_cores` | `int[]` | `[0,1,2,3,4,5]` | Allowed CPU core list. Size this list for the TX thread, modulation thread, data-ingest thread, each enabled sensing channel's RX/sensing threads, and the main thread. If cores are limited, keep one dedicated core for the main thread first, then prioritize the TX and sensing RX threads, and only after that the modulation/data-ingest/sensing-processing threads because the latter stages have deeper buffers. |

If `data_resource_blocks` is enabled, copy the same rectangles into `Demodulator.yaml`; `sync_pos` and `pilot_positions` still take precedence over overlapping payload rectangles.

When `sensing_output_mode=compact_mask`, sensing switches to one UDP frame per OFDM frame and only the selected RE channel estimates are sent in compact mode. `STRD` remains ignored because the stride is defined implicitly by `sensing_mask_blocks`. If every selected symbol uses the same subcarrier set and the selected symbols are equally spaced on the frame ring, runtime `MTI` and `SKIP` are enabled: `SKIP=1` keeps the compact raw-RE payload, while `SKIP=0` switches back to the existing dense Delay-Doppler output. For that regular compact mode, config normalization expands `range_fft_size` to at least the selected subcarrier count and `doppler_fft_size` to at least the selected sensing symbol count. The compact sensing UDP payload begins with `CompactSensingFrameHeader { magic/version, mask_hash, re_count, frame_start_symbol_index }`, followed by `re_count` raw `complex<float>` values in the configured fixed order. Existing `plot_sensing*.py` viewers do not parse this compact payload yet.

`sensing_rx_channels` object fields:

| Key | Type | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `usrp_channel` | `int` | `0` | USRP RX channel index. |
| `device_args` | `string` | `""` | Per-channel USRP args. |
| `clock_source` | `string` | `""` | Per-channel clock source override. |
| `time_source` | `string` | `""` | Per-channel time source override. |
| `wire_format_rx` | `string` | `""` | Per-channel RX wire format override. |
| `rx_gain` | `float` | `30` | RX gain for this channel. |
| `alignment` | `int` | `63` | Per-channel alignment offset (samples). |
| `rx_antenna` | `string` | `""` | RX antenna name, e.g. `TX/RX`, `RX1`. |
| `enable_system_delay_estimation` | `bool` | `false` | If `true`, this channel performs a ZC-based system delay estimation at startup and then once every 434 frames, while keeping the sensing pipeline disabled and continuing to drain frames. |
| `sensing_ip` | `string` | `127.0.0.1` | Per-channel sensing destination IP. |
| `sensing_port` | `int` | `8888` | Per-channel sensing destination port. |

Notes:
* If `sensing_rx_channels` is empty and `sensing_rx_channel_count > 0`, default channels `0..N-1` are generated automatically.
* If the count and list size differ, the list is resized to match `sensing_rx_channel_count`.
* When `enable_system_delay_estimation=true` for a channel, that channel performs one system delay estimation near startup and then repeats it once every 434 frames while continuing to drain frames. Normal sensing processing and sensing UDP output remain disabled.
* In practice, keep hardware-specific fields such as `device_args`, `wire_format_*`, per-channel RX antenna selection, and output IPs aligned with the actual radio/deployment you are using; the sample YAMLs are starting points, not universal presets.

### OFDM Demodulator

`OFDMDemodulator` is configured through `Demodulator.yaml`.
Use `config/Demodulator_X310.yaml` or `config/Demodulator_B210.yaml` as a starting template.

`Demodulator.yaml` parameter reference:

| Key | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `fft_size` | `int` | `1024` | OFDM FFT size. |
| `cp_length` | `int` | `128` | Cyclic prefix length (samples). |
| `sync_pos` | `int` | `1` | Sync symbol index in one frame. |
| `sample_rate` | `float` / Hz | `50000000` | Baseband sample rate. |
| `bandwidth` | `float` / Hz | `50000000` | Analog bandwidth, usually same as `sample_rate`. |
| `center_freq` | `float` / Hz | `2400000000` | RF center frequency. |
| `rx_gain` | `float` / dB | `50` | RX gain. |
| `rx_agc_enable` | `bool` | `false` | Enable hardware RX AGC. Tracking AGC uses the filtered `delay_spectrum` main peak to adjust USRP RX gain, applies stale-frame timestamp gating like alignment/frequency updates, and forces gain reduction when the sync symbol approaches ADC full scale. |
| `rx_agc_low_threshold_db` | `float` / dB | `11.0` | Lower bound of the tracking AGC window. Gain is increased only when the filtered `delay_spectrum` main peak falls below this threshold. |
| `rx_agc_high_threshold_db` | `float` / dB | `13.0` | Upper bound of the tracking AGC window. Gain is decreased only when the filtered `delay_spectrum` main peak rises above this threshold. |
| `rx_agc_max_step_db` | `float` / dB | `3.0` | Maximum RX gain step applied by one AGC update. Saturation-triggered protection also uses this step size when forcing gain down. |
| `rx_agc_update_frames` | `int` | `4` | Minimum processed-frame interval between tracking-stage AGC updates. Values below `1` are clamped to `1`. |
| `rx_channel` | `int` | `0` | RX channel index. |
| `zc_root` | `int` | `29` | Zadoff-Chu root index. |
| `num_symbols` | `int` | `100` | Number of OFDM symbols per frame. |
| `sensing_symbol_num` | `int` | `100` | Number of symbols used for sensing processing. |
| `sensing_output_mode` | `string` | `dense` | Bistatic sensing UDP output mode. `dense` keeps the legacy STRD-based full-buffer output. `compact_mask` switches sensing to per-frame compact RE extraction. |
| `frame_queue_size` | `int` | `8` | Capacity of the demodulator RX frame queue. Values below `1` are clamped to `1`. |
| `sync_queue_size` | `int` | `8` | Capacity of the demodulator sync-search batch queue. Values below `1` are clamped to `1`. |
| `reset_hold_s` | `float` / s | `0.5` | How long invalid delay conditions must persist before the demodulator forces a hard reset back to sync search. Internally this is converted to a frame count from `samples_per_frame / sample_rate`. Values below `0` are clamped to `0.5`. |
| `range_fft_size` | `int` | `1024` | Range FFT size. |
| `doppler_fft_size` | `int` | `100` | Doppler FFT size. |
| `pilot_positions` | `int[]` | `[571,...,451]` | Pilot subcarrier indices. |
| `data_resource_blocks` | `object[]` | omitted | Optional payload resource rectangles. Omit the key to keep legacy full-grid payload extraction; set `[]` to extract no payload LLR. Each block uses `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`, and the effective payload RE are the selected rectangles after removing `sync_pos` and `pilot_positions`. This setting must match `Modulator.yaml` when enabled. |
| `sensing_mask_blocks` | `object[]` | omitted | Optional compact sensing RE rectangles. Used only when `sensing_output_mode=compact_mask`. Blocks use absolute frame symbol indices and raw FFT-bin indices, allow sync/pilot RE, are unioned on overlap, and are emitted in symbol-major then subcarrier-major order. If every selected symbol uses the same subcarrier set and the selected symbols are equally spaced on the frame ring, compact sensing can also enable runtime MTI and local Delay-Doppler processing. |
| `device_args` | `string` | `""` | USRP device args. |
| `clock_source` | `string` | `internal/external/gpsdo` | Clock source. |
| `wire_format_rx` | `string` | `sc16` | RX wire format, typically `sc16` or `sc8`. |
| `software_sync` | `bool` | `true` | Enable software synchronization tracking. |
| `hardware_sync` | `bool` | `false` | Enable hardware synchronization. |
| `hardware_sync_tty` | `string` | `/dev/ttyUSB0` | TTY device used by hardware sync controller. |
| `ocxo_pi_switch_abs_error_ppm` | `float` | `0.0002` | Switch to slow-stage OCXO PI when absolute `error_ppm` stays below this threshold. |
| `akf_enable` | `bool` | `true` | Enable adaptive Kalman filtering (AKF) on hardware-sync `error_ppm`. |
| `akf_bootstrap_frames` | `int` | `64` | Cold-start frame count before AKF starts normal KF updates. |
| `akf_innovation_window` | `int` | `64` | Innovation history window used for ACF/LS adaptation. |
| `akf_max_lag` | `int` | `4` | Maximum innovation autocorrelation lag used in LS fitting. |
| `akf_adapt_interval` | `int` | `64` | Frame interval for adaptive `Q/R` least-squares updates. |
| `akf_gate_sigma` | `float` | `3.0` | Innovation gating threshold (sigma). |
| `akf_tikhonov_lambda` | `float` | `1e-3` | Tikhonov regularization weight for LS adaptation. |
| `akf_update_smooth` | `float` | `0.2` | Exponential smoothing factor for updated `Q/R`. |
| `akf_q_wf_min` | `float` | `1e-10` | Lower bound of white-frequency-noise coefficient. |
| `akf_q_wf_max` | `float` | `1e2` | Upper bound of white-frequency-noise coefficient. |
| `akf_q_rw_min` | `float` | `1e-12` | Lower bound of random-walk-frequency-noise coefficient. |
| `akf_q_rw_max` | `float` | `1e1` | Upper bound of random-walk-frequency-noise coefficient. |
| `akf_r_min` | `float` | `1e-8` | Lower bound of observation-noise variance `R`. |
| `akf_r_max` | `float` | `1e3` | Upper bound of observation-noise variance `R`. |
| `ppm_adjust_factor` | `float` | `0.05` | Frequency offset compensation factor. |
| `desired_peak_pos` | `int` | `20` | Target delay-peak position used by alignment logic. |
| `enable_bi_sensing` | `bool` | `true` | Enable the bistatic sensing pipeline and UDP output. When set to `false`, the demodulator skips bistatic sensing channel startup. |
| `bi_sensing_ip` | `string` / IPv4 | `127.0.0.1` | Destination IP for bistatic sensing output. |
| `bi_sensing_port` | `int` | `8889` | Destination port for bistatic sensing output. |
| `channel_ip` | `string` / IPv4 | `127.0.0.1` | Destination IP for channel-estimation output. |
| `channel_port` | `int` | `12348` | Destination port for channel-estimation output. |
| `pdf_ip` | `string` / IPv4 | `127.0.0.1` | Destination IP for PDP/PDF output. |
| `pdf_port` | `int` | `12349` | Destination port for PDP/PDF output. |
| `constellation_ip` | `string` / IPv4 | `127.0.0.1` | Destination IP for constellation output. |
| `constellation_port` | `int` | `12346` | Destination port for constellation output. |
| `vofa_debug_ip` | `string` / IPv4 | `127.0.0.1` | Destination IP for VOFA+ debug output. |
| `vofa_debug_port` | `int` | `12347` | Destination port for VOFA+ debug output. |
| `udp_output_ip` | `string` / IPv4 | `127.0.0.1` | Destination IP for decoded payload output. |
| `udp_output_port` | `int` | `50001` | Destination port for decoded payload output. |
| `default_ip` | `string` / IPv4 | `127.0.0.1` | Default destination IP used when output IP fields are empty. |
| `control_port` | `int` | `9999` | UDP port for control commands. |
| `measurement_enable` | `bool` | `false` | Enable CPU internal measurement mode. In this mode, decoded measurement payloads are consumed locally for BER/BLER/EVM statistics instead of being forwarded to `udp_output_*`. |
| `measurement_mode` | `string` | `internal_prbs` | Measurement mode selector. Only `internal_prbs` is supported. Unsupported values disable measurement mode during config normalization. |
| `measurement_run_id` | `string` | `""` | Run identifier written into measurement CSV summaries. |
| `measurement_output_dir` | `string` | `""` | Output directory used by the CPU measurement summaries. |
| `measurement_payload_bytes` | `int` | `1024` | Expected bytes per measurement payload. Values below the internal header size are clamped up. |
| `measurement_prbs_seed` | `int` | `0x5A` | Base seed used to rebuild deterministic PRBS measurement payloads. |
| `measurement_packets_per_point` | `int` | `1` | Expected measurement payload count for each online `MRST` epoch. Values below `1` are clamped to `1`. |
| `profiling_modules` | `string` | `""` | Profiling module list, comma-separated. Common values include `demodulation`, `agc`, `align`, and `snr`; `all` enables every module. `agc` gates AGC logs, `align` gates runtime `ALGN:` logs, and `snr` prints periodic `_snr_db / _noise_var / _llr_scale` updates from the active demodulator path. |
| `cpu_cores` | `int[]` | `[0,1,2,3,4,5]` | Allowed CPU core list. If cores are limited, keep one dedicated core for the main thread first, then prioritize `rx_proc`, and only after that `process_proc`, `sensing_process_proc`, and `bit_processing_proc`, because those later stages have larger buffers and can better tolerate short scheduling jitter. |

When `sensing_output_mode=compact_mask`, bistatic sensing also switches to one UDP frame per OFDM frame and only the selected RE channel estimates are sent in compact mode. `STRD` remains ignored because the stride is defined implicitly by `sensing_mask_blocks`. If every selected symbol uses the same subcarrier set and the selected symbols are equally spaced on the frame ring, runtime `MTI` and `SKIP` are enabled: `SKIP=1` keeps the compact raw-RE payload, while `SKIP=0` switches back to the existing dense Delay-Doppler output. For that regular compact mode, config normalization expands `range_fft_size` to at least the selected subcarrier count and `doppler_fft_size` to at least the selected sensing symbol count. The compact payload uses the same `CompactSensingFrameHeader` and fixed RE ordering as the modulator side.

Notes:
* RX AGC has two phases. During `SYNC_SEARCH`, the receiver resets gain to the configured `rx_gain` and performs a coarse search sweep (+1 dB every 10 frames, wrapping from max gain back to min gain). After lock, tracking AGC uses the filtered `delay_spectrum` peak window defined by `rx_agc_low_threshold_db` and `rx_agc_high_threshold_db`.
* Sync-symbol time-domain samples are also checked for near/full-scale ADC usage. If too many I/Q components approach full scale, the demodulator forces gain reduction and temporarily blocks gain increases to avoid ping-pong behavior.
* A hard reset clears timing/frequency tracking state, flushes pending queues, resets the tracking AGC state, and returns the receiver to `SYNC_SEARCH`. `reset_hold_s` controls how long bad delay conditions must persist before this happens.
