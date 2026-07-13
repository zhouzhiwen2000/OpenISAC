<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo_light.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/logo.svg">
    <img src="docs/images/logo.svg" alt="OpenISAC Logo" width="400">
  </picture>
</p>

# OpenISAC

[中文版本](README_zh.md) | [Changelog](CHANGELOG.md)

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
| BS backend | `BS`, `config/BS_*.yaml` | Transmit OFDM frames, ingest UDP payloads, and output monostatic sensing data |
| UE backend | `UE`, `config/UE_*.yaml` | Receive and decode frames, output payload data, and run bistatic sensing |
| Frontend tools | `scripts/plot_*.py`, `scripts/config_web_editor.py` | Visualize sensing/channel results and edit runtime configs |

## Quick Navigation

- Setup and installation: [Hardware Setup](#hardware-setup), [Software Installation](#software-installation)
- First end-to-end run: [Typical Usage Example](#typical-usage-example)
- Runtime configuration: [BS](#bs), [UE](#ue)
- Web control UI: [Web Config Console](#8-web-config-console)
- Recent updates: [Changelog](CHANGELOG.md)

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
| Run the BS side | `BS` | `config/BS_X310.yaml` or `config/BS_B210.yaml` | `plot_sensing_fast.py` |
| Run the UE side | `UE` | `config/UE_X310.yaml` or `config/UE_B210.yaml` | `plot_bi_sensing_fast.py` |
| Tune parameters from a browser | `scripts/config_web_editor.py` | Reads `build/BS.yaml` and `build/UE.yaml` | Browser at `http://<host>:8765` |

## Before the First OTA Run

- Prepare two backend nodes. In downlink-only mode, BS uses one TX plus one sensing RX antenna path and UE uses one RX path. If duplex/uplink is enabled, UE also needs a TX antenna/RF chain and BS needs an RX antenna/RF chain for the uplink; in FDD mode the radios must also support the configured uplink carrier and the required TX/RX isolation.
- Start from the sample YAML that is closest to your hardware. The repository includes X310/B210 examples, but it is not limited to those USRP models.
- Copy runtime YAMLs into `build/`, because both binaries read `BS.yaml` or `UE.yaml` from their working directory.
- If the frontend runs on another machine, point the Python viewer at the backend with `--host <backend-ip>` or the viewer's Backend IP field. `default_out_ip` is only for destination-style output IPs, not UDP/ZMQ listen addresses.
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
*   **Antennas**: 3 for downlink-only operation; 4 if duplex/uplink is enabled and the UE TX path uses a separate antenna port
*   **OCXO/GPSDO**: 2 units (Required for both USRPs)
 
#### Connection Setup
 
The system consists of two main nodes:
 
1.  **BS Node**
    *   **Hardware**: 1x Computer, 1x USRP.
    *   **Antennas**: Connect 2 antennas to this USRP (1 for downlink TX, 1 for sensing/uplink RX).
    *   **Clock**: Connect an OCXO or GPSDO to the REFIN port of the USRP.
    *   **Function**: Transmits the OFDM signal and receives the radar echo.
 
2.  **UE Node**
    *   **Hardware**: 1x Computer, 1x USRP.
    *   **Antennas**: Connect 1 antenna to the RX port for downlink-only operation. If duplex/uplink is enabled, also connect the UE TX antenna/RF chain; in FDD mode, ensure the configured uplink carrier and RF isolation are supported.
    *   **Clock**: Connect an OCXO or GPSDO to the REFIN port of the USRP.
    *   **High-precision DAC (Optional)**: Use a high-precision DAC to enable finetuning the OCXO.
    *   **Function**: Receives the OFDM signal for communication and bistatic sensing; when duplex/uplink is enabled, also transmits UE->BS uplink payloads.
 
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
    *   Guide: [Separate macOS build guide](https://github.com/zhouzhiwen2000/OpenISAC/blob/main/docs/macos_build.md)
 
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
sudo apt-get install libyaml-cpp-dev libzmq3-dev cppzmq-dev
cd OpenISAC
mkdir build
cd build
cmake ..
make -j$(nproc)
```

> Backend↔Frontend communication (sensing streams + the control/params channel) runs over **ZeroMQ**. The backend binds PUB sockets for the sensing/debug streams and a ROUTER socket for control on the configured listen IP/ports; the sample YAMLs use `0.0.0.0` for sensing/debug/control listen IPs. Python viewers connect with SUB/DEALER sockets. Point a viewer at a remote backend with its `--host <ip>` flag or Backend IP field (default `127.0.0.1`).
 
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
sudo ../scripts/isolate_cpus.bash run ./BS
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
    `BS` reads `BS.yaml`, and `UE` reads `UE.yaml` from the working directory.
*   **First run**:
    Template YAML files live in `config/`. Copy `config/BS_X310.yaml` / `config/BS_B210.yaml` or
    `config/UE_X310.yaml` / `config/UE_B210.yaml` to `BS.yaml` / `UE.yaml`, then edit them in place.
    For the B210 TDD duplex preset, copy `config/BS_B210_Duplex.yaml` and `config/UE_B210_Duplex.yaml`.
 
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
 
If an Nvidia GPU is available, install `cupy-cuda12x` to enable GPU acceleration:

> **Note:** Please ensure that the CUDA Toolkit is installed before installing CuPy.
 
```bash
pip install cupy-cuda12x
```

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

The OpenISAC frontend automatically detects available GPU backends. The priority order is:
1. Nvidia GPU (CUDA)
2. Intel iGPU (dpnp)
3. CPU (fallback)

No code modifications are required; the system will automatically select the best available backend.

## Typical Usage Example

### 1. Startup of the BS
```bash
sudo -s
cd build
# For X310:
cp ../config/BS_X310.yaml BS.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./BS

# For B210:
cp ../config/BS_B210.yaml BS.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./BS

# For B210 duplex:
cp ../config/BS_B210_Duplex.yaml BS.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./BS
```
*If you are using a separate frontend computer, point the monostatic viewer at the BS backend IP with `--host` or the viewer's Backend IP field.*

### 2. Startup of the UE
```bash
sudo -s
cd build
# For X310:
cp ../config/UE_X310.yaml UE.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./UE

# For B210:
cp ../config/UE_B210.yaml UE.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./UE

# For B210 duplex:
cp ../config/UE_B210_Duplex.yaml UE.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./UE
```
*If you are using a separate frontend computer, point the bistatic viewer at the UE backend IP with `--host` or the viewer's Backend IP field. Set `default_out_ip` only for decoded/debug outputs that should be sent to another machine.*

### 3. Stream video to the BS
```bash
ffmpeg -re -stream_loop -1 -fflags +genpts -i video.mp4 -an -c:v libx264 -x264-params keyint=5:min-keyint=1 -b:v 3000k -minrate 3000k -maxrate 3000k -bufsize 1M -f rtp -sdp_file video.sdp "rtp://<your IP of the BS>:50000"
```
*If you are streaming locally, your IP of the BS can be set to 127.0.0.1.*

If you want to stream audio together with the video, use RTP MPEG-TS instead:
```bash
ffmpeg -re -stream_loop -1 -fflags +genpts -i video.mp4 -c:v libx264 -x264-params keyint=5:min-keyint=1 -b:v 30000k -minrate 30000k -maxrate 30000k -bufsize 1M -c:a aac -b:a 128k -ar 48000 -ac 2 -f rtp_mpegts "rtp://<your IP of the BS>:50000"
```

### 4. Play video from the UE
Copy `video.sdp` to the video receiver, modify `m=video 50000 RTP/AVP 96` to `m=video 50001 RTP/AVP 96`.
```bash
ffplay -protocol_whitelist file,rtp,udp -i video1.sdp
```
*Note that this command should be run on the frontend.*

For the RTP MPEG-TS stream with audio, you can play it directly without an SDP file:
```bash
ffplay rtp://0.0.0.0:50001
```

### 5. Run monostatic frontend
```bash
python3 ./scripts/plot_sensing_fast.py
```
This maintained viewer auto-selects CUDA, MLX, Intel GPU, or CPU backends.

### 6. Run bistatic frontend
```bash
python3 ./scripts/plot_bi_sensing_fast.py
```
This maintained viewer auto-selects CUDA, MLX, Intel GPU, or CPU backends.

### 7. Calibrate sensing channel response

Before calibration, connect the sensing RF path directly: connect the transmit RF output to the corresponding sensing RX input and keep the connection stable during calibration.

If the transmit power is high, reduce the transmit power first. If needed, insert a suitable attenuator in the direct RF path to avoid RX saturation. The attenuator should be as flat as possible across the signal bandwidth; otherwise its in-band ripple will be included in the calibration result.

For monostatic sensing, start the BS backend and the monostatic sensing frontend, select the sensing channel you want to calibrate in the viewer, then click `Calibrate Hsys`. For multichannel monostatic setups, repeat this for each channel that needs its own RF path calibration.

For bistatic sensing, start the BS and UE backends, open the bistatic sensing frontend, then click `Calibrate Hsys` in the bistatic viewer while the RF path is directly connected.

Wait for the backend log to report that calibration has completed and the calibration file has been saved. The current run will immediately use the new response calibration. On later launches, the backend automatically loads the matching calibration file; if no matching file is found, sensing continues without calibration and prints a notice.

After calibration is complete, restore the normal antenna or experiment connection before continuing OTA measurements.

### 8. Web Config Console
For remote-friendly configuration editing and process control, run:
```bash
python3 scripts/config_web_editor.py --host 0.0.0.0 --port 8765
```

Then open `http://<your-host>:8765` in a browser.

What it does:
* Provides separate BS / UE tabs, plus a `Resource Planner` tab for `data_resource_blocks` and a `Sensing Resource Map` tab for `mask_blocks`.
* Edits `build/BS.yaml` and `build/UE.yaml` as parameter/value forms instead of a raw YAML text area.
* Provides module-local CPU-binding fields for downlink, uplink, sensing real-time loops, and the main thread.
* Saves the current form back to YAML and starts/stops BS and UE processes from the `build/` directory.
* Includes launch options such as enabling/disabling CPU isolation and overriding the isolate CPU list.
* Includes CPU/CUDA command presets and a custom command field for each tab.
* Lets you draw payload / sensing-pilot rectangles for `data_resource_blocks`, or compact sensing rectangles for `mask_blocks`, snap the block boundaries to integer RE grid points, and apply the result independently to the transmitter or receiver YAML.
* Includes a `Guard Band Grid` preset that follows `scripts/plot_const.py`, i.e. it keeps only subcarriers `1..489` and `535..N-1` before the normal sync/pilot stripping rules are applied.

Notes:
* Default commands are `./BS` and `./UE`; switch to the CUDA preset if needed.
* The editor currently targets the runtime YAML files in `build/`, because the binaries read `BS.yaml` / `UE.yaml` from their working directory.
* `Resource Planner` edits `data_resource_blocks`: it decides which RE carry payload and which RE are reserved as `sensing_pilot`.
* `Sensing Resource Map` edits `mask_blocks`: it decides which RE are exported on the compact sensing path when `output_mode=compact_mask`.
* Both planners can be applied to either side. During experiments TX and RX may differ temporarily, but normal operation still expects matching `data_resource_blocks` on both sides.
* If CPU cores are limited, reserve a dedicated core for `main thread affinity` first, then prioritize TX/RX threads, and finally modulation/demodulation plus sensing/signal-processing threads; these compute-heavy stages typically have larger buffers and tolerate transient jitter better.
* CPU affinity is configured only for real-time pipeline threads and the main thread. Non-real-time service/output/helper threads are intentionally left unbound.
* Use `-1`, `[]`, or an omitted optional field to leave that module unbound.
* If `Enable runtime CPU isolation` is on, the console derives the default isolated CPU set from all non-negative real-time CPU fields and calls `scripts/isolate_cpus.bash` before launch.
* If `Override CPU isolation list` is enabled, the runtime isolation text box is seeded from the default isolate list and you may edit it manually for this launch.
* If `Enable runtime CPU isolation` is off, the console still launches the selected command through the privileged runtime path, but it does not call `scripts/isolate_cpus.bash`.
* The runtime panel also provides an optional sudo-password field and a `Reset CPU isolation` action.
* Because the console can launch arbitrary commands entered in the web UI, bind it only to trusted networks or keep the default `127.0.0.1`.

## Parameter Reference

### BS

`BS` is configured through `BS.yaml`.
Use `config/BS_X310.yaml`, `config/BS_B210.yaml`, or `config/BS_B210_Duplex.yaml` as a starting template.

`BS.yaml` parameter reference:

| Key | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `fft_size` | `int` | `1024` | OFDM FFT size. |
| `cp_length` | `int` | `128` | Cyclic prefix length (samples). |
| `sync_pos` | `int` | `1` | Sync symbol index in one frame. |
| `enable_sec_sync_symbol` | `bool` | `false` | Reserve `sync_pos-1` as a duplicate ZC sync symbol. When enabled, the receiver uses the two consecutive sync symbols for Schmidl-Cox-style coarse timing/modulo-CFO estimation, resolves CFO aliases by local ZC correlation around `sync_pos`, then refines the main sync. Requires `sync_pos >= 1`. |
| `enable_cfo_training_sequence` | `bool` | `false` | Reserve `sync_pos+1` as a dedicated repeated CFO training field. The receiver still uses ZC or the optional second sync symbol for frame timing and modulo CFO first; when this field is enabled, it is used only to deambiguate the CFO alias. TX/RX must use the same setting. |
| `cfo_training_period_samples` | `int` / samples | `16` | Repetition period of the CFO training field. It must divide `fft_size`; the unambiguous CFO span is approximately `+-sample_rate/(2*period)`. |
| `sample_rate` | `float` / Hz | `50000000` | Baseband sample rate. |
| `bandwidth` | `float` / Hz | `50000000` | Analog bandwidth, usually same as `sample_rate`. |
| `center_freq` | `float` / Hz | `2400000000` | RF center frequency. |
| `tx_gain` | `float` / dB | `30` | TX gain. |
| `tx_channel` | `int` | `0` | TX channel index. |
| `zc_root` | `int` | `29` | Zadoff-Chu root index. |
| `num_symbols` | `int` | `100` | Number of OFDM symbols per frame. |
| `output_mode` | `string` | `dense` | Sensing output mode. `dense` keeps the legacy STRD-based full-buffer output. `compact_mask` switches sensing to per-frame compact RE extraction. |
| `cuda_mod_pipeline_slots` | `int` | `2` | Number of CUDA modulation pipeline slots. Values below `1` are clamped to `1`. |
| `pilot_positions` | `int[]` | `[571,631,...,451]` | Configurable comb-pilot subcarrier indices spread across the occupied band. |
| `midframe_pilot_symbols` | `int[]` | `[]` | Optional mid-frame BPSK pilot symbol indices inside each frame, such as `[25,50,75]`. These symbols are excluded from payload mapping; configured comb-pilot RE keep the comb-pilot sequence for phase tracking, while the remaining RE in those symbols use deterministic BPSK. |
| `midframe_pilot_seed` | `int` | `1296453708` | Deterministic BPSK pilot seed. It must match between `BS.yaml` and `UE.yaml`. |
| `data_resource_blocks` | `object[]` | omitted | Optional communication resource map. It answers: "which RE are allowed to carry payload?" Omit the key to keep the legacy behavior, where every non-reserved-sync, non-comb-pilot RE carries payload. Set `[]` to disable payload RE entirely. Each block is a rectangle with `symbol_start`, `symbol_count`, `subcarrier_start`, `subcarrier_count`, and optional `kind`. `kind: payload` means those RE carry real payload. `kind: sensing_pilot` means those RE transmit a deterministic sensing-pilot reference sequence instead, so they stay predictable for sensing and are excluded from payload mapping. This sensing-pilot sequence is generated from an alternate Zadoff-Chu root that is different from the frame sync root, which avoids confusing sensing-pilot symbols with the dedicated sync symbol. Any remaining non-reserved-sync, non-comb-pilot, non-mid-frame-BPSK-pilot RE outside `payload` blocks transmit pre-generated QPSK. |
| `mask_blocks` | `object[]` | omitted | Optional compact sensing resource map. It answers: "which RE should be exported on the sensing output path?" It is used only when `output_mode=compact_mask`; in `dense` mode it is ignored. Each block is a rectangle in absolute frame-symbol index and raw FFT-bin index. ZC sync symbols, comb-pilot, and mid-frame BPSK pilot RE are allowed here; the optional CFO training field is rejected because it is not a valid sensing symbol. Overlapping blocks are merged automatically, and the exported order is fixed as symbol-major then subcarrier-major. If every selected symbol uses the same subcarrier set and the selected symbols are evenly spaced on the frame ring, runtime MTI and local Delay-Doppler processing can also be enabled. |
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
| `rx_channel` | `int` | `0` | BS uplink RX channel index on the shared TX/RX USRP. |
| `rx_wire_format` | `string` | `sc16` | BS uplink RX wire format, typically `sc16` or `sc8`. |
| `rx_wire_format` | `string` | `sc16` | BS sensing RX default wire format, typically `sc16` or `sc8`. |
| `udp_input_ip` | `string` / IPv4 | `0.0.0.0` | BS downlink payload UDP bind IP. This is the input stream transmitted on the BS->UE downlink. |
| `udp_input_port` | `int` | `50000` | BS downlink payload UDP bind port. |
| `udp_output_ip` | `string` / IPv4 | `127.0.0.1` | BS decoded uplink payload UDP destination IP. This is the output stream recovered from UE->BS uplink. |
| `udp_output_port` | `int` | `50003` | BS decoded uplink payload UDP destination port. |
| `duplex_mode` | `string` | `tdd` | Duplexing scheme. `tdd` time-multiplexes UE uplink symbols into the BS frame on the downlink center frequency; `fdd` keeps BS downlink active while UE uplink uses `uplink.center_freq`. |
| `uplink` | `object` | `symbol_start=90`, `symbol_count=10`, `guard_symbols=1`, `center_freq=2500000000` | Uplink/duplex settings. In TDD, `symbol_start`, `symbol_count`, and `guard_symbols` define the DL/UL boundary in OFDM symbols, and `center_freq` is ignored. In FDD, `center_freq` defines the UE->BS carrier, while `symbol_start`, `symbol_count`, and `guard_symbols` are ignored and the uplink uses the full frame. Enabling uplink requires a UE TX antenna/RF chain and a BS RX antenna/RF chain; FDD additionally requires enough RF separation or isolation for simultaneous TX/RX. |
| `bs_dl_ul_timing_diff` | `int` / samples | `63` | BS-side DL/UL timing offset for the uplink RX window. It is normalized modulo one frame at startup and can be adjusted at runtime with `DUTI`. |
| `mono_sensing_ip` | `string` / IPv4 | `0.0.0.0` | ZMQ listen IP for the monostatic sensing stream and control channel. Use `0.0.0.0` to accept remote viewers, or `127.0.0.1` for local-only access. |
| `mono_sensing_port` | `int` | `8888` | ZeroMQ PUB bind port for the monostatic sensing stream. |
| `uplink_channel_ip` | `string` / IPv4 | `0.0.0.0` | ZeroMQ PUB listen IP for the BS uplink channel-estimation debug stream. |
| `uplink_channel_port` | `int` | `12358` | ZeroMQ PUB bind port for the BS uplink channel-estimation debug stream. |
| `uplink_pdf_ip` | `string` / IPv4 | `0.0.0.0` | ZeroMQ PUB listen IP for the BS uplink delay-profile debug stream. |
| `uplink_pdf_port` | `int` | `12359` | ZeroMQ PUB bind port for the BS uplink delay-profile debug stream. |
| `uplink_constellation_ip` | `string` / IPv4 | `0.0.0.0` | ZeroMQ PUB listen IP for the BS uplink constellation debug stream. |
| `uplink_constellation_port` | `int` | `12356` | ZeroMQ PUB bind port for the BS uplink constellation debug stream. |
| `rx_channel_count` | `int` | `1` | Number of sensing RX channels (`0` disables sensing RX). |
| `rx_channels` | `object[]` | `[]` | Per-channel sensing RX settings (see table below). |
| `tx_circular_buffer_size` | `int` | `32` | Capacity of the modulated-frame queue feeding TX. |
| `paired_frame_queue_size` | `int` | `64` | Capacity of each sensing channel's RX/TX frame-pairing queues. Keep this above `tx_circular_buffer_size` so it can retain TX references while RX startup, network buffering, and alignment complete. A continuously full queue after startup indicates insufficient sensing-processing throughput rather than a need for unlimited buffering. |
| `control_port` | `int` | `9999` | ZeroMQ ROUTER bind port for the bidirectional control channel (commands in, params/heartbeat out). |
| `measurement_enable` | `bool` | `false` | Enable CPU internal measurement mode. When enabled, `BS` generates deterministic PRBS payloads instead of listening on `udp_input_*`, and `UE` switches decoded measurement payloads into BER/BLER/EVM accounting. CUDA binaries ignore this mode. |
| `measurement_mode` | `string` | `internal_prbs` | Measurement mode selector. Only `internal_prbs` is supported. Unsupported values disable measurement mode during config normalization. |
| `measurement_run_id` | `string` | `""` | Run identifier written into measurement CSV summaries. |
| `measurement_output_dir` | `string` | `""` | Output directory used by the CPU measurement summaries. |
| `measurement_payload_bytes` | `int` | `1024` | Bytes per internally generated measurement payload. Values below the internal header size are clamped up. |
| `measurement_prbs_seed` | `int` | `0x5A` | Base seed used to derive deterministic PRBS payload contents. |
| `measurement_packets_per_point` | `int` | `1` | Number of measurement payloads sent for each online `MRST` epoch. Values below `1` are clamped to `1`. |
| `profiling_modules` | `string` | `""` | Profiling module list, comma-separated. Common values include `modulation`, `latency`, `data_ingest`, and `sensing_proc`; `all` enables every module. BS end-to-end latency profiling is enabled only when both `modulation` and `latency` are included. |
| `downlink_cpu_cores` | `int[]` | `[]` | BS downlink CPU cores: index `0` binds TX, `1` binds modulation, and `2` binds data ingest. |
| `uplink_cpu_cores` | `int[]` | `[]` | BS uplink CPU cores: indices `0`, `1`, and `2` bind RX sample ingest, OFDM/LLR signal processing, and LDPC decode + UDP output. |
| `main_cpu_core` | `int` | `-1` | Main-thread CPU core. |

Quick mental model:
* `data_resource_blocks` decides where communication data goes.
* `mask_blocks` decides which RE are exported for compact sensing.
* The first affects payload mapping; the second affects sensing output only.

If `data_resource_blocks` is enabled, copy the same rectangles and `kind` values into `UE.yaml`. If a block overlaps `sync_pos`, the optional second sync symbol at `sync_pos-1`, `midframe_pilot_symbols`, or `pilot_positions`, the built-in ZC sync symbols, comb-pilot RE, and mid-frame BPSK pilots still take precedence. The optional CFO training field at `sync_pos+1` is reserved for CFO acquisition/deambiguation and is not a valid sensing-pilot or sensing-mask symbol. Priority is always `ZC sync symbols > CFO training field > comb-pilot RE > mid-frame BPSK pilot > sensing_pilot > payload/random QPSK`.

In dense sensing mode, the configured `symbol_stride` and runtime `STRD` command are rejected if they would sample the optional CFO training field at `sync_pos+1`. Runtime `STRD` changes restart the deterministic sampling phase at the scheduled frame boundary, so the new stride does not inherit a drifting phase from the old stride. ZC sync symbols remain valid sensing symbols.

When `output_mode=compact_mask`, sensing sends one compact message per OFDM frame and includes only the RE selected by `mask_blocks`. In this mode `STRD` is ignored, because the mask itself defines the sampling pattern. If the mask is "regular" (same subcarrier set on every selected symbol, and selected symbols evenly spaced around the frame), runtime `MTI` and local Delay-Doppler processing can also be enabled: `SKIP=1` keeps the raw compact RE output, while `SKIP=0` switches back to dense Delay-Doppler output computed from that regular selection. Config normalization also expands `range_fft_size` and `doppler_fft_size` when needed so they cover the selected subcarriers and symbols. The compact sensing payload begins with `CompactSensingFrameHeader { magic/version, mask_hash, re_count, frame_start_symbol_index }`, followed by `re_count` raw `complex<float>` values in fixed order. Existing `plot_sensing*.py` viewers cannot handle non-"regular" compact payloads yet.

`rx_channels` object fields:

| Key | Type | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `usrp_channel` | `int` | `0` | USRP RX channel index. |
| `device_args` | `string` | `""` | Per-channel USRP args. |
| `clock_source` | `string` | `""` | Per-channel clock source override. |
| `time_source` | `string` | `""` | Per-channel time source override. |
| `wire_format` | `string` | `""` | Per-channel sensing RX wire format override. |
| `rx_gain` | `float` | `30` | RX gain for this channel. |
| `alignment` | `int` | `63` | Per-channel alignment offset (samples). |
| `rx_antenna` | `string` | `""` | RX antenna name, e.g. `TX/RX`, `RX1`. |
| `enable_system_delay_estimation` | `bool` | `false` | If `true`, this channel performs a ZC-based system delay estimation at startup and then once every 434 frames, while keeping the sensing pipeline disabled and continuing to drain frames. |
| `rx_cpu_core` | `int` | `-1` | CPU core for this channel's RX loop. |
| `processing_cpu_core` | `int` | `-1` | CPU core for this channel's sensing-processing loop. |

Notes:
* If `rx_channels` is empty and `rx_channel_count > 0`, default channels `0..N-1` are generated automatically.
* If the count and list size differ, the list is resized to match `rx_channel_count`.
* When `enable_system_delay_estimation=true` for a channel, that channel performs one system delay estimation near startup and then repeats it once every 434 frames while continuing to drain frames. Normal sensing processing and sensing output remain disabled.
* In practice, keep hardware-specific fields such as `device_args`, `wire_format_*`, per-channel RX antenna selection, and output IPs aligned with the actual radio/deployment you are using; the sample YAMLs are starting points, not universal presets.

### UE

`UE` is configured through `UE.yaml`.
Use `config/UE_X310.yaml`, `config/UE_B210.yaml`, or `config/UE_B210_Duplex.yaml` as a starting template.

`UE.yaml` parameter reference:

| Key | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `fft_size` | `int` | `1024` | OFDM FFT size. |
| `cp_length` | `int` | `128` | Cyclic prefix length (samples). |
| `sync_pos` | `int` | `1` | Sync symbol index in one frame. |
| `enable_sec_sync_symbol` | `bool` | `false` | Reserve `sync_pos-1` as a duplicate ZC sync symbol. When enabled, initial synchronization first uses the two consecutive sync symbols for Schmidl-Cox-style coarse timing/modulo-CFO estimation, resolves CFO aliases with local ZC correlation, then refines the main sync. Requires the transmitter to use the same setting. |
| `enable_cfo_training_sequence` | `bool` | `false` | Use the dedicated CFO training field at `sync_pos+1` to deambiguate CFO aliases. Frame timing still comes from ZC or the optional second sync symbol; CP/second-sync modulo CFO is estimated first, then the CFO field selects the alias closest to its repeated-training CFO estimate. Requires the transmitter to use the same setting. |
| `cfo_training_period_samples` | `int` / samples | `16` | Repetition period of the CFO training field. It must divide `fft_size`; the unambiguous CFO span is approximately `+-sample_rate/(2*period)`. |
| `sync_cfo_alias_search_range_hz` | `float` / Hz | `800000` | Maximum absolute CFO span covered by the sync alias resolver. The receiver converts this physical range to the required integer alias search span for the CP and second-sync modulo periods. Set `profiling_modules` to include `sync` to print per-alias peak comparisons. |
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
| `tx_channel` | `int` | `0` | TX channel index used by UE uplink when duplex/uplink is enabled. Downlink-only UE runs do not require this TX path. |
| `zc_root` | `int` | `29` | Zadoff-Chu root index. |
| `num_symbols` | `int` | `100` | Number of OFDM symbols per frame. |
| `sensing_symbol_num` | `int` | `100` | Number of symbols used for sensing processing. |
| `output_mode` | `string` | `dense` | Bistatic sensing output mode. `dense` keeps the legacy STRD-based full-buffer output. `compact_mask` switches sensing to per-frame compact RE extraction. |
| `bi_enabled` | `bool` | `true` | Enable the bistatic sensing processing pipeline. When set to `false`, both `UE` and `CUDAUE` skip bistatic sensing channel startup. |
| `duplex_mode` | `string` | `tdd` | Must match `BS.yaml`. `tdd` shares the downlink center frequency and sends only in the configured uplink symbol window; `fdd` transmits continuously over the full frame on `uplink.center_freq`. |
| `idle_waveform` | `string` | `random_qpsk` | UE uplink idle waveform when no UDP payload is queued. `random_qpsk` sends a zero-length mini-header followed by deterministic random QPSK filler; `zero` sends the zero-length mini-header and leaves the remaining payload RE at zero. |
| `uplink` | `object` | `symbol_start=90`, `symbol_count=10`, `guard_symbols=1`, `center_freq=2500000000` | UE uplink settings. TDD uses `symbol_start`, `symbol_count`, and `guard_symbols` and ignores `center_freq`; FDD uses `center_freq` and ignores the TDD symbol-window fields, transmitting over the full frame. Enabling uplink requires a UE TX antenna/RF chain; the BS must also have an uplink RX path. |
| `ue_timing_advance` | `int` / samples | `63` | UE-side uplink transmit timing advance. UE starts UL TX with the receiver at launch and later shifts future UL frames from RX synchronization/alignment plus this runtime-adjustable `TADV` value. |
| `cuda_demod_pipeline_slots` | `int` | `3` | Number of CUDA demodulation pipeline slots. Values below `1` are clamped to `1`. |
| `frame_queue_size` | `int` | `8` | Capacity of the UE RX frame queue. Values below `1` are clamped to `1`. |
| `sync_queue_size` | `int` | `8` | Capacity of the UE sync-search batch queue. Values below `1` are clamped to `1`. |
| `reset_hold_s` | `float` / s | `0.5` | How long invalid delay conditions must persist before the UE forces a hard reset back to sync search. Internally this is converted to a frame count from `samples_per_frame / sample_rate`. Values below `0` are clamped to `0.5`. |
| `range_fft_size` | `int` | `1024` | Range FFT size. |
| `doppler_fft_size` | `int` | `100` | Doppler FFT size. |
| `pilot_positions` | `int[]` | `[571,631,...,451]` | Configurable comb-pilot subcarrier indices spread across the occupied band. |
| `midframe_pilot_symbols` | `int[]` | `[]` | Optional mid-frame BPSK pilot symbol indices inside each frame. The receiver uses the full known symbol as an additional channel-estimation anchor, keeps comb-pilot RE available for phase tracking, and excludes the symbol from payload LLR extraction. |
| `midframe_pilot_seed` | `int` | `1296453708` | Deterministic BPSK pilot seed. It must match the transmitter. |
| `equalizer_mode` | `string` | `mmse` | Communication equalizer inverse. `zf` uses a floored channel-power denominator; `mmse` adds `noise_var` to that denominator to reduce noise enhancement on deep fades. |
| `channel_tracking_mode` | `string` | `pilot_phase` | Per-symbol comb-pilot tracking for communication equalization on both CPU and CUDA demodulators. `disabled` keeps the sync-only channel estimate, while `pilot_phase` fits common and linear residual phase from comb pilots on each data symbol. |
| `equalizer_mag_floor` | `float` | `1e-6` | Lower bound for channel magnitude squared during inversion, used by both `zf` and `mmse`. |
| `channel_tracking_min_pilot_snr` | `float` | `1e-4` | Minimum comb-pilot residual power/weight accepted by per-symbol tracking before falling back to the sync-only correction. |
| `data_resource_blocks` | `object[]` | omitted | Receiver-side communication resource map. It answers: "which RE should be interpreted as payload?" Omit the key to keep the legacy behavior, where every non-sync, non-comb-pilot RE is treated as payload. Set `[]` to extract no payload LLR at all. Use the same rectangles and `kind` values as the transmitter. Blocks with `kind: payload` produce payload LLR; blocks with `kind: sensing_pilot` are treated as known reference RE instead and are excluded from payload extraction. The known sensing-pilot reference uses the same alternate Zadoff-Chu root as the transmitter, distinct from the frame sync root. |
| `mask_blocks` | `object[]` | omitted | Receiver-side compact sensing resource map. It answers: "which RE should be exported on the bistatic sensing path in `compact_mask` mode?" The coordinate system and behavior are the same as on the BS side: absolute frame-symbol index, raw FFT-bin index, ZC sync-symbol, comb-pilot, and mid-frame BPSK pilot RE allowed, CFO training field rejected, overlapping blocks merged automatically, and exported order fixed as symbol-major then subcarrier-major. If the mask is regular, runtime MTI and local Delay-Doppler processing can also be enabled. |
| `device_args` | `string` | `""` | USRP device args. |
| `clock_source` | `string` | `internal/external/gpsdo` | Clock source. |
| `wire_format_tx` | `string` | `sc16` | TX wire format for the optional UE uplink path, typically `sc16` or `sc8`. |
| `rx_wire_format` | `string` | `sc16` | UE downlink RX wire format, typically `sc16` or `sc8`. |
| `software_sync` | `bool` | `true` | Enable software synchronization tracking. |
| `predictive_delay` | `bool` | `true` | Enable CFO-based predictive delay compensation during initial alignment and tracking delay correction. Use this only when the sample clock and carrier frequency are derived from the same reference, and there is no secondary frequency conversion outside the USRP. |
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
| `bi_sensing_output_enabled` | `bool` | `true` | Enable the bistatic sensing ZeroMQ PUB output. The processing pipeline can remain enabled while this output is disabled. |
| `bi_sensing_ip` | `string` / IPv4 | `0.0.0.0` | ZMQ bind IP for the bistatic sensing stream and control channel. Use `0.0.0.0` to accept remote viewers, or `127.0.0.1` for local-only access. |
| `bi_sensing_port` | `int` | `8889` | ZeroMQ PUB bind port for the bistatic sensing data stream. |
| `channel_ip` | `string` / IPv4 | `0.0.0.0` | ZeroMQ PUB listen IP for channel-estimation output. Empty values also resolve to `0.0.0.0`, not `default_out_ip`. |
| `channel_port` | `int` | `12348` | ZeroMQ PUB bind port for channel-estimation output. |
| `pdf_ip` | `string` / IPv4 | `0.0.0.0` | ZeroMQ PUB listen IP for PDP/PDF output. Empty values also resolve to `0.0.0.0`, not `default_out_ip`. |
| `pdf_port` | `int` | `12349` | ZeroMQ PUB bind port for PDP/PDF output. |
| `constellation_ip` | `string` / IPv4 | `0.0.0.0` | ZeroMQ PUB listen IP for constellation output. Empty values also resolve to `0.0.0.0`, not `default_out_ip`. |
| `constellation_port` | `int` | `12346` | ZeroMQ PUB bind port for constellation output. |
| `vofa_debug_ip` | `string` / IPv4 | `127.0.0.1` | Destination IP for VOFA+ debug output. |
| `vofa_debug_port` | `int` | `12347` | Destination port for VOFA+ debug output. |
| `udp_input_ip` | `string` / IPv4 | `0.0.0.0` | UE uplink payload UDP bind IP. This is the input stream transmitted on the UE->BS uplink. |
| `udp_input_port` | `int` | `50002` | UE uplink payload UDP bind port. |
| `udp_output_ip` | `string` / IPv4 | `127.0.0.1` | UE decoded downlink payload UDP destination IP. This is the output stream recovered from the BS->UE downlink. |
| `udp_output_port` | `int` | `50001` | UE decoded downlink payload UDP destination port. |
| `default_out_ip` | `string` / IPv4 | `127.0.0.1` | Default destination IP for UDP payload and VOFA+ debug outputs when those IP fields are empty. ZeroMQ PUB listen IPs do not inherit this value. |
| `control_port` | `int` | `10001` | ZeroMQ ROUTER bind port for the bidirectional control channel. |
| `measurement_enable` | `bool` | `false` | Enable CPU internal measurement mode. In this mode, decoded measurement payloads are consumed locally for BER/BLER/EVM statistics instead of being forwarded to `udp_output_*`. CUDA binaries ignore this mode. |
| `measurement_mode` | `string` | `internal_prbs` | Measurement mode selector. Only `internal_prbs` is supported. Unsupported values disable measurement mode during config normalization. |
| `measurement_run_id` | `string` | `""` | Run identifier written into measurement CSV summaries. |
| `measurement_output_dir` | `string` | `""` | Output directory used by the CPU measurement summaries. |
| `measurement_payload_bytes` | `int` | `1024` | Expected bytes per measurement payload. Values below the internal header size are clamped up. |
| `measurement_prbs_seed` | `int` | `0x5A` | Base seed used to rebuild deterministic PRBS measurement payloads. |
| `measurement_packets_per_point` | `int` | `1` | Expected measurement payload count for each online `MRST` epoch. Values below `1` are clamped to `1`. |
| `profiling_modules` | `string` | `""` | Profiling module list, comma-separated. Common values include `demodulation`, `sync`, `agc`, `align`, `snr`, and `uplink`; `all` enables every module. `sync` gates per-alias synchronization peak comparisons, `agc` gates AGC logs, `align` gates runtime `ALGN:` logs, `snr` prints periodic `_snr_db / _noise_var / _llr_scale` updates, and `uplink` gates `[UL-TX]` timing/waveform diagnostics. |
| `downlink_cpu_cores` | `int[]` | `[]` | UE downlink CPU cores: indices `0..3` bind `rx_proc`, `process_proc`, `sensing_process_proc`, and `bit_processing_proc`. |
| `uplink_cpu_cores` | `int[]` | `[]` | UE uplink CPU cores: indices `0`, `1`, and `2` bind `UplinkTxEngine::_udp_ingest_proc`, `_mod_proc`, and `_tx_proc`. |
| `main_cpu_core` | `int` | `-1` | Main-thread CPU core. |

Receiver-side note:
* `data_resource_blocks` should normally match the transmitter exactly, including `kind`.
* If a resource block overlaps `sync_pos`, the optional second sync symbol at `sync_pos-1`, `midframe_pilot_symbols`, or `pilot_positions`, the built-in ZC sync symbols, comb-pilot RE, and mid-frame BPSK pilots still win. The optional CFO training field at `sync_pos+1` is rejected for sensing-pilot and sensing-mask selection. Priority is `ZC sync symbols > CFO training field > comb-pilot RE > mid-frame BPSK pilot > sensing_pilot > payload/random QPSK`.
* In dense mode, `symbol_stride` / runtime `STRD` is rejected if it would sample the optional CFO training field at `sync_pos+1`. Runtime `STRD` changes restart the deterministic sampling phase at the scheduled frame boundary; ZC sync symbols remain valid sensing symbols.
* In `compact_mask` mode, bistatic sensing also sends one compact message per OFDM frame and includes only the RE selected by `mask_blocks`; `STRD` is ignored in this mode because the mask already defines the sampling pattern.
* The compact payload format is the same as on the BS side: `CompactSensingFrameHeader` followed by fixed-order raw `complex<float>` samples.

Notes:
* RX AGC has two phases. During `SYNC_SEARCH`, the receiver resets gain to the configured `rx_gain` and performs a coarse search sweep (+1 dB every 10 frames, wrapping from max gain back to min gain). After lock, tracking AGC uses the filtered `delay_spectrum` peak window defined by `rx_agc_low_threshold_db` and `rx_agc_high_threshold_db`.
* Sync-symbol time-domain samples are also checked for near/full-scale ADC usage. If too many I/Q components approach full scale, the UE receiver forces gain reduction and temporarily blocks gain increases to avoid ping-pong behavior.
* A hard reset clears timing/frequency tracking state, flushes pending queues, resets the tracking AGC state, and returns the receiver to `SYNC_SEARCH`. `reset_hold_s` controls how long bad delay conditions must persist before this happens.
