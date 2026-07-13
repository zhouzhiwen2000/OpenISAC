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
- USRP-free simulation: [Channel Simulator](docs/CHANNEL_SIMULATOR.md)
- Web control UI: [Web Config Console](#9-web-config-console)
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
| Run without USRPs | `ChannelSimulator`, `BS`, `UE` | `config/BS_Sim.yaml` and `config/UE_Sim.yaml` | See [Channel Simulator](docs/CHANNEL_SIMULATOR.md) |
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

### 7. Calibrate system delay

Run the system-delay calibration before `Calibrate Hsys`, because the response calibration assumes the sensing RX frame is already aligned to the direct-path timing reference.

First reduce transmit power for the direct connection. Lower `downlink.tx_gain` in `build/BS.yaml` and, if needed, insert a suitable attenuator so the sensing RX path is not saturated. Then connect the transmit RF output directly to the sensing RX input for the channel being measured, and keep that cable path stable during the test.

Enable system-delay estimation for only the sensing channel being measured:

```yaml
sensing:
  rx_channels:
    - usrp_channel: 1
      alignment: 63
      enable_system_delay_estimation: true
```

Start the BS backend from the `build/` directory. In this mode, the selected sensing channel disables the normal sensing pipeline and periodically runs a ZC-based delay test. Watch the BS console for `[SYSDLY CH <n>]` on CPU builds or `[CUDA SYSDLY CH <n>]` on CUDA builds. The CPU log prints `alignment_suggest=<value>`; the CUDA log prints `suggest=<value>`.

When the suggested value is stable, stop the backend, write that value back to the same channel's `alignment` field in `build/BS.yaml`, and turn the estimation mode off:

```yaml
sensing:
  rx_channels:
    - usrp_channel: 1
      alignment: <suggested value>
      enable_system_delay_estimation: false
```

Repeat the same direct-connection measurement for every sensing channel. For multichannel setups, move the RF direct connection to the next sensing RX path and update that channel's own `alignment`; do not reuse one channel's value for another RF path.

### 8. Calibrate sensing channel response

Before calibration, connect the sensing RF path directly: connect the transmit RF output to the corresponding sensing RX input and keep the connection stable during calibration.

If the transmit power is high, reduce the transmit power first. If needed, insert a suitable attenuator in the direct RF path to avoid RX saturation. The attenuator should be as flat as possible across the signal bandwidth; otherwise its in-band ripple will be included in the calibration result.

For monostatic sensing, start the BS backend and the monostatic sensing frontend, select the sensing channel you want to calibrate in the viewer, then click `Calibrate Hsys`. For multichannel monostatic setups, repeat this for each channel that needs its own RF path calibration.

For bistatic sensing, start the BS and UE backends, open the bistatic sensing frontend, then click `Calibrate Hsys` in the bistatic viewer while the RF path is directly connected.

Wait for the backend log to report that calibration has completed and the calibration file has been saved. The current run will immediately use the new response calibration. On later launches, the backend automatically loads the matching calibration file; if no matching file is found, sensing continues without calibration and prints a notice.

After calibration is complete, restore the normal antenna or experiment connection before continuing OTA measurements.

### 9. Web Config Console
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

The runtime config is hierarchical YAML. The tables below use full YAML paths so similarly named fields, such as `downlink.arq_enabled` and `uplink.arq_enabled`, stay unambiguous. Optional sections can be omitted; missing values use the parser defaults and the sample files under `config/` show common hardware and simulator presets.

### BS

`BS` reads `BS.yaml` from its current working directory. Start from `config/BS_X310.yaml`, `config/BS_B210.yaml`, the duplex presets, or the simulator presets.

#### BS radio

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `radio.radio_backend` | `string` | `uhd` | Radio I/O backend. Use `uhd` for real USRPs or `sim` for the shared-memory channel simulator. |

#### BS simulation

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
| `simulation.cfo_hz` | `float` / Hz | `0` | Initial carrier offset injected before UE RX correction. |
| `simulation.sample_rate_offset_ppm` | `float` / ppm | `0` | UE sample-clock offset relative to the BS clock. |
| `simulation.timing_offset_samples` | `int` / samples | `0` | Constant integer sample delay injected on RX. |
| `simulation.array_spacing_m` | `float` / m | `0.04283` | Physical ULA element spacing; set `<=0` to use `array_spacing_lambda`. |
| `simulation.array_spacing_lambda` | `float` / lambda | `0.5` | Legacy ULA spacing in wavelengths. |
| `simulation.ring_capacity_samples` | `int` / samples | `262144` | Per-stream shared-memory ring capacity. |
| `simulation.steering_override_file` | `string` | `""` | Optional steering matrix file; empty uses ULA steering. |
| `simulation.comm_multipath_taps[]` | `object[]` | optional | Communication tapped-delay-line taps with `delay_samples`, `gain_db`, and `phase_deg`. |
| `simulation.targets[]` | `object[]` | optional | Monostatic point scatterers with `range_m`, `velocity_mps`, `gain_db`, and `angle_deg`. |
| `simulation.bistatic_targets[]` | `object[]` | optional | Bistatic/communication point scatterers with the same target fields. |

#### BS rf_sampling

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `rf_sampling.sample_rate` | `float` / Hz | `50000000` | Baseband sample rate. |
| `rf_sampling.bandwidth` | `float` / Hz | `50000000` | Analog bandwidth, usually matching `sample_rate`. |

#### BS usrp_device

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `usrp_device.device_args` | `string` | `addr=...` | Shared USRP device args fallback. |

#### BS clock_time

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `clock_time.clock_source` | `string` | `external` | Global clock source: `internal`, `external`, or `gpsdo`. |
| `clock_time.time_source` | `string` | `internal` | Global time/PPS source; empty follows `clock_source`. |

#### BS ofdm_frame

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

#### BS cuda

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `cuda.cuda_mod_pipeline_slots` | `int` | `3` | CUDA modulation pipeline slots; values below `1` are clamped. |

#### BS ldpc

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `ldpc.fixed_point` | `bool` | `false` | Use the int16/Q16 layered-NMS CPU decoder instead of float32. |
| `ldpc.fixed_point_scale` | `int` | `16` | Power-of-two LLR scale before int16 saturation in fixed-point mode. |

#### BS downlink

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
| `downlink.arq_window_packets` | `int` | `32767` | Downlink ARQ outstanding packet window. |
| `downlink.arq_retransmit_timeout_ms` | `int` / ms | `100` | Downlink ARQ retransmission timeout. |
| `downlink.arq_max_retries` | `int` | `5` | Max downlink retransmission retries; `0` means unlimited within the window. |

#### BS downlink_pipeline

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `downlink_pipeline.tx_circular_buffer_size` | `int` | `8` | Capacity of the modulated-frame queue feeding TX. |
| `downlink_pipeline.data_packet_buffer_size` | `int` | `256` | Capacity of the encoded-packet buffer. |

#### BS uplink

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `uplink.enabled` | `bool` | `false` | Master switch for the UE-to-BS uplink/duplex path. |
| `uplink.duplex_mode` | `string` | `tdd` | `tdd` uses an uplink symbol window; `fdd` uses `uplink.center_freq` and a full-frame uplink. |
| `uplink.center_freq` | `float` / Hz | `2500000000` | FDD-only uplink carrier. TDD uses the downlink center frequency. |
| `uplink.symbol_start` | `int` | `90` | TDD-only first uplink symbol in the downlink frame. |
| `uplink.symbol_count` | `int` | `10` | TDD-only uplink window length; `0` disables TDD uplink. |
| `uplink.guard_symbols` | `int` | `1` | TDD-only leading guard symbols inside the uplink window. |
| `uplink.bs_dl_ul_timing_diff` | `int` / samples | `63` | BS-side uplink RX window offset relative to the downlink TX frame anchor. |
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
| `uplink.arq_window_packets` | `int` | `32767` | Uplink ARQ receive/reorder window. |
| `uplink.arq_feedback_interval_ms` | `int` / ms | `10` | Minimum interval between uplink ARQ ACK feedback packets. |

#### BS sensing

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

#### BS sensing.rx_channels[] fields

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

#### BS resource_preview

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `resource_preview.data_resource_blocks[]` | `object[]` | optional | Payload / sensing-pilot RE rectangles. Each item has `kind`, `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`. |
| `resource_preview.mask_blocks[]` | `object[]` | optional | Compact sensing RE rectangles with `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`. |

#### BS measurement

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

#### BS network_output

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

#### BS cpu_cores

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `cpu_cores.downlink_cpu_cores` | `int[]` | `[1,2,3,-1]` | BS downlink cores: TX, modulation, LDPC encode, UDP receive. |
| `cpu_cores.uplink_cpu_cores` | `int[]` | `[]` | BS uplink cores: RX ingest, OFDM/LLR processing, LDPC decode + UDP output. |
| `cpu_cores.main_cpu_core` | `int` | `-1` | Main-thread CPU core; `-1` disables binding. |

#### BS runtime

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `runtime.profiling_modules` | `string` | `""` | Comma-separated profiling modules such as `modulation`, `latency`, `ldpc_encode`, `sensing_proc`, `agc`, `arq`, `uplink`, `ertm`, or `all`. |

### UE

`UE` reads `UE.yaml` from its current working directory. Start from `config/UE_X310.yaml`, `config/UE_B210.yaml`, the duplex presets, or the simulator presets.

#### UE radio

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `radio.radio_backend` | `string` | `uhd` | Radio I/O backend. Use `uhd` for real USRPs or `sim` for the channel simulator. |

#### UE simulation

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
| `simulation.cfo_hz` | `float` / Hz | `0` | Initial carrier offset. |
| `simulation.sample_rate_offset_ppm` | `float` / ppm | `0` | UE sample-clock offset relative to BS. |
| `simulation.timing_offset_samples` | `int` / samples | `0` | Constant integer sample delay. |
| `simulation.array_spacing_m` | `float` / m | `0.04283` | Physical ULA spacing. |
| `simulation.array_spacing_lambda` | `float` | `0.5` | Legacy ULA spacing in wavelengths. |
| `simulation.ring_capacity_samples` | `int` | `262144` | Shared-memory ring capacity. |
| `simulation.steering_override_file` | `string` | `""` | Optional steering matrix file. |
| `simulation.comm_multipath_taps[]` | `object[]` | optional | Communication taps with `delay_samples`, `gain_db`, and `phase_deg`. |
| `simulation.targets[]` | `object[]` | optional | Monostatic point scatterers. |
| `simulation.bistatic_targets[]` | `object[]` | optional | Bistatic/communication point scatterers. |

#### UE rf_sampling

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

#### UE usrp_device

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `usrp_device.device_args` | `string` | `addr=...` | USRP device args. |
| `usrp_device.clock_source` | `string` | `external` | UE clock source: `internal`, `external`, or `gpsdo`. |

#### UE ofdm_frame

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

#### UE cuda

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `cuda.cuda_demod_pipeline_slots` | `int` | `3` | CUDA demodulation pipeline slots. |
| `cuda.cuda_ldpc_decoder_backend` | `string` | `gpu` | CUDA demod LDPC decoder backend: `gpu` or `cpu`. |
| `cuda.cuda_ldpc_worker_buffers` | `int` | `3` | CUDA LDPC async worker batch buffers. |
| `cuda.cuda_ldpc_cross_frame_flush_frames` | `int` | `2` | Max frames accumulated before CUDA LDPC batch decode. |
| `cuda.cuda_ldpc_cross_frame_flush_us` | `float` / us | `1000` | Max CUDA LDPC cross-frame batch wait time. |

#### UE ldpc

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `ldpc.fixed_point` | `bool` | `false` | Use int16/Q16 CPU LDPC decode path. |
| `ldpc.fixed_point_scale` | `int` | `16` | LLR scale before int16 saturation. |

#### UE downlink

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

#### UE uplink

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
| `uplink.ue_timing_advance` | `int` / samples | `63` | UE uplink transmit timing advance. |
| `uplink.debug_self_channel` | `bool` | `false` | Estimate UE self-TX leakage channel from RX windows for `TADV` debugging. |
| `uplink.ertm_to_enable` | `bool` | `false` | Enable eRTM TO payload consumption and TO logs. |
| `uplink.ertm_delay_oversample_factor` | `int` | `10` | eRTM delay-spectrum IFFT oversampling factor. |
| `uplink.ertm_dl_rf_delay_ns` | `float` / ns | `0` | Calibrated downlink RF-chain delay for eRTM equations. |
| `uplink.ertm_ul_rf_delay_ns` | `float` / ns | `0` | Calibrated uplink RF-chain delay for eRTM equations. |
| `uplink.ertm_debug_output_enabled` | `bool` | `false` | Enable UE-side eRTM debug ZMQ spectra. |
| `uplink.ertm_report_interval_frames` | `int` / frames | `32` | BS eRTM report cadence; keep matched with BS. |
| `uplink.arq_enabled` | `bool` | `false` | Enable uplink ARQ on the UE transmitter. |
| `uplink.arq_window_packets` | `int` | `32767` | Uplink ARQ outstanding packet window. |
| `uplink.arq_retransmit_timeout_ms` | `int` / ms | `100` | Uplink ARQ retransmission timeout. |
| `uplink.arq_max_retries` | `int` | `5` | Max uplink retransmission retries; `0` means unlimited within the window. |

#### UE sync_tracking

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

#### UE sensing

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

#### UE resource_preview

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `resource_preview.data_resource_blocks[]` | `object[]` | optional | Payload / sensing-pilot RE rectangles; each item has `kind`, `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`. |
| `resource_preview.mask_blocks[]` | `object[]` | optional | Compact sensing RE rectangles with `symbol_start`, `symbol_count`, `subcarrier_start`, and `subcarrier_count`. |

#### UE measurement

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

#### UE network_output

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

#### UE cpu_cores

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `cpu_cores.downlink_cpu_cores` | `int[]` | `[1,2,3]` | UE downlink cores: RX, processing, bit processing. |
| `cpu_cores.demod_worker_cpu_cores` | `int[]` | `[]` | UE CPU demod worker cores; empty starts one unbound worker. |
| `cpu_cores.ldpc_worker_cpu_cores` | `int[]` | `[]` | UE CPU LDPC decode worker cores; empty starts one unbound worker. |
| `cpu_cores.sensing_cpu_cores` | `int[]` | `[4]` | UE bistatic sensing cores. |
| `cpu_cores.uplink_cpu_cores` | `int[]` | `[]` | UE uplink cores: LDPC encode, modulation, TX send, UDP receive. |
| `cpu_cores.main_cpu_core` | `int` | `-1` | Main-thread CPU core; `-1` disables binding. |

#### UE runtime

| Path | Type/Unit | Typical Value | Description |
| :--- | :--- | :--- | :--- |
| `runtime.default_out_ip` | `string` / IPv4 | `127.0.0.1` | Default destination IP for UDP and VOFA+ outputs when specific IP fields are empty. |
| `runtime.vofa_debug_ip` | `string` / IPv4 | `""` | VOFA+ debug destination IP; empty uses `default_out_ip`. |
| `runtime.vofa_debug_port` | `int` | `12347` | VOFA+ debug destination port. |
| `runtime.profiling_modules` | `string` | `""` | Comma-separated modules such as `demodulation`, `cfo`, `sync`, `agc`, `align`, `snr`, `arq`, `uplink`, `ertm`, or `all`. |

Resource-map notes:
* `resource_preview.data_resource_blocks` should normally match between BS and UE, including `kind`.
* Built-in ZC sync symbols, the optional CFO training field, comb pilots, and mid-frame BPSK pilots take precedence over configured resource blocks.
* `resource_preview.mask_blocks` controls compact sensing export only. In `output_mode=compact_mask`, runtime `STRD` is ignored because the mask defines the sampling pattern.
* Compact sensing payloads begin with `CompactSensingFrameHeader`, followed by fixed-order raw `complex<float>` samples.
