# OpenISAC

[中文版本](README_zh.md)

OpenISAC is a simple OFDM-based communication and sensing system designed for academic experiments and rapid algorithm validation.

Its goal is to provide a clean, minimal, easy-to-modify OFDM platform so researchers can iterate quickly on PHY/sensing ideas without the overhead of a full standard-compliant stack.

Because it focuses on simplicity, OpenISAC typically requires less compute and can often run at higher sampling rates than more complex, feature-complete systems.

If you find this repository useful, please cite our paper:

> Z. Zhou, C. Zhang, X. Xu, and Y. Zeng, "OpenISAC: An Open-Source Real-Time Experimentation Platform for OFDM-ISAC with Over-the-Air Synchronization," submitted to *IEEE Trans. Wireless Commun.*, Jan. 2026.
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

## What it is — and what it is not

### What OpenISAC is

- A minimal OFDM-based PHY for joint communication and sensing research
- Designed for prototyping, academic experiments, and rapid algorithm validation

### What OpenISAC is not

- A standard-compliant implementation (it does not aim to comply with standards such as Wi‑Fi or 5G NR)
- A replacement for full-stack standard implementations like openwifi or OpenAirInterface

If your goal is interoperability, standards compliance, or a production-grade protocol stack, those projects are the right direction.

### When to use it

- Prototyping and testing new OFDM/ISAC algorithms
- Fast “idea → experiment” cycles with a minimal PHY
- Research setups where interoperability is not required

### When not to use it

- Building a Wi‑Fi/NR-compatible system
- Needing real-world standard features (full MAC/stack behavior, interoperability, certification-oriented behavior, etc.)

## Hardware Requirements
 
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
*   **GPU**: A Nvidia GPU is recommended for acceleration.


## Software Requirements
 
### Backend (C++)
 
#### Operating System
*   **Ubuntu 24.04 LTS**
    *   Download: [http://www.ubuntu.com/download/desktop](http://www.ubuntu.com/download/desktop)
 
#### Dependencies & Installation
 
##### 1. UHD (USRP Hardware Driver)
Install the UHD toolchain by following the official Ettus guide (Please follow the tutorial for Ubuntu 24.04):
*   [Building and Installing the USRP Open-Source Toolchain on Linux](https://kb.ettus.com/Building_and_Installing_the_USRP_Open-Source_Toolchain_(UHD_and_GNU_Radio)_on_Linux#Update_and_Install_dependencies)
 
> **Note:** This code has been tested on UHD v4.9.0.1. You can checkout this version using `git checkout v4.9.0.1`.
 
##### 2. Install Aff3ct
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
 
##### 3. Clone Repository
Clone the OpenISAC repository:
 
```bash
cd ~
git clone https://github.com/zhouzhiwen2000/OpenISAC.git
```
 
##### 4. Build OpenISAC
Build the project using CMake:
 
```bash
sudo apt-get install libyaml-cpp-dev
cd OpenISAC
mkdir build
cd build
cmake ..
make -j$(nproc)
```
 
##### 5. System Performance Tuning
Run the provided script to optimize your system settings for real-time processing:
 
```bash
cd ~/OpenISAC
chmod +x set_performance.bash
./set_performance.bash
```
 
> **Note:** `secure_boot` needs to be turned off in your BIOS settings to enable `RT_RUNTIME_SHARE` functionality.
 
###### UHD Thread Priority Configuration
 
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
 
##### 6. CPU Isolation and Execution
To ensure stable real-time performance, it is recommended to isolate CPU cores for the signal processing tasks. We provide a script `isolate_cpus.bash` to handle this automatically.
 
**Step 1: Isolate System Cores**
This command will restrict system processes to E-Cores (on Intel Hybrid Architecture) or a subset of cores, leaving P-Cores or other CPU cores free for the application.
```bash
cd ~/OpenISAC
chmod +x isolate_cpus.bash
sudo ./isolate_cpus.bash
```
 
**Step 2: Run Application on Isolated Cores**
Use the `run` command of the script to launch your application on the isolated Cores. This ensures the application has exclusive access to high-performance resources.
 
```bash
cd build
sudo ../isolate_cpus.bash run ./OFDMModulator
```
 
> **Note:** Always use `sudo ../isolate_cpus.bash run ...` to launch the application if you have enabled CPU isolation. Direct execution or using `taskset` manually might fail due to affinity restrictions.
 
**Reset Configuration (Optional)**
To restore the system to its default state (allowing all processes to use all cores):
```bash
sudo ./isolate_cpus.bash reset
```

##### 7. Configuration
The system now supports YAML configuration for easier parameter management.

*   **Default Behavior**:
    If `Modulator.yaml` or `Demodulator.yaml` exists in the execution directory, it will be loaded automatically.

*   **Specify Config File**:
    Use `-c` or `--config` to specify a custom configuration file:
    ```bash
    ./OFDMModulator -c my_config.yaml
    ```

*   **Save Configuration**:
    Use `-s` or `--save-config` to save the current parameters (including defaults and command-line overrides) to a YAML file:
    ```bash
    ./OFDMModulator --save-config
    ```
    This will generate `Modulator.yaml` (or `Demodulator.yaml`) with the current settings.
    
    You can also specify a custom filename:
    ```bash
    ./OFDMModulator -s myconfig.yaml
    ```
 
### Frontend (Python)

It is recommended to use a `conda` or `venv` environment with **Python 3.13**.
 
#### Install Dependencies
 
```bash
pip install -r requirements.txt
```

**Note:** `ffmpeg` is required for video streaming demonstrations.
*   **Ubuntu:** `sudo apt install ffmpeg`
*   **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH, or place executable in the working directory.
 
#### Enable GPU Acceleration (Optional)
 
If an Nvidia GPU is available, install `cupy-cuda12x` to enable GPU acceleration:
 
```bash
pip install cupy-cuda12x
```

## Typical Usage Example

### 1. Startup of the BS
```bash
sudo -s
cd build
# For X310:
cp ../Modulator_X310.yaml Modulator.yaml
sudo ../isolate_cpus.bash run ./OFDMModulator

# For B210:
cp ../Modulator_B210.yaml Modulator.yaml
sudo ../isolate_cpus.bash run ./OFDMModulator
```
*Add `--default-ip=<your front end IP>` if you are using a separate computer for the frontend.*

### 2. Startup of the UE
```bash
sudo -s
cd build
# For X310:
cp ../Demodulator_X310.yaml Demodulator.yaml
sudo ../isolate_cpus.bash run ./OFDMDemodulator

# For B210:
cp ../Demodulator_B210.yaml Demodulator.yaml
sudo ../isolate_cpus.bash run ./OFDMDemodulator
```
*Add `--default-ip=<your front end IP>` if you are using a separate computer for the frontend.*

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
python3 ./plot_sensing_microDoppler_CPP_MTI.py
```

### 6. Run bistatic frontend
```bash
python3 ./plot_bi_sensing_microDoppler_CPP_MTI.py
```

### OFDM Modulator

The `OFDMModulator` (BS Node) can be configured using the following command-line arguments:

| Argument | Default Value | Description |
| :--- | :--- | :--- |
| `--args` | `addr=192.168.40.2, master_clock_rate=200e6, num_recv_frames=512, num_send_frames=512` | USRP device arguments |
| `--fft-size` | `1024` | FFT size |
| `--cp-length` | `128` | Cyclic Prefix length |
| `--sync-pos` | `1` | Synchronization symbol position index |
| `--sample-rate` | `50e6` | Sample rate (Hz) |
| `--bandwidth` | `50e6` | Analog bandwidth (Hz) |
| `--center-freq` | `2.4e9` | Center frequency (Hz) |
| `--tx-gain` | `20` | Transmission gain (dB) |
| `--rx-gain` | `30` | Reception gain (dB) |
| `--rx-channel` | `1` | RX channel index on the USRP |
| `--zc-root` | `29` | Zadoff-Chu sequence root |
| `--num-symbols` | `100` | Number of OFDM symbols per frame |
| `--clock-source` | `external` | Clock source (`internal` or `external`) |
| `--system-delay` | `63` | System delay samples for alignment |
| `--wire-format-tx` | `sc16` | USRP TX wire format (`sc8` or `sc16`) |
| `--wire-format-rx` | `sc16` | USRP RX wire format (`sc8` or `sc16`) |
| `--mod-udp-ip` | `0.0.0.0` | IP to bind for incoming UDP data payload |
| `--mod-udp-port` | `50000` | Port to bind for incoming UDP data payload |
| `--sensing-ip` | `127.0.0.1` | Destination IP for sensing data |
| `--sensing-port` | `8888` | Destination port for sensing data |
| `--default-ip` | `127.0.0.1` | Default IP for all services |
| `--cpu-cores` | `0,1,2,3,4,5` | Comma-separated list of CPU cores to use |

### OFDM Demodulator

The `OFDMDemodulator` (UE Node) supports the following arguments:

| Argument | Default Value | Description |
| :--- | :--- | :--- |
| `--device-args` | `num_recv_frames=512, num_send_frames=512, send_frame_size=11520, recv_frame_size=11520` | USRP device arguments |
| `--fft-size` | `1024` | FFT size |
| `--cp-length` | `128` | Cyclic Prefix length |
| `--center-freq` | `2.4e9` | Center frequency (Hz) |
| `--sample-rate` | `50e6` | Sample rate (Hz) |
| `--bandwidth` | `50e6` | Analog bandwidth (Hz) |
| `--rx-gain` | `60` | Reception gain (dB) |
| `--rx-channel` | `0` | RX channel index |
| `--sync-pos` | `1` | Synchronization symbol position index |
| `--sensing-ip` | `127.0.0.1` | IP for sending sensing data |
| `--sensing-port` | `8889` | Port for sending sensing data |
| `--control-port` | `9999` | Port for receiving control commands (Heartbeat, MTI, etc.) |
| `--channel-ip` | `127.0.0.1` | IP for channel estimation output |
| `--channel-port` | `12348` | Port for channel estimation output |
| `--pdf-ip` | `127.0.0.1` | IP for raw power delay profile (PDF) data output |
| `--pdf-port` | `12349` | Port for raw power delay profile (PDF) data output |
| `--constellation-ip` | `127.0.0.1` | IP for constellation diagram data |
| `--constellation-port` | `12346` | Port for constellation diagram data |
| `--freq-offset-ip` | `127.0.0.1` | IP for frequency offset data |
| `--freq-offset-port` | `12347` | Port for frequency offset data |
| `--udp-output-ip` | `127.0.0.1` | Destination IP for decoded user data |
| `--udp-output-port` | `50001` | Destination port for decoded user data |
| `--zc-root` | `29` | Zadoff-Chu sequence root |
| `--num-symbols` | `100` | Number of symbols per frame |
| `--sensing-symbol-num`| `100` | Number of symbols used for sensing processing |
| `--clock-source` | `external` | Clock source (`internal` or `external`) |
| `--software-sync` | `true` | Enable software synchronization/tracking |
| `--hardware-sync` | `false` | Enable hardware synchronization (disables software sync) |
| `--hardware-sync-tty` | `/dev/ttyUSB0` | TTY device for hardware sync |
| `--wire-format-rx` | `sc16` | USRP RX wire format |
| `--default-ip` | `127.0.0.1` | Default IP for all services |
| `--cpu-cores` | `0,1,2,3,4,5` | Comma-separated list of CPU cores to use |
