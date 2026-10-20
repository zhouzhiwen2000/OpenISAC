---
title: First OTA Run
description: Minimal launch sequence for an end-to-end over-the-air OpenISAC test.
---

## Before the First OTA Run

- Prepare two backend nodes. In downlink-only mode, BS uses one TX plus one sensing RX antenna path and UE uses one RX path. If duplex/uplink is enabled, UE also needs a TX antenna/RF chain and BS needs an RX antenna/RF chain for the uplink; in FDD mode the radios must also support the configured uplink carrier.
- Start from the sample YAML that is closest to your hardware. The repository includes X310/B210 examples, but it is not limited to those USRP models.
- Copy runtime YAMLs into `build/`, because both binaries read `BS.yaml` or `UE.yaml` from their working directory.
- If the frontend runs on another machine, point the Python viewer at the backend with `--host <backend-ip>` or the viewer's Backend IP field. `default_out_ip` is only for destination-style output IPs, not UDP/ZMQ listen addresses.
- If you care about stable real-time behavior, run `scripts/set_performance.bash`, then apply CPU isolation with `sudo ../scripts/isolate_cpus.py` (asks BS/UE/BS+UE and, when enough kernel-isolated CPUs exist, optionally configures NIC IRQ isolation/RX-TX splitting), and only then launch via `sudo ../scripts/isolate_cpus.py run ...`.

## Typical Usage Example

### 1. Startup of the BS
```bash
sudo -s
cd build
# For X310:
cp ../config/BS_X310.yaml BS.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS

# For B210:
cp ../config/BS_B210.yaml BS.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS

# For B210 duplex:
cp ../config/BS_B210_Duplex.yaml BS.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS
```
*If you are using a separate frontend computer, point the monostatic viewer at the BS backend IP with `--host` or the viewer's Backend IP field.*

### 2. Startup of the UE
```bash
sudo -s
cd build
# For X310:
cp ../config/UE_X310.yaml UE.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./UE

# For B210:
cp ../config/UE_B210.yaml UE.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./UE

# For B210 duplex:
cp ../config/UE_B210_Duplex.yaml UE.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./UE
```
*If you are using a separate frontend computer, point the bistatic viewer at the UE backend IP with `--host` or the viewer's Backend IP field. Set `default_out_ip` only for decoded/debug outputs that should be sent to another machine.*

### 3. Send video on the BS computer

Confirm that BS and UE are running and UE has synchronized with BS. Open another terminal on the BS computer. Put `video.mp4` in the current directory, or replace `video.mp4` in the command below with the actual video file path.

The command below uses NVIDIA NVENC encoding, a 60 Mb/s video target, and a 70 Mb/s UDP sending limit for an established 100 MS/s X310 link. For a first test or a configuration with a lower sample rate, change all four `60M` values to `3M` and `bitrate=70000000` to `bitrate=4000000`.

```bash
ffmpeg -re -stream_loop -1 -i video.mp4 \
  -map 0:v:0 -map '0:a:0?' \
  -c:v h264_nvenc -preset p4 -tune ll -rc cbr \
  -b:v 60M -minrate:v 60M -maxrate:v 60M -bufsize:v 60M \
  -g 30 -bf 0 -pix_fmt yuv420p -c:a aac -b:a 128k \
  -f mpegts -mpegts_flags +resend_headers -muxdelay 0 -muxpreload 0 \
  "udp://127.0.0.1:50000?pkt_size=1316&bitrate=70000000&burst_bits=10528&buffer_size=4194304"
```

`ffmpeg` sends the video and any audio to the local BS program on input port `50000`. If it runs on another computer, replace `127.0.0.1` with the BS computer's IP. Match the port to `network_output.udp_input_port` in `BS.yaml`.

For detailed video streaming settings, see [Video and UDP Workflows](/docs/tools-workflows/video-udp-workflows/).

### 4. Play video on the UE computer

Set `network_output.udp_output_ip` to `127.0.0.1` and `network_output.udp_output_port` to `50001` in `UE.yaml`. Open another terminal on the UE computer and run:

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -sync ext \
  "udp://127.0.0.1:50001?fifo_size=65536&overrun_nonfatal=1&buffer_size=16777216"
```

The UE program sends the received video data to local port `50001`, where `ffplay` receives and plays it. For playback on another computer, follow the address and port settings in [Video and UDP Workflows](/docs/tools-workflows/video-udp-workflows/).

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

## Optional UE receiver

You can also use [XR1710G](/docs/tools-workflows/xr1710g-receiver/) for UE reception. This option is still experimental and is not recommended for formal experiments.
