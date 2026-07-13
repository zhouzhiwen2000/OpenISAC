---
title: First OTA Run
description: Minimal launch sequence for an end-to-end over-the-air OpenISAC test.
---

## Before the First OTA Run

- Prepare two backend nodes. In downlink-only mode, BS uses one TX plus one sensing RX antenna path and UE uses one RX path. If duplex/uplink is enabled, UE also needs a TX antenna/RF chain and BS needs an RX antenna/RF chain for the uplink; in FDD mode the radios must also support the configured uplink carrier.
- Start from the sample YAML that is closest to your hardware. The repository includes X310/B210 examples, but it is not limited to those USRP models.
- Copy runtime YAMLs into `build/`, because both binaries read `BS.yaml` or `UE.yaml` from their working directory.
- If the frontend runs on another machine, point the Python viewer at the backend with `--host <backend-ip>` or the viewer's Backend IP field. `default_out_ip` is only for destination-style output IPs, not UDP/ZMQ listen addresses.
- If you care about stable real-time behavior, run `scripts/set_performance.bash`, then apply CPU isolation with `sudo ../scripts/isolate_cpus.bash` (or a custom CPU set), and only then launch via `sudo ../scripts/isolate_cpus.bash run ...`.

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
