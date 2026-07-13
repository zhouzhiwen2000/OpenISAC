---
title: 首次 OTA 运行
description: 端到端空口 OpenISAC 测试的最小启动流程。
---

## 首次 OTA 运行前检查

- 准备两台后端节点。仅下行模式下，BS 端需要一路 TX 和一路感知 RX 天线链路，UE 端需要一路 RX 链路。若开启双工/上行，UE 端还必须具备一路 TX 天线/RF 链路，BS 端也必须具备用于接收上行的一路 RX 天线/RF 链路；FDD 模式还要求射频设备支持配置的上行载波。
- 先选一份最接近你硬件的 YAML 模板。仓库自带 X310/B210 示例，但项目并不限于这两种 USRP。
- 运行时 YAML 需要放在 `build/` 目录，因为两个二进制程序都会从当前工作目录读取 `BS.yaml` 或 `UE.yaml`。
- 如果前端跑在另一台机器上，请在 Python 前端中用 `--host <后端IP>` 或 Backend IP 输入框指向后端。`default_out_ip` 只用于目的输出 IP，不用于 UDP/ZMQ 监听地址。
- 如果你关心实时稳定性，请先执行 `scripts/set_performance.bash`，然后先用 `sudo ../scripts/isolate_cpus.bash` 做 CPU 隔离，再用 `sudo ../scripts/isolate_cpus.bash run ...` 启动程序。

## 典型使用示例

### 1. 启动 BS (基站)
```bash
sudo -s
cd build
# 对于 X310:
cp ../config/BS_X310.yaml BS.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./BS

# 对于 B210:
cp ../config/BS_B210.yaml BS.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./BS

# 对于 B210 双工:
cp ../config/BS_B210_Duplex.yaml BS.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./BS
```
*如果您使用单独的计算机作为前端，请在单站感知前端中用 `--host` 或 Backend IP 输入框指向 BS 后端 IP。*

### 2. 启动 UE (用户端)
```bash
sudo -s
cd build
# 对于 X310:
cp ../config/UE_X310.yaml UE.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./UE

# 对于 B210:
cp ../config/UE_B210.yaml UE.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./UE

# 对于 B210 双工:
cp ../config/UE_B210_Duplex.yaml UE.yaml
sudo ../scripts/isolate_cpus.bash
sudo ../scripts/isolate_cpus.bash run ./UE
```
*如果您使用单独的计算机作为前端，请在双站感知前端中用 `--host` 或 Backend IP 输入框指向 UE 后端 IP。只有解码/调试类输出需要发往另一台机器时才设置 `default_out_ip`。*

### 3. 将视频流传输到 BS
```bash
ffmpeg -re -stream_loop -1 -fflags +genpts -i video.mp4 -an -c:v libx264 -x264-params keyint=5:min-keyint=1 -b:v 3000k -minrate 3000k -maxrate 3000k -bufsize 1M -f rtp -sdp_file video.sdp "rtp://<your IP of the BS>:50000"
```
*如果您在BS本地传输视频流，BS 的 IP 可以设置为 127.0.0.1。*

如果您希望连同音频一起传输，可以改用 RTP MPEG-TS：
```bash
ffmpeg -re -stream_loop -1 -fflags +genpts -i video.mp4 -c:v libx264 -x264-params keyint=5:min-keyint=1 -b:v 30000k -minrate 30000k -maxrate 30000k -bufsize 1M -c:a aac -b:a 128k -ar 48000 -ac 2 -f rtp_mpegts "rtp://<your IP of the BS>:50000"
```

### 4. 从 UE 播放视频
将 `video.sdp` 复制到视频接收端，将 `m=video 50000 RTP/AVP 96` 修改为 `m=video 50001 RTP/AVP 96`。
```bash
ffplay -protocol_whitelist file,rtp,udp -i video1.sdp
```
*注意：此命令应在前端运行。*

对于带音频的 RTP MPEG-TS 流，则可以不依赖 SDP 文件，直接播放：
```bash
ffplay rtp://0.0.0.0:50001
```

### 5. 运行单站感知前端
```bash
python3 ./scripts/plot_sensing_fast.py
```
这是当前维护的统一前端，会自动选择 CUDA、MLX、Intel GPU 或 CPU 后端。

### 6. 运行双站感知前端
```bash
python3 ./scripts/plot_bi_sensing_fast.py
```
这是当前维护的统一前端，会自动选择 CUDA、MLX、Intel GPU 或 CPU 后端。
