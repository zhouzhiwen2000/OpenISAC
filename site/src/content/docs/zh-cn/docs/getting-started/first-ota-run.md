---
title: 首次 OTA 运行
description: 端到端空口 OpenISAC 测试的最小启动流程。
---

## 首次 OTA 运行前检查

- 准备两台后端节点。仅下行模式下，BS 端需要一路 TX 和一路感知 RX 天线链路，UE 端需要一路 RX 链路。若开启双工/上行，UE 端还必须具备一路 TX 天线/RF 链路，BS 端也必须具备用于接收上行的一路 RX 天线/RF 链路；FDD 模式还要求射频设备支持配置的上行载波。
- 先选一份最接近你硬件的 YAML 模板。仓库自带 X310/B210 示例，但项目并不限于这两种 USRP。
- 运行时 YAML 需要放在 `build/` 目录，因为两个二进制程序都会从当前工作目录读取 `BS.yaml` 或 `UE.yaml`。
- 如果前端跑在另一台机器上，请在 Python 前端中用 `--host <后端IP>` 或 Backend IP 输入框指向后端。`default_out_ip` 只用于目的输出 IP，不用于 UDP/ZMQ 监听地址。
- 如果你关心实时稳定性，请先执行 `scripts/set_performance.bash`，然后用 `sudo ../scripts/isolate_cpus.py` 做 CPU 隔离（会询问 BS/UE/BS+UE；内核隔离核足够时还可配置网卡 IRQ 隔离和 RX/TX 分流），再用 `sudo ../scripts/isolate_cpus.py run ...` 启动程序。

## 典型使用示例

### 1. 启动 BS (基站)
```bash
sudo -s
cd build
# 对于 X310:
cp ../config/BS_X310.yaml BS.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS

# 对于 B210:
cp ../config/BS_B210.yaml BS.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS

# 对于 B210 双工:
cp ../config/BS_B210_Duplex.yaml BS.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./BS
```
*如果您使用单独的计算机作为前端，请在单站感知前端中用 `--host` 或 Backend IP 输入框指向 BS 后端 IP。*

### 2. 启动 UE (用户端)
```bash
sudo -s
cd build
# 对于 X310:
cp ../config/UE_X310.yaml UE.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./UE

# 对于 B210:
cp ../config/UE_B210.yaml UE.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./UE

# 对于 B210 双工:
cp ../config/UE_B210_Duplex.yaml UE.yaml
sudo ../scripts/isolate_cpus.py
sudo ../scripts/isolate_cpus.py run ./UE
```
*如果您使用单独的计算机作为前端，请在双站感知前端中用 `--host` 或 Backend IP 输入框指向 UE 后端 IP。只有解码/调试类输出需要发往另一台机器时才设置 `default_out_ip`。*

### 3. 在 BS 电脑上推送视频

确认 BS 和 UE 都已启动，UE 已同步到 BS。在 BS 电脑上另开一个终端，把 `video.mp4` 放在当前目录，或将下方命令中的 `video.mp4` 替换为视频文件的实际路径。

下面的命令使用 NVIDIA NVENC 编码，视频目标码率为 60 Mb/s，UDP 发送上限为 70 Mb/s，适用于已调通的 100 MS/s X310 链路。首次测试或使用较低采样率时，先将四个 `60M` 改为 `3M`，将 `bitrate=70000000` 改为 `bitrate=4000000`。

```bash
ffmpeg -re -stream_loop -1 -i video.mp4 \
  -map 0:v:0 -map '0:a:0?' \
  -c:v h264_nvenc -preset p4 -tune ll -rc cbr \
  -b:v 60M -minrate:v 60M -maxrate:v 60M -bufsize:v 60M \
  -g 30 -bf 0 -pix_fmt yuv420p -c:a aac -b:a 128k \
  -f mpegts -mpegts_flags +resend_headers -muxdelay 0 -muxpreload 0 \
  "udp://127.0.0.1:50000?pkt_size=1316&bitrate=70000000&burst_bits=10528&buffer_size=4194304"
```

`ffmpeg` 会把视频和文件中已有的音频发送给本机 BS 程序，输入端口为 `50000`。如果 `ffmpeg` 在另一台电脑上运行，请将 `127.0.0.1` 改为 BS 电脑的 IP。端口应与 `BS.yaml` 的 `network_output.udp_input_port` 一致。

需要视频推流配置的详细信息，请参考[视频与 UDP 工作流](/zh-cn/docs/tools-workflows/video-udp-workflows/)。

### 4. 在 UE 电脑上播放视频

确认 `UE.yaml` 的 `network_output.udp_output_ip` 为 `127.0.0.1`，`network_output.udp_output_port` 为 `50001`。在 UE 电脑上另开一个终端，运行：

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -sync ext \
  "udp://127.0.0.1:50001?fifo_size=65536&overrun_nonfatal=1&buffer_size=16777216"
```

UE 程序将收到的视频数据发到本机 `50001` 端口，`ffplay` 从这里接收并播放。播放器在另一台电脑上运行时，地址和端口的设置方法见[视频与 UDP 工作流](/zh-cn/docs/tools-workflows/video-udp-workflows/)。

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

## 可选的 UE 接收设备

UE 接收也可以选择 [XR1710G](/zh-cn/docs/tools-workflows/xr1710g-receiver/)。目前仍处于测试阶段，不建议用于正式实验。
