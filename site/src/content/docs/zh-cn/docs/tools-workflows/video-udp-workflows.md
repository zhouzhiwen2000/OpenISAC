---
title: 视频与 UDP 工作流
description: 在 BS 电脑上推送视频，在 UE 电脑上接收播放，并控制 UDP 发送速率。
---

这个演示用 OpenISAC 把视频从 BS 传到 UE。`ffmpeg` 负责读取视频文件、编码并发送数据，BS 和 UE 程序负责无线传输，`ffplay` 负责接收视频数据并播放画面和声音。

## 数据路径

1. 在 **BS 电脑**上，`ffmpeg` 把视频数据发到本机 UDP 端口 `50000`，交给 BS 程序。
2. BS 程序通过 USRP 发射无线信号，UE 程序通过另一台 USRP 接收，还原出视频数据。
3. 在 **UE 电脑**上，UE 程序把收到的数据发到本机 UDP 端口 `50001`，`ffplay` 监听这个端口并播放视频。

`50000` 和 `50001` 是示例端口。实际使用时，分别核对 `BS.yaml` 的 `network_output.udp_input_port` 和 `UE.yaml` 的 `network_output.udp_output_port`。播放器在 UE 本机运行时，将 `UE.yaml` 的 `network_output.udp_output_ip` 设为 `127.0.0.1`。

## 1. 启动 BS 和 UE

按[首次 OTA 运行](/zh-cn/docs/getting-started/first-ota-run/)启动两个后端，确认 UE 已同步到 BS，再发送视频。

下面沿用 100 MS/s X310 测试中的设置：视频目标码率为 **60 Mb/s**，UDP 发送上限为 **70 Mb/s**。使用较低采样率的配置或首次测试时，可以先把视频命令中的四个 `60M` 改为 `3M`，把 URL 中的 `bitrate=70000000` 改为 `bitrate=4000000`；播放正常后再逐步提高。

## 2. 在 BS 电脑上推送视频

在 BS 电脑上另开一个终端。把 `video.mp4` 放在当前目录，或将下方命令中的 `video.mp4` 替换为视频文件的实际路径。

BS 电脑支持 NVIDIA NVENC 硬件编码时，运行：

```bash
ffmpeg -re -stream_loop -1 -i video.mp4 \
  -map 0:v:0 -map '0:a:0?' \
  -c:v h264_nvenc -preset p4 -tune ll -rc cbr \
  -b:v 60M -minrate:v 60M -maxrate:v 60M -bufsize:v 60M \
  -g 30 -bf 0 -pix_fmt yuv420p -c:a aac -b:a 128k \
  -f mpegts -mpegts_flags +resend_headers -muxdelay 0 -muxpreload 0 \
  "udp://127.0.0.1:50000?pkt_size=1316&bitrate=70000000&burst_bits=10528&buffer_size=4194304"
```

这条命令会循环读取视频文件，将视频和文件中已有的音频打包成 MPEG-TS，再通过 UDP 发给本机 BS 程序。`127.0.0.1` 表示当前电脑，`50000` 是 BS 的视频数据输入端口。

使用 CPU 软件编码时，将 `-c:v h264_nvenc -preset p4 -tune ll -rc cbr` 换成 `-c:v libx264 -preset veryfast -tune zerolatency`。观察 `ffmpeg` 输出中的 `speed`，编码速度应能持续达到 `1x`。

## 3. 在 UE 电脑上播放视频

在 UE 电脑上另开一个终端，运行：

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -sync ext \
  "udp://127.0.0.1:50001?fifo_size=65536&overrun_nonfatal=1&buffer_size=16777216"
```

`ffplay` 从本机 `50001` 端口接收 UE 程序送出的视频数据，把压缩视频解码成画面，并播放音频。BS 上的 `ffmpeg`、两个后端和 UE 上的 `ffplay` 都需要保持运行。

如果播放器放在另一台电脑上，将 `UE.yaml` 的 `network_output.udp_output_ip` 改为那台电脑的 IP。在播放器电脑上运行同一条 `ffplay` 命令，将 URL 中的 `127.0.0.1` 改为 `0.0.0.0`，表示监听本机所有网卡。端口也要与 `network_output.udp_output_port` 一致。

## 为什么还要限制 UDP 突发

一帧视频编码完成后，`ffmpeg` 可能集中发送许多 UDP 包，较大的关键帧尤其明显。即使平均码率不高，这种短时突发也会给后端队列带来压力。

在输出 URL 中同时设置 `bitrate` 和 `burst_bits`，可以把数据包的发送节奏拉平。URL 要加引号，确保 shell 将 `&` 后面的参数也交给 `ffmpeg`。

| 参数 | 怎么理解 |
| --- | --- |
| `-b:v 60M` | 视频编码的目标码率是 60 Mb/s。音频和封装还会占用额外带宽。 |
| `-minrate:v 60M`、`-maxrate:v 60M` | 配合编码器的码率控制设置视频码率范围。 |
| `-bufsize:v 60M` | 编码器的码率控制缓冲容量为 60 Mbit。 |
| `bitrate=70000000` | 按最高约 70 Mb/s 的节奏发送 UDP 数据；数据不足时，实际发送速率更低。 |
| `pkt_size=1316` | 每个 UDP 包最多装 1316 字节数据，相当于七个 188 字节 MPEG-TS 包。 |
| `burst_bits=10528` | 一次突发约允许一个 UDP 包的数据量：`1316 × 8 = 10528` bit。改变包长时一起调整。 |
| `buffer_size=4194304` | 向操作系统申请 4 MiB 的 UDP 发送缓冲，实际大小受系统设置限制。 |

调整码率时，UDP 发送上限要给视频、音频和封装留出余量，同时不能超过无线链路可持续传输的数据量。设置过低会积压数据，设置过高又会放大突发。示例中的 60/70 Mb/s 需要结合当前设备和配置验证。

`-re` 让视频文件按播放速度读取，编码器参数控制生成多少视频数据，UDP URL 中的参数控制这些数据怎样发送。三者配合使用。100 MS/s 是射频采样率，视频能传多快还取决于调制方式、LDPC、CP、导频和装帧开销。

## 播放不顺畅时怎么检查

按数据经过的顺序检查：

1. **视频发送端**：确认文件能读取，`ffmpeg` 没有报错，`speed` 能保持 `1x`。
2. **无线链路**：确认 UE 持续同步，检查包头校验、LDPC 和 RX overflow 等日志。
3. **UDP 传输**：核对地址与端口，检查丢包计数和队列是否积压。实际接收速率可用“收到的字节数 × 8 ÷ 时间”计算。
4. **播放器**：分别查看视频解码错误、播放器丢帧和音画时间差。`-framedrop` 允许播放器丢弃来不及显示的帧，以减小播放延迟；这项统计与无线数据包丢失不同。

UDP 参数的完整定义见 [FFmpeg UDP 协议文档](https://ffmpeg.org/ffmpeg-protocols.html#udp)。
