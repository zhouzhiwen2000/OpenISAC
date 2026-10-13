---
title: 视频与 UDP 工作流
description: 通过 OFDM 链路传输 UDP 数据，并逐步搭建视频演示。
---

OpenISAC 可以通过 OFDM 通信链路传输 UDP 数据报。视频演示不需要单独的无线传输协议：编码器产生的压缩视频包可以直接进入同一条 UDP payload 路径。

## 启动顺序

建议按以下顺序逐步增加系统复杂度：

1. 先发送低速率 UDP 流量，确认同步、信道估计和 LDPC 解码稳定。
2. 逐步提高 UDP payload 速率，观察丢包、队列深度和端到端吞吐。
3. 链路稳定后，再接入视频编码器、解码器和播放器。

## 限制 UDP 视频突发

即使编码目标码率固定，关键帧也可能形成短时突发。在 FFmpeg 的 **UDP 输出 URL** 中加入 `bitrate` 与 `burst_bits`，控制进入 BS 的数据报发送节奏。URL 必须加引号，确保 shell 保留 `&`。

发送主机支持 NVIDIA NVENC 时，以下示例编码 60 Mb/s 视频，并为音频与 MPEG-TS 封装留出余量，将 UDP payload 发送上限设为 70 Mb/s：

```bash
ffmpeg -re -stream_loop -1 -i video.mp4 \
  -map 0:v:0 -map '0:a:0?' \
  -c:v h264_nvenc -preset p4 -tune ll -rc cbr \
  -b:v 60M -minrate:v 60M -maxrate:v 60M -bufsize:v 60M \
  -g 30 -bf 0 -pix_fmt yuv420p -c:a aac -b:a 128k \
  -f mpegts -mpegts_flags +resend_headers -muxdelay 0 -muxpreload 0 \
  "udp://127.0.0.1:50000?pkt_size=1316&bitrate=70000000&burst_bits=10528&buffer_size=4194304"
```

BS 的输入地址和端口以 YAML 为准。软件编码时，将 NVENC 选项替换为 `-c:v libx264 -preset veryfast -tune zerolatency`，并确认编码速度达到实时要求；UDP 发送节奏参数同样适用。

| 参数 | 含义 |
| --- | --- |
| `-b:v 60M` | 视频编码目标码率，60,000,000 bit/s。 |
| `bitrate=70000000` | UDP 发送节奏上限，单位 bit/s；数据不足时，实际发送速率可以更低。 |
| `pkt_size=1316` | UDP payload 大小，单位字节，可容纳七个 188 字节 MPEG-TS 包。 |
| `burst_bits=10528` | 突发预算，单位 bit；`1316 × 8` 约为一个数据报，改变包长时同步调整。 |
| `buffer_size=4194304` | 请求的 UDP socket 缓冲大小，单位字节，实际值受操作系统限制。 |

UDP 发送上限应高于实测的视频、音频与封装平均速率，并低于 OFDM 链路可持续的净荷吞吐。例如可测试 40 Mb/s 视频搭配 `bitrate=55000000`，或 50 Mb/s 视频搭配 `bitrate=65000000`。净荷容量需要计算调制方式、LDPC、CP、导频、包头、填充和整包装帧，不能仅由 100 MS/s 采样率确定。发送上限低于输入流平均速率会造成积压。`-re`、编码器 `-maxrate` 和 UDP URL 中的限速参数分别控制不同环节，需要一起配置。

上述 MPEG-TS-over-UDP 流可直接播放 UE 的解码输出：

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -sync ext \
  "udp://0.0.0.0:50001?fifo_size=65536&overrun_nonfatal=1&buffer_size=16777216"
```

接收端口以 YAML 为准。如果已有 relay 接收解码输出，播放器应使用其转发端口（例如 50101），避免两个接收程序绑定同一端口。这里使用原始 MPEG-TS over UDP；[首次 OTA 运行](/zh-cn/docs/getting-started/first-ota-run/) 中的 RTP/SDP 示例有对应的接收命令。测试时按接收字节数和时间计算吞吐，检查 UDP 丢包与后端队列，并分别统计播放器的解码错误和丢帧。参数定义见 [FFmpeg UDP 协议文档](https://ffmpeg.org/ffmpeg-protocols.html#udp)。

## 调试

出现卡顿或丢包时，先判断问题位于无线链路、UDP 传输还是视频应用：

- 同步不稳定时，先修无线链路。
- LDPC 解码失败时，检查信道和偏移估计。
- 解码稳定但 UDP 丢包时，检查队列、datagram 大小和应用 pacing。
- UDP 到达但视频卡顿时，检查 codec 和播放器行为。

按上述顺序排查，可以避免把编码器、播放器或应用队列的问题误认为无线链路故障。
