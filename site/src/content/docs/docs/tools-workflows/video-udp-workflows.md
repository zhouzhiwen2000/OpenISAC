---
title: Video and UDP Workflows
description: Send video from the BS computer, play it on the UE computer, and control UDP transmission rate.
---

This demonstration sends video from BS to UE through OpenISAC. `ffmpeg` reads
and encodes the video file, the BS and UE programs carry the data over the
radio link, and `ffplay` receives it and plays the picture and sound.

## Data path

1. On the **BS computer**, `ffmpeg` sends video data to local UDP port `50000`, where the BS program receives it.
2. BS transmits through its USRP. The UE program receives the signal through the other USRP and recovers the video data.
3. On the **UE computer**, the UE program sends that data to local UDP port `50001`. `ffplay` listens on this port and plays the video.

These ports are examples. Check `network_output.udp_input_port` in `BS.yaml`
and `network_output.udp_output_port` in `UE.yaml`. When the player runs on the
UE computer, set `network_output.udp_output_ip` in `UE.yaml` to `127.0.0.1`.

## 1. Start BS and UE

Follow [First OTA Run](/docs/getting-started/first-ota-run/) to start both
backends. Confirm that UE has synchronized with BS before sending video.

The commands below use the settings from a 100 MS/s X310 test: a **60 Mb/s**
video target and a **70 Mb/s** UDP sending limit. For a first test or a
configuration with a lower sample rate, start by changing all four `60M`
values to `3M` and `bitrate=70000000` to `bitrate=4000000`. Increase them
gradually after playback works.

## 2. Send video on the BS computer

Open another terminal on the BS computer. Put `video.mp4` in the current
directory, or replace `video.mp4` in the command below with the actual video file path.

If the BS computer supports NVIDIA NVENC hardware encoding, run:

```bash
ffmpeg -re -stream_loop -1 -i video.mp4 \
  -map 0:v:0 -map '0:a:0?' \
  -c:v h264_nvenc -preset p4 -tune ll -rc cbr \
  -b:v 60M -minrate:v 60M -maxrate:v 60M -bufsize:v 60M \
  -g 30 -bf 0 -pix_fmt yuv420p -c:a aac -b:a 128k \
  -f mpegts -mpegts_flags +resend_headers -muxdelay 0 -muxpreload 0 \
  "udp://127.0.0.1:50000?pkt_size=1316&bitrate=70000000&burst_bits=10528&buffer_size=4194304"
```

The command loops the file, packages the video and any audio as MPEG-TS, and
sends the data over UDP to the local BS program. `127.0.0.1` means this
computer; `50000` is the BS video-data input port.

For CPU software encoding, replace
`-c:v h264_nvenc -preset p4 -tune ll -rc cbr` with
`-c:v libx264 -preset veryfast -tune zerolatency`. Check the `speed` reported
by `ffmpeg`: encoding should sustain `1x`.

## 3. Play video on the UE computer

Open another terminal on the UE computer and run:

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -sync ext \
  "udp://127.0.0.1:50001?fifo_size=65536&overrun_nonfatal=1&buffer_size=16777216"
```

`ffplay` receives the video data sent by the UE program to local port `50001`,
decodes the compressed video into pictures, and plays the audio. Keep
`ffmpeg` on BS, both backends, and `ffplay` on UE running.

If the player runs on another computer, set
`network_output.udp_output_ip` in `UE.yaml` to that computer's IP. On the
player computer, run the same `ffplay` command with `127.0.0.1` replaced
by `0.0.0.0` to listen on all local network interfaces. Match the port to
`network_output.udp_output_port` as well.

## Why limit UDP bursts too?

After encoding a frame, `ffmpeg` may send many UDP packets together. Large
keyframes make this more noticeable. These short bursts can put pressure
on backend queues even when the average rate is moderate.

Set both `bitrate` and `burst_bits` in the output URL to spread packet
transmissions over time. Quote the URL so the shell passes the parameters
after `&` to `ffmpeg` too.

| Setting | What it controls |
| --- | --- |
| `-b:v 60M` | A 60 Mb/s target for video encoding. Audio and packaging need additional bandwidth. |
| `-minrate:v 60M`, `-maxrate:v 60M` | The video rate range used by the encoder's rate control. |
| `-bufsize:v 60M` | A 60 Mbit encoder rate-control buffer. |
| `bitrate=70000000` | UDP sending paced at up to about 70 Mb/s; output is lower when less data is available. |
| `pkt_size=1316` | Up to 1316 data bytes per UDP packet: seven 188-byte MPEG-TS packets. |
| `burst_bits=10528` | About one UDP packet's worth of data per burst: `1316 × 8 = 10528` bits. Adjust it with packet size. |
| `buffer_size=4194304` | A request for a 4 MiB UDP send buffer, subject to OS settings. |

Leave room for video, audio, and packaging when choosing the UDP limit, while
keeping it within the radio link's sustainable data throughput. Too low a
limit causes backlog; too high a limit allows larger bursts. Validate the
example's 60/70 Mb/s settings with your own device and configuration.

`-re` reads the file at playback speed, the encoder options control how much
video data is produced, and the UDP URL controls how it is sent. Use them
together. A 100 MS/s RF sample rate is only one factor in video capacity;
modulation, LDPC, CP, pilots, and frame overhead also matter.

## Check problems in transmission order

1. **Video sender:** confirm that the file opens, `ffmpeg` reports no errors, and `speed` stays at `1x`.
2. **Radio link:** confirm continuous UE synchronization and check header validation, LDPC, and RX overflow logs.
3. **UDP transfer:** check addresses and ports, packet-drop counters, and queue buildup. Calculate received throughput as received bytes × 8 ÷ elapsed time.
4. **Player:** inspect video decode errors, player frame drops, and audio/video time difference separately. `-framedrop` lets the player skip frames that cannot be displayed in time to reduce playback delay; that count differs from radio packet loss.

For complete option definitions, see the
[FFmpeg UDP protocol documentation](https://ffmpeg.org/ffmpeg-protocols.html#udp).
