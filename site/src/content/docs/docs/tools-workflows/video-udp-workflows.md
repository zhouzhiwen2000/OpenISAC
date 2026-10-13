---
title: Video and UDP Workflows
description: Payload workflows for UDP traffic and video streaming demonstrations.
---

OpenISAC can carry UDP payloads over the OFDM communication link. Video demos usually feed compressed packets into the same payload path.

## Bring-up order

First validate the PHY with low-rate traffic. Then increase UDP payload rate. Finally add video encoding or decoding tools after link stability is confirmed.

## Pace UDP video bursts

Video keyframes can produce short bursts even with a constant target encoding
rate. Add `bitrate` and `burst_bits` to the FFmpeg **UDP output URL** to pace
datagrams entering the BS. Keep the URL in quotes so the shell preserves `&`.

For a sender with NVIDIA NVENC, this example encodes 60 Mb/s video and allows
up to 70 Mb/s of UDP payload, including audio and MPEG-TS overhead:

```bash
ffmpeg -re -stream_loop -1 -i video.mp4 \
  -map 0:v:0 -map '0:a:0?' \
  -c:v h264_nvenc -preset p4 -tune ll -rc cbr \
  -b:v 60M -minrate:v 60M -maxrate:v 60M -bufsize:v 60M \
  -g 30 -bf 0 -pix_fmt yuv420p -c:a aac -b:a 128k \
  -f mpegts -mpegts_flags +resend_headers -muxdelay 0 -muxpreload 0 \
  "udp://127.0.0.1:50000?pkt_size=1316&bitrate=70000000&burst_bits=10528&buffer_size=4194304"
```

Use the BS input address and port from YAML. For software encoding, replace
the NVENC options with `-c:v libx264 -preset veryfast -tune zerolatency` and
verify that encoding keeps up with real time. The UDP pacing options also
apply to that encoder.

| Setting | Meaning |
| --- | --- |
| `-b:v 60M` | Target video encoding rate: 60,000,000 bits/s. |
| `bitrate=70000000` | UDP output pacing rate in bits/s; actual output can be lower when less data is available. |
| `pkt_size=1316` | UDP payload size in bytes, holding seven 188-byte MPEG-TS packets. |
| `burst_bits=10528` | Burst budget in bits: `1316 × 8`, approximately one datagram. Update it when changing packet size. |
| `buffer_size=4194304` | Requested UDP socket buffer in bytes, subject to OS limits. |

Choose the UDP pacing rate above the measured average video/audio/container
rate and below the OFDM link's sustainable payload throughput. For example,
40 Mb/s video with `bitrate=55000000` or 50 Mb/s video with
`bitrate=65000000` are settings to test. A 100 MS/s RF sample rate alone does
not determine payload capacity; account for modulation, LDPC, CP, pilots,
packet headers, padding, and complete-packet packing. A pacing rate below the
average incoming stream causes backlog. `-re` and encoder `-maxrate` control
different parts of this path, so also configure the UDP URL.

For this MPEG-TS-over-UDP stream, play the UE's decoded output directly:

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -sync ext \
  "udp://0.0.0.0:50001?fifo_size=65536&overrun_nonfatal=1&buffer_size=16777216"
```

Match the receiver port to YAML. If a relay already receives that decoded
output, use its player port (for example, 50101) instead of binding a second
receiver to the same port. This command uses raw MPEG-TS over UDP;
the RTP/SDP examples in [First OTA Run](/docs/getting-started/first-ota-run/)
have their own matching receiver commands. Measure received bytes over time,
check UDP drops and backend queues, and inspect player decode errors and
frame drops separately. Option definitions are in the
[FFmpeg UDP protocol documentation](https://ffmpeg.org/ffmpeg-protocols.html#udp).

## Debugging

Separate PHY failures from application failures:

- If synchronization is unstable, fix the radio path first.
- If LDPC decode fails, inspect channel and offset estimates.
- If UDP drops occur with stable decode, inspect queues, datagram size, and application pacing.
- If video freezes but UDP arrives, inspect codec and player behavior.

This ordering prevents frontend or application symptoms from being mistaken for radio problems.
