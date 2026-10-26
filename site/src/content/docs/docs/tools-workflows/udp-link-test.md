---
title: UDP Link Test
description: Measure UDP throughput, packet loss and bit errors with a live dashboard on the BS computer.
---

`UdpLinkTest` sends numbered UDP datagrams with known contents. The BS passes
them over the radio link, and a second instance on the UE checks the recovered
data. The dashboard runs on the **BS computer** and shows both ends together.

## Build on both computers

From the OpenISAC directory:

```bash
cmake -S . -B build
cmake --build build --target UdpLinkTest -j4
```

The dashboard uses the project's Python environment with PyQt6, pyqtgraph,
NumPy and pyzmq installed.

## Start the test

Start the BS and UE backends with matching radio settings. Stop video sending,
video players and any relay listening on the UE output port before starting
this test. The examples use BS input `127.0.0.1:50000` and UE output
`127.0.0.1:50001`; match these to the backend YAML files.

**On the UE computer**, start the receiver first. Replace `192.168.50.1` with
the BS computer's address on the management network:

```bash
./build/UdpLinkTest rx --port 50001 \
  --telemetry tcp://0.0.0.0:18101 --control tcp://0.0.0.0:18103 \
  --sender-telemetry tcp://192.168.50.1:18100
```

**On the BS computer**, start a 50 Mbps stream:

```bash
./build/UdpLinkTest tx --port 50000 --rate-mbps 50 --packet-bytes 1316 \
  --telemetry tcp://0.0.0.0:18100
```

**In another BS terminal**, open the dashboard. Replace `192.168.50.19` with
the UE computer's management address:

```bash
python scripts/plot_udp_link.py \
  --tx tcp://127.0.0.1:18100 --rx tcp://192.168.50.19:18101 \
  --control tcp://127.0.0.1:18102 --rx-control tcp://192.168.50.19:18103 \
  --code-rate 1/2 --rate-mbps 50
```

Change **目标速率** and click **应用** to change the sender's target rate.
Set it to zero to pause traffic. Enter a fraction such as `1/2` or `2/3` in
**LDPC 码率** and press Enter to update the encoded-rate display. This field
only affects the displayed rates. Match it to `ldpc.code_rate` in both YAML files. To change the actual payload code rate, edit YAML or the configuration editor and restart both backends.

Click **清除** beside the packet-loss value to restart its statistics from the current test progress.
The **清除** button beside BER resets both BER and the accumulated bit-error count.
The two buttons act independently and leave traffic, cumulative received data and the flow chart running.

Use Ctrl+C to stop the sender, then allow at least one second for the receiver
to confirm the last packet counts before stopping it. Closing the dashboard
leaves the sender and receiver running. Restart both tester processes for a
new test session.

## Statistics

- **Sending rate:** bytes successfully submitted to the BS UDP input per
  second, including the tester's packet header and payload; IP, UDP and Ethernet
  headers are excluded.
- **Receiving rate:** actual received datagram bytes per second. Packets with
  bit errors are included. Identifiable duplicates, packets from other sessions
  and packets older than the retained sequence window are excluded. Datagrams
  with an unrecoverable test header still count toward received throughput.
- **LDPC encoded rate:** displayed UDP rate divided by the user-entered code
  rate. It excludes additional air-interface framing, padding, pilots and CP.
- **Packet loss:** missing identifiable sequence numbers divided by the
  confirmed sent count in the current statistics interval. Counting begins with
  the first identifiable RX packet from this test; earlier packets are excluded.
  A one-second grace interval allows for transport delay and reordering. After
  counting starts, sender telemetry also accounts for trailing losses and complete
  outages. Late packets within the retained window correct loss only in the
  interval to which they belong.
- **BER:** differing bits divided by compared bits in received, identifiable,
  nonduplicate datagrams. It includes bit errors in recoverable packet headers;
  lost packets are not counted as bit errors. Unrecoverable headers cannot be
  compared and remain missing from the sequence statistics.

The JSON telemetry also includes duplicate, reordered, late, invalid-header and
send-error counts. If sender telemetry is unavailable, loss can only be inferred
through the highest received sequence number; trailing and total-outage losses
then remain unknown. Keep the management network connected during a test.

The default packet size is 1316 bytes. Use `--help` for packet size, finite
packet-count, duration and telemetry options. The data path is UDP; telemetry
and controls use ZeroMQ over the management network.
