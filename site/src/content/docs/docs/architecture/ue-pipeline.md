---
title: UE Runtime Pipeline
description: Main UE processing stages for synchronization, demodulation, and decoding.
---

The UE runtime receives downlink samples, finds frame timing, estimates carrier and sampling offsets, demodulates OFDM symbols, and decodes payload data.

![UE software architecture](/images/SoftArchUE.png)

## Main stages

- USRP receive captures the downlink stream.
- Synchronization searches for the known preamble or sync resources.
- CFO/SFO tracking keeps the demodulator aligned over time.
- Channel estimation and equalization prepare data symbols for demapping.
- LDPC decoding reconstructs payload bits.
- Optional frontend publication exposes status and sensing data.

## Operational behavior

The UE should be started before the BS in most OTA tests. During bring-up, first confirm that synchronization is stable, then inspect decode results and only then tune higher-level workflows such as UDP or video traffic.
