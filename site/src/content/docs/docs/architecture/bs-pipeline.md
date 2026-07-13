---
title: BS Runtime Pipeline
description: Main BS processing stages and thread responsibilities.
---

The BS runtime is a staged pipeline that separates payload preparation, OFDM waveform generation, USRP transmit, USRP receive, and sensing processing.

![BS software architecture](/images/SoftArchBS.png)

## Main stages

- Bit processing receives UDP payloads, fills frames, and prepares coded bits.
- OFDM transmit processing maps symbols, inserts pilots, performs IFFT, and adds cyclic prefixes.
- USRP transmit streams baseband samples to the radio.
- USRP receive captures sensing samples.
- Sensing processing demodulates received OFDM resources and produces delay-Doppler outputs.

## Design intent

The pipeline avoids mixing hardware orchestration with reusable DSP. Shared signal-processing logic belongs in headers such as `include/OFDMCore.hpp`, while runtime ownership, queues, and UHD control stay near the executable entry point.

Hot paths should avoid unbounded allocations and blocking diagnostics.
