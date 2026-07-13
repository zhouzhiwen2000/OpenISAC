---
title: System Architecture
description: OpenISAC bidirectional communication, multi-channel sensing, and end-to-end data flow.
---

OpenISAC is a host-based real-time OFDM integrated sensing and communication platform. The current system consists of one base station (BS) and one user equipment (UE), with one or more USRPs selected according to the required number of transmit and receive channels. Real-time communication and sensing run in the C++ backend, while results, status, and controls are exposed through a Python environment.

![OpenISAC bidirectional communication and multi-channel sensing architecture](/images/SysArch_Duplex.png)

## Supported functions

The current implementation supports four data flows:

- **Downlink communication:** the BS sends data to the UE, which performs synchronization, channel equalization, and payload decoding.
- **Uplink communication:** the UE encodes and OFDM-modulates input payloads, while the BS receives, equalizes, and decodes UE-to-BS data.
- **BS multi-channel monostatic sensing:** the BS captures multiple echo channels in parallel and publishes time-aligned sensing data; this function does not require the UE to be active.
- **UE bistatic sensing:** the UE reuses the downlink signal for both communication reception and bistatic sensing of dynamic scatterers along BS-target-UE propagation paths.

The uplink can operate in TDD or FDD. TDD reserves a contiguous OFDM-symbol window on the downlink carrier and uses guard symbols for the transmit/receive transition. FDD uses a separate uplink carrier and transmits continuous full OFDM frames. Both modes use the same OFDM parameter framework for consistent waveform generation and reception.

## BS node

The BS concurrently handles downlink transmission, multi-channel monostatic sensing, and uplink reception. The downlink path accepts payloads and performs channel coding, OFDM resource mapping, and waveform generation. One or more sensing receive paths capture echoes and use the known transmit grid to form per-channel time-frequency and delay-Doppler products. The uplink path receives the UE waveform and performs synchronization, channel estimation, equalization, and payload decoding.

Each sensing path may select its own USRP device or channel, gain, antenna port, and sample-alignment offset. The backend matches channels using a common frame-start symbol index and aggregates them into one synchronized stream with channel identifiers, allowing the frontend to compare or jointly process all receive channels at the same instant.

## UE node

The UE runs downlink reception, bistatic sensing, and uplink transmission as concurrent pipelines. Downlink reception establishes frame synchronization and estimates carrier-frequency offset, sampling-clock mismatch, and channel state. Equalized symbols feed communication decoding and reconstruct the transmit reference used for bistatic sensing. A separate uplink path accepts local UDP payloads, performs LDPC encoding and OFDM modulation, and passes the waveform to the USRP transmitter.

Because the BS and UE use independent local clocks, the UE continuously tracks frequency and timing drift. OpenISAC supports software-only OTA synchronization and sensing-side timing compensation, as well as external clock references or optional hardware clock trimming for improved long-term communication and bistatic-sensing coherence.

## End-to-end data flow

A complete data flow is:

1. A downlink payload enters the BS, is encoded and OFDM-modulated, and is transmitted to the UE.
2. Multiple BS sensing channels capture target echoes in sync, while the UE uses the downlink reception for both payload decoding and bistatic sensing.
3. An uplink payload enters the UE, passes through an independent transmit pipeline, and is decoded by the BS uplink receive pipeline.
4. The BS publishes aggregated, time-aligned multi-channel monostatic data; the UE publishes bistatic sensing data; runtime status and optional intermediate results are delivered to the Python interaction layer.

In TDD, downlink and uplink alternate within the frame. In FDD, they can run concurrently on separate carriers. In both cases, multi-channel sensing, communication coding/decoding, and frontend publication advance in independent pipelines so one processing path does not block the other real-time tasks. See the [BS runtime pipeline](/docs/architecture/bs-pipeline/) and [UE runtime pipeline](/docs/architecture/ue-pipeline/) for the node-level data flows.
