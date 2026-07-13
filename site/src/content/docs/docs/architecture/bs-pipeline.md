---
title: BS Runtime Pipeline
description: BS data flow for downlink transmission, multi-channel monostatic sensing, and uplink reception.
---

The BS runtime contains three concurrent pipeline groups: downlink transmission, multi-channel monostatic sensing, and uplink reception. FIFO queues connect the stages so communication coding/decoding, waveform processing, and multiple radio streams can advance at the same time.

![BS bidirectional communication and multi-channel sensing software architecture](/images/SoftArchBSDuplex.png)

## From payload to radio transmission

1. The BS receives communication payloads over UDP and performs frame filling and channel coding.
2. The OFDM modulator maps synchronization, pilot, and data symbols onto a time-frequency resource grid, then applies IFFT and cyclic-prefix insertion to form the time-domain waveform.
3. The waveform enters the transmit queue and is sent by the USRP with continuous timing.
4. The modulator also retains the corresponding frequency-domain transmit grid as the known reference for monostatic sensing.

When communication payloads are temporarily unavailable, padding symbols preserve continuous frame timing. With TDD uplink enabled, the modulator leaves the configured uplink window and guard symbols free of downlink data. In FDD, the downlink continues to occupy complete frames on its own carrier.

## Multi-channel monostatic sensing

1. Every enabled sensing receive channel captures echoes and owns an independent receive and processing pipeline.
2. Each channel pairs its receive frame with the transmit grid from the same frame, then performs cyclic-prefix removal, FFT, and communication-symbol cancellation.
3. Each path produces its own time-frequency channel or delay-Doppler data with a common frame-start symbol index.
4. The aggregator waits for all channels at that instant and publishes one multi-channel packet in logical-channel order.

Each sensing path may independently select a device, USRP channel, gain, antenna port, and sample-alignment offset. Backend processing supports live results, while dense or compact data output, configurable stride, and Python bypass provide tradeoffs among transport bandwidth, update load, and algorithm flexibility.

## UE-to-BS uplink reception

1. The BS uplink receive path captures the UE waveform according to the duplex mode: only the in-frame uplink window in TDD, or continuous full frames on a separate carrier in FDD.
2. Uplink demodulation performs synchronization, FFT, channel estimation, pilot tracking, and equalization to produce QPSK soft information.
3. LDPC decoding recovers UE payloads and forwards them to BS-side applications over UDP.

Uplink reception runs independently of downlink transmission and sensing reception. Optional channel-estimate, delay-spectrum, and constellation streams expose uplink state for live observation.

## Why a concurrent pipeline

OpenISAC separates downlink transmission, multiple sensing channels, and uplink reception into pipelines that progress concurrently. No sensing computation or communication decode path needs to block another radio stream, increasing aggregate throughput when bidirectional communication and multi-channel sensing run together.

Queues absorb short-term rate variation between stages, while multi-channel aggregation ensures the frontend receives a coherent set of channels for each instant. USRP/UHD control remains separate from DSP processing, allowing channels to share a device or use separate USRPs while preserving the same OFDM and sensing algorithms.
