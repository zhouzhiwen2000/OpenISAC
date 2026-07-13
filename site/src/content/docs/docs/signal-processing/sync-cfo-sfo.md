---
title: Sync, CFO, and SFO
description: Timing and frequency alignment concepts in the UE and sensing paths.
---

Synchronization aligns the receiver to frame boundaries. Carrier frequency offset (CFO) appears as phase rotation across time. Sampling frequency offset (SFO) appears as gradual timing drift and subcarrier-dependent phase behavior.

## Timing

The receiver searches for known structure in the stream and selects a frame start. A wrong start position corrupts both payload decoding and delay-domain sensing.

## CFO

Residual CFO rotates OFDM symbols over time. If not corrected, it reduces coherent accumulation and degrades demapping.

## SFO

SFO causes the receiver sampling clock to drift relative to the transmitter. It is especially visible in longer runs, bistatic sensing, and high-bandwidth operation.

OpenISAC exposes timing and offset controls through runtime configuration and dedicated scripts so experiments can separate radio impairment, estimator behavior, and sensing interpretation.
