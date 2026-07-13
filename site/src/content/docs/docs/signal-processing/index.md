---
title: Signal Processing Overview
description: A unified chain from bidirectional OFDM transmission to multichannel monostatic and bistatic sensing.
---

OpenISAC unifies four processing functions under one OFDM signal model: BS-to-UE downlink communication, UE-to-BS uplink communication, multichannel monostatic sensing at the BS, and bistatic sensing at the UE. They share the same frame, subcarrier, and reference-signal definitions, so timing, frequency-offset, and channel estimates obtained for communication can also support sensing.

## End-to-End Chain

1. The BS and UE map synchronization sequences, pilots, and coded QPSK symbols onto separate downlink and uplink resource grids.
2. IFFT and cyclic-prefix insertion convert each grid into a continuous OFDM waveform. TDD partitions one frame into downlink, guard, and uplink symbols; FDD transmits continuously on two carriers.
3. The UE uses the downlink synchronization fields to acquire and correct its initial timing and frequency offsets relative to the BS. In normal operation, the downlink and uplink estimate their own channels and use the same residual-CFO/SFO estimation and compensation form on their respective references.
4. Each communication receiver equalizes its data resources, computes QPSK soft information, and applies LDPC decoding.
5. The BS divides each sensing channel's received grid by the known downlink grid, producing a multichannel time-frequency channel tensor for range–Doppler–angle or micro-Doppler processing.
6. The UE reconstructs unknown downlink data symbols before forming its bistatic channel grid. Continuous timing-offset and SFO compensation keep bistatic delay-Doppler and micro-Doppler sensing stable over long observations.
7. When both directions are enabled, eRTM uses uplink/downlink channel reciprocity to estimate the timing offsets at the BS and UE.

## Shared Notation

| Symbol | Meaning |
|---|---|
| $x\in\{\mathrm{DL},\mathrm{UL}\}$ | Downlink or uplink |
| $\gamma$ | Continuous OFDM frame index |
| $m$ | OFDM symbol index within a frame |
| $n$ | FFT index; $\kappa_n$ is the subcarrier index corresponding to that FFT index |
| $r=0,\ldots,R-1$ | BS sensing-array channel index |
| $\boldsymbol B_\gamma^x$ | Transmit resource grid for link $x$ |
| $\boldsymbol Y_\gamma^x$ | Received resource grid after the FFT |
| $\boldsymbol H_\gamma^x$ | Communication-channel frequency response |
| $\boldsymbol F_\gamma$ | Sensing channel symbols after modulation removal |
| $\tau_{l,\mathrm{prop}}^x(t)$ | True wireless propagation delay of path $l$ on link $x$ |
| $\tau_x^\mathrm{RF}$ | Fixed combined transmit/receive RF group delay of link $x$ |
| $\tau_d^q(t)$ | Time-varying offset of receiver $q$'s current demodulation window relative to the transmitter frame boundary |
| $\tau_\mathrm{TO}^q(t)$ | Common displacement from true propagation delay to the path delay observed locally at endpoint $q\in\{\mathrm{BS},\mathrm{UE}\}$ |
| $\tau_\mathrm{TO}^{\mathrm{BS-UE}}(t)$ | $\tau_\mathrm{TO}^\mathrm{BS}(t)-\tau_\mathrm{TO}^\mathrm{UE}(t)$ |

Italic lower-case, bold lower-case, and bold upper-case symbols denote scalars, vectors, and matrices. $(\cdot)^T$, $(\cdot)^H$, and $(\cdot)^*$ denote transpose, Hermitian transpose, and complex conjugation.

## Reading Order

- [Signal Model](/docs/signal-processing/signal-model/) defines the downlink, uplink, multichannel monostatic, and bistatic propagation models.
- [OFDM Resources](/docs/signal-processing/ofdm-resources/) defines the continuous waveform, TDD/FDD sets, and reference signals.
- [Initial Synchronization](/docs/signal-processing/initial-synchronization/) explains how the UE acquires and corrects initial timing and frequency offsets from the downlink fields.
- [Downlink Communication](/docs/signal-processing/ue-reception/) and [Uplink Communication](/docs/signal-processing/uplink-communication/) cover channel estimation, residual CFO/SFO compensation, and bidirectional information recovery.
- [Monostatic Sensing](/docs/signal-processing/monostatic-sensing/) and [Bistatic Sensing](/docs/signal-processing/bistatic-sensing/) derive sensing products from the same grids. The latter presents OTA LoS tracking and eRTM as alternative bistatic-timing methods.

The derivations assume that the maximum delay spread fits inside the cyclic prefix and that Doppler within one OFDM symbol is much smaller than the subcarrier spacing. Under these conditions, inter-symbol and inter-carrier interference are negligible and each received resource element can be modeled independently.
