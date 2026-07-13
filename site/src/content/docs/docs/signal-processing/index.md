---
title: Signal Processing Overview
description: A unified chain from bidirectional OFDM transmission to multichannel monostatic and bistatic sensing.
---

OpenISAC connects four functions through one OFDM signal model: BS-to-UE downlink communication, UE-to-BS uplink communication, multichannel monostatic sensing at the BS, and bistatic sensing at the UE. They share the same frame, subcarrier, and reference-signal definitions, so timing, frequency-offset, and channel estimates obtained for communication can also support sensing.

## End-to-End Chain

1. The BS and UE map synchronization sequences, pilots, and coded QPSK symbols onto separate downlink and uplink resource grids.
2. IFFT and cyclic-prefix insertion convert each grid into a continuous OFDM waveform. TDD partitions one frame into downlink, guard, and uplink symbols; FDD transmits continuously on two carriers.
3. A full-band Zadoff–Chu (ZC) symbol provides frame timing and the initial channel estimate. Pilots then track residual carrier-frequency offset (CFO) and sampling-frequency offset (SFO).
4. Each communication receiver equalizes its data resources, computes QPSK soft information, and applies LDPC decoding.
5. The BS divides each sensing channel's received grid by the known downlink grid, producing a multichannel time-frequency channel tensor for range–Doppler–angle or micro-Doppler processing.
6. The UE reconstructs unknown downlink data symbols before forming its bistatic channel grid. Continuous timing-offset and SFO compensation stabilize the delay axis over long observations.
7. When both directions are available, eRTM combines downlink and uplink delay relations with calibrated fixed terms to separate the BS-side and UE-side timing offsets.

## Shared Notation

| Symbol | Meaning |
|---|---|
| $x\in\{\mathrm{DL},\mathrm{UL}\}$ | Downlink or uplink |
| $\gamma$ | Continuous OFDM frame index |
| $m$ | OFDM symbol index within a frame |
| $n$ | FFT storage index; $\kappa_n$ is its signed subcarrier index |
| $r=0,\ldots,R-1$ | BS sensing-array channel index |
| $\boldsymbol B_\gamma^x$ | Transmit resource grid for link $x$ |
| $\boldsymbol Y_\gamma^x$ | Received resource grid after the FFT |
| $\boldsymbol H_\gamma^x$ | Communication-channel frequency response |
| $\boldsymbol F_\gamma$ | Sensing channel symbols after modulation removal |

Italic lower-case, bold lower-case, and bold upper-case symbols denote scalars, vectors, and matrices. $(\cdot)^T$, $(\cdot)^H$, and $(\cdot)^*$ denote transpose, Hermitian transpose, and complex conjugation.

## Reading Order

- [Signal Model](/docs/signal-processing/signal-model/) defines the downlink, uplink, multichannel monostatic, and bistatic propagation models.
- [OFDM Resources](/docs/signal-processing/ofdm-resources/) defines the continuous waveform, TDD/FDD sets, and reference signals.
- [Synchronization, CFO, and SFO](/docs/signal-processing/sync-cfo-sfo/) gives the acquisition and tracking operations shared by both communication directions.
- [Downlink Communication](/docs/signal-processing/ue-reception/) and [Uplink Communication](/docs/signal-processing/uplink-communication/) complete bidirectional information recovery.
- [Monostatic Sensing](/docs/signal-processing/monostatic-sensing/) and [Bistatic Sensing](/docs/signal-processing/bistatic-sensing/) derive sensing products from the same grids.
- [OTA and eRTM Timing](/docs/signal-processing/ota-ertm-timing/) connects relative delay, absolute timing offset, and long-term stability.

The derivations assume that the maximum delay spread fits inside the cyclic prefix and that Doppler within one OFDM symbol is much smaller than the subcarrier spacing. Under these conditions, inter-symbol and inter-carrier interference are negligible and each received resource element can be modeled independently.
