---
title: OFDM Resources
description: Frame structure, pilots, synchronization symbols, and resource mapping.
---

OpenISAC uses a configurable OFDM resource grid rather than a standard Wi-Fi/NR frame. This keeps the PHY compact and makes sensing resources explicit.

![Example OFDM frame structure](/images/OpenISAC_FrameStructure.png)

## Continuous Frames

OpenISAC adopts continuous OFDM transmission instead of traffic-driven packet bursts.

![Packet radio and continuous waveform timing](/images/PacketVSCW.png)

Packet radios have irregular frame intervals, which makes slow-time sampling uneven and degrades Doppler or micro-Doppler processing. Continuous frames provide deterministic OFDM-symbol spacing, so sensing symbols can be accumulated over long coherent windows.

## Symbol Sets

Let $m_\mathrm{sync}$ be the mandatory full-band Zadoff-Chu synchronization symbol. Optional acquisition fields can add a second ZC symbol at $m_\mathrm{sync}-1$ and a repeated CFO training field at $m_\mathrm{sync}+1$.

Define:

$$
\mathcal{S}_\mathrm{ZC}=\{m_\mathrm{sync}\}\cup\mathcal{S}_\mathrm{sec},
\quad
\mathcal{S}_\mathrm{CFO}=\{m_\mathrm{sync}+1\}\ \text{if enabled}
$$

Let $\mathcal{P}$ be the pilot-subcarrier set and $\mathcal{D}=\{0,\ldots,N-1\}\setminus\mathcal{P}$ be the data-subcarrier set.

The frequency-domain resource mapping is

$$
b_{n,m,\gamma}=
\begin{cases}
z_n, & m\in\mathcal{S}_\mathrm{ZC},\\
c_n^\mathrm{CFO}, & m\in\mathcal{S}_\mathrm{CFO},\\
z_n, & m\notin\mathcal{S}_\mathrm{ZC}\cup\mathcal{S}_\mathrm{CFO},\ n\in\mathcal{P},\\
d_{n,m,\gamma}, & m\notin\mathcal{S}_\mathrm{ZC}\cup\mathcal{S}_\mathrm{CFO},\ n\in\mathcal{D}.
\end{cases}
$$

Here $z_n$ is the ZC reference sequence, $c_n^\mathrm{CFO}$ is the optional CFO-training symbol, and $d_{n,m,\gamma}$ is the QPSK data symbol after coding and scrambling.

## Resource Roles

- Full-band ZC symbols provide timing, coarse CFO support, and initial channel acquisition.
- Pilot subcarriers track residual CFO/SFO and support per-frame channel propagation.
- Data subcarriers carry communication payloads or random QPSK padding when no payload is available.
- The full resource grid is retained by the backend so sensing can remove modulation and estimate the channel matrix.

## Design Tradeoffs

Increasing $N$ or bandwidth improves delay resolution and increases compute and I/O load. Increasing sensing-frame length improves Doppler resolution and increases latency. Enabling extra acquisition symbols improves initial robustness but consumes resources that otherwise remain available for sensing.
