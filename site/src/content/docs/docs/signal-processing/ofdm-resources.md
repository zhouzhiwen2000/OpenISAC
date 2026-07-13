---
title: OFDM Resources
description: Continuous OFDM waveforms, TDD/FDD frame partitions, and bidirectional reference and data resources.
---

The downlink and uplink share one OFDM numerology but use separate resource grids and ZC roots. With FFT size $N$, cyclic-prefix length $N_\mathrm{CP}$, and subcarrier spacing $\Delta f$,

$$
T=\frac{1}{\Delta f},\qquad
B=N\Delta f,\qquad
T_s=\frac{1}{B},
$$

$$
N_s=N+N_\mathrm{CP},\qquad
T_\mathrm{CP}=N_\mathrm{CP}T_s,\qquad
T_O=N_sT_s=T+T_\mathrm{CP}.
$$

A frame has $M$ OFDM symbols and duration $T_F=MT_O$. The storage index $n\in\{0,\ldots,N-1\}$ maps to signed subcarrier index $\kappa_n$ and baseband frequency $f_n=\kappa_n\Delta f$. $\operatorname{rect}(\cdot)$ denotes a rectangular symbol window of duration $T_O$.

## Continuous OFDM Waveforms

For $x\in\{\mathrm{DL},\mathrm{UL}\}$,

$$
s_x(t)=\sum_{\gamma=0}^{\infty}s_{x,\gamma}(t-\gamma T_F),
$$

$$
s_{x,\gamma}(t)=
\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}
b_{n,m,\gamma}^{x}
e^{j2\pi\kappa_n\Delta f(t-mT_O-T_\mathrm{CP})}
\operatorname{rect}\!\left(\frac{t-mT_O}{T_O}\right).
$$

$\boldsymbol B_\gamma^x=[b_{n,m,\gamma}^x]\in\mathbb C^{N\times M}$ is the transmit grid, with zeros on resources inactive for link $x$. Continuous, uniformly spaced OFDM symbols provide uniform slow-time samples and avoid the Doppler sidelobes caused by irregular packet intervals.

## TDD and FDD Sets

Let $\mathcal M=\{0,\ldots,M-1\}$. TDD partitions every frame into disjoint sets

$$
\mathcal M
=\mathcal S_\mathrm{DL}\,\dot\cup\,
\mathcal S_\mathrm{G}\,\dot\cup\,
\mathcal S_\mathrm{UL},
$$

for downlink, guard, and uplink symbols. The compact uplink frame begins at the first active symbol after the guard interval. In FDD, both links remain continuous on separate carriers:

$$
\mathcal S_\mathrm{DL}=\mathcal S_\mathrm{UL}=\mathcal M,
\qquad
\mathcal S_\mathrm{G}=\varnothing.
$$

Below, $\mathcal S_x$ denotes the active symbol set for link $x$. Duplexing changes only the time-frequency occupancy; FFT demodulation, channel estimation, equalization, and soft demapping retain the same mathematical form.

## Synchronization, Reference, and Data Resources

Let $\mathcal P\subset\{0,\ldots,N-1\}$ be the comb-pilot subcarrier set and $\mathcal D=\{0,\ldots,N-1\}\setminus\mathcal P$ the data-subcarrier set. For each link, define full-band ZC symbols $\mathcal S_\mathrm{ZC}^x$, optional repeated CFO-training symbols $\mathcal S_\mathrm{CFO}^x$, known pilot or channel-reference resources $\Omega_\mathrm{ref}^x$, and coded-data resources $\Omega_\mathrm{data}^x$. Then

$$
b_{n,m,\gamma}^{x}=
\begin{cases}
z_n^{x},&m\in\mathcal S_\mathrm{ZC}^{x},\\[2pt]
c_n^{x,\mathrm{CFO}},&m\in\mathcal S_\mathrm{CFO}^{x},\\[2pt]
p_{n,m}^{x},&(n,m)\in\Omega_\mathrm{ref}^{x},\\[2pt]
d_{n,m,\gamma}^{x},&(n,m)\in\Omega_\mathrm{data}^{x},\\[2pt]
0,&m\notin\mathcal S_x.
\end{cases}
$$

$z_n^x$ is a link-specific constant-modulus ZC sequence, $p_{n,m}^x$ is a known pilot, and $c_n^{x,\mathrm{CFO}}$ is an optional CFO-training symbol. Data uses normalized QPSK:

$$
\mathcal A_\mathrm{QPSK}
=\left\{\frac{1}{\sqrt2}(\pm1\pm j)\right\}.
$$

The current downlink has at least one full-band ZC per frame and may add a second ZC, a CFO-training field, and mid-frame full-band channel references. The current uplink places one full-band ZC in its first active symbol, followed by comb pilots and QPSK data. Distinct ZC roots identify the two links.

## Sensing Resources

BS monostatic sensing uses known downlink resources

$$
\Omega_\mathrm{sens}
\subseteq
\{(n,m):m\in\mathcal S_\mathrm{DL}\}.
$$

Synchronization, pilot, communication-data, and random idle QPSK resources can all contribute when their transmitted symbols are known. Constant-modulus ZC/QPSK prevents element-wise modulation removal from excessively amplifying noise. Uniformly spaced selections support a direct Doppler FFT; nonuniform selections require their actual sampling times.

## Resolution

Using $N$ contiguous subcarriers gives basic delay resolution

$$
\Delta\tau=\frac{1}{B}.
$$

The monostatic range resolution is $\Delta r_\mathrm{mono}=c/(2B)$, while LoS-referenced bistatic excess-path resolution is $\Delta d_\mathrm{bi}=c/B$. The cyclic prefix bounds the delay spread that remains free of inter-symbol interference. Greater bandwidth improves delay resolution, while a longer coherent slow-time observation improves Doppler resolution.
