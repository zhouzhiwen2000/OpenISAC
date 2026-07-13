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

A frame has $M$ OFDM symbols and duration $T_F=MT_O$. The FFT index $n\in\{0,\ldots,N-1\}$ maps to the subcarrier index

$$
\kappa_n=
\begin{cases}
n, & 0\le n<\dfrac{N}{2},\\
n-N, & \dfrac{N}{2}\le n<N.
\end{cases}
$$

The corresponding baseband frequency is $f_n=\kappa_n\Delta f$. $\operatorname{rect}(\cdot)$ denotes a rectangular symbol window of duration $T_O$.

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

for downlink, guard, and uplink symbols. The downlink and uplink share the same frame-wide symbol index $m$. The uplink grid is zero on $\mathcal S_\mathrm{DL}\cup\mathcal S_\mathrm{G}$ and carries active uplink symbols on $\mathcal S_\mathrm{UL}$ after the guard interval. In FDD, both links remain continuous on separate carriers:

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

The downlink always places a main full-band ZC at $m_\mathrm{sync}$. Two acquisition fields may be enabled independently to trade resource overhead for robustness to a larger initial CFO:

- **Second synchronization symbol:** the same ZC OFDM symbol is placed at $m_\mathrm{sync}-1$. The useful parts of the two consecutive ZC symbols provide Schmidl-Cox-type coarse timing and modulo-CFO estimates; the main ZC still provides the final fine timing estimate.
- **CFO training field:** a symbol whose useful part repeats every $N_\mathrm{CFO}$ samples is placed at $m_\mathrm{sync}+1$. It supplies an independent CFO reference for ambiguity resolution among CFO candidates; it does not replace the final CP-tail CFO refinement.

The compact configuration keeps only the main ZC and is suitable when a stable reference already limits the initial CFO. The second synchronization symbol primarily protects coarse acquisition from CFO-induced degradation of ZC correlation, while the CFO field makes candidate selection more reliable. The CFO field is not a sensing resource. Optional mid-frame full-band references provide additional channel anchors in normal reception.

The current uplink places one full-band ZC in its first active symbol, followed by comb pilots and QPSK data. Distinct ZC roots identify the two links.

## Sensing Resources

BS monostatic sensing uses known downlink resources

$$
\Omega_\mathrm{sens}
\subseteq
\{(n,m):m\in\mathcal S_\mathrm{DL}\}.
$$

Synchronization, pilot, communication-data, and random idle QPSK resources can all contribute when their transmitted symbols are known. Constant-modulus ZC/QPSK prevents element-wise modulation removal from excessively amplifying noise. Uniformly spaced resources support a direct delay-Doppler 2D FFT; nonuniform resources require their actual frequency locations and sampling times. Resolution and unambiguous-range expressions are given with the [monostatic](/docs/signal-processing/monostatic-sensing/#resolution-and-unambiguous-ranges) and [bistatic](/docs/signal-processing/bistatic-sensing/#bistatic-output-and-resolution) processors where they are used.
