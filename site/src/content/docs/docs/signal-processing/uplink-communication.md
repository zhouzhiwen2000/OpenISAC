---
title: Uplink Communication
description: The UE-to-BS compact OFDM frame, duplex timing, channel estimation, equalization, and decoding.
---

The uplink reuses $N$, $N_\mathrm{CP}$, $\Delta f$, and the pilot positions, but has an independent grid $\boldsymbol B_\gamma^\mathrm{UL}$ and ZC root. It is a complete UE-to-BS communication link rather than a reversal of downlink symbols. Let $M_\mathrm{UL}$ be the active-symbol count of each local uplink frame: $M_\mathrm{UL}=|\mathcal S_\mathrm{UL}|$ in TDD and $M_\mathrm{UL}=M$ in FDD.

## Uplink Frame

The first active symbol of every local uplink frame is full-band ZC:

$$
b_{n,0,\gamma}^\mathrm{UL}=z_n^\mathrm{UL}.
$$

Subsequent symbols carry known pilots on $n\in\mathcal P$ and coded QPSK on $n\in\mathcal D$:

$$
b_{n,m,\gamma}^\mathrm{UL}=
\begin{cases}
p_{n,m}^\mathrm{UL},&n\in\mathcal P,\\
d_{n,m,\gamma}^\mathrm{UL},&n\in\mathcal D.
\end{cases}
$$

In TDD, this compact frame occupies $\mathcal S_\mathrm{UL}$ after the guard interval. A positive timing advance $t_\mathrm{TA,UE}$ moves the UE waveform earlier so that propagation places it in the BS uplink observation interval. In FDD, the uplink uses a continuous $M$-symbol frame on its own carrier.

## Frequency-Domain Model at the BS

After frame-boundary alignment, cyclic-prefix removal, and the FFT,

$$
Y_{n,m,\gamma}^\mathrm{UL}
=b_{n,m,\gamma}^\mathrm{UL}
H_{n,m,\gamma}^\mathrm{UL}
+Z_{n,m,\gamma}^\mathrm{UL},
$$

$$
H_{n,m,\gamma}^\mathrm{UL}
=\sum_{l=1}^{L_\mathrm{UL}}
\alpha_l^\mathrm{UL}
e^{j2\pi\left[
(f_{D,l}^\mathrm{UL}+\Delta\bar f_{c,\gamma}^\mathrm{UL})
(m+\gamma M)T_O
-\kappa_n\Delta f
(\tau_l^\mathrm{UL}+\bar\tau_{d,\gamma,m}^\mathrm{UL})
\right]}.
$$

The uplink ZC gives

$$
\hat H_{n,0,\gamma}^\mathrm{UL,LS}
=\frac{Y_{n,0,\gamma}^\mathrm{UL}}{z_n^\mathrm{UL}}.
$$

Limiting its delay-domain support to the cyclic-prefix region and applying Wiener smoothing suppresses noise while retaining the multipath structure.

## Uplink Phase Tracking

Let $\mathcal A_\mathrm{UL}$ contain indices for which two adjacent local-uplink symbols both carry pilots. When $M_\mathrm{UL}\ge3$ and all data symbols are consecutive, $\mathcal A_\mathrm{UL}=\{1,\ldots,M_\mathrm{UL}-2\}$. Then

$$
\bar R_\gamma^\mathrm{UL}[n]
=\frac{1}{|\mathcal A_\mathrm{UL}|}
\sum_{m\in\mathcal A_\mathrm{UL}}
(Y_{n,m,\gamma}^\mathrm{UL})^*
Y_{n,m+1,\gamma}^\mathrm{UL},
$$

with unwrapped phase

$$
\arg\bar R_\gamma^\mathrm{UL}[n]
\approx
2\pi\left(
f_{o,\gamma}^\mathrm{UL}T_O
-\kappa_n\Delta fN_s\Delta T_{s,\gamma}^\mathrm{UL}
\right).
$$

Weighted linear regression again separates the common phase term from the subcarrier slope and propagates the synchronization-symbol channel to every data symbol. If $\mathcal A_\mathrm{UL}$ is empty, the local frame provides no cross-symbol pilot fit. Uplink and downlink estimate these quantities independently rather than assuming ideal reciprocity.

## Equalization and Information Recovery

With ZF or MMSE coefficient $G_{n,m}^\mathrm{UL}$,

$$
\hat d_{n,m,\gamma}^\mathrm{UL}
=G_{n,m}^\mathrm{UL}Y_{n,m,\gamma}^\mathrm{UL}.
$$

Equalized pilot residuals estimate $\hat\sigma_\mathrm{eq}^2$, which scales the QPSK LLRs. Soft deinterleaving, descrambling, and LDPC decoding then recover the UE information bits. All channel, frequency-offset, and noise estimates come from the uplink's own references.

## Relation to eRTM

The uplink also supplies the delay structure in $\hat H^\mathrm{UL}[n]$. Comparing it with the downlink delay profile observed at the UE constrains the relative timing of the two directions and enables separation of BS-side and UE-side timing offsets; see [OTA and eRTM Timing](/docs/signal-processing/ota-ertm-timing/).
