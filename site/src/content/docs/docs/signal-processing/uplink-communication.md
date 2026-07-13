---
title: Uplink Communication
description: The UE-to-BS compact OFDM frame, duplex timing, channel estimation, equalization, and decoding.
---

The uplink reuses $N$, $N_\mathrm{CP}$, $\Delta f$, the frame-wide symbol index $m$, and the pilot positions, but has an independent grid $\boldsymbol B_\gamma^\mathrm{UL}$ and ZC root. The UE has already synchronized to the BS through the downlink; the uplink is a complete UE-to-BS communication link, not a separate initial-acquisition procedure or a reversal of downlink symbols. In TDD, the uplink grid is zero for $m\in\mathcal S_\mathrm{DL}\cup\mathcal S_\mathrm{G}$ and carries uplink symbols only for $m\in\mathcal S_\mathrm{UL}$.

## Uplink Frame

Let $m_{\mathrm{UL},0}=\min\mathcal S_\mathrm{UL}$. The first active uplink OFDM symbol in the frame is full-band ZC:

$$
b_{n,m_{\mathrm{UL},0},\gamma}^\mathrm{UL}=z_n^\mathrm{UL}.
$$

The remaining symbols with $m\in\mathcal S_\mathrm{UL}\setminus\{m_{\mathrm{UL},0}\}$ carry known pilots on $n\in\mathcal P$ and coded QPSK on $n\in\mathcal D$:

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
\alpha_l^\mathrm{UL}(t_{m,\gamma}^\mathrm{UL})
e^{j2\pi\left[
(f_{D,l}^\mathrm{UL}+\Delta\bar f_{c,\gamma}^\mathrm{UL})
t_{m,\gamma}^\mathrm{UL}
-\kappa_n\Delta f
(\tau_{l,\mathrm{prop}}(t_{m,\gamma}^\mathrm{UL})
+\tau_\mathrm{UL}^\mathrm{RF}
-\tau_d^\mathrm{BS}(t_{m,\gamma}^\mathrm{UL}))
\right]}.
$$

$t_{m,\gamma}^\mathrm{UL}$ is the actual reference time of uplink symbol $m$ within the frame, and $\tau_d^\mathrm{BS}(t)$ is the time-varying offset of the BS's current demodulation window relative to the uplink transmitter frame boundary. See the [Signal Model](/docs/signal-processing/signal-model/#bidirectional-channel-delay-components) for its relation to propagation delay, uplink RF group delay, and the locally observed TO.

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

The same weighted fit used in the downlink gives

$$
(\hat a_\mathrm{UL},\hat b_\mathrm{UL})
=\arg\min_{a,b}\sum_{n\in\mathcal P}
|\bar R_\gamma^\mathrm{UL}[n]|^2
\left|
\operatorname{unwrap}(\arg\bar R_\gamma^\mathrm{UL}[n])-a-b\kappa_n
\right|^2,
$$

$$
\hat f_{o,\gamma}^\mathrm{UL}
=\frac{\hat a_\mathrm{UL}}{2\pi T_O},
\qquad
\Delta\hat T_{s,\gamma}^\mathrm{UL}
=-\frac{\hat b_\mathrm{UL}}{2\pi\Delta fN_s}.
$$

The channel is propagated as

$$
\hat H_{n,m,\gamma}^\mathrm{UL}
=\hat H_{n,0,\gamma}^\mathrm{UL}
e^{j2\pi m(
\hat f_{o,\gamma}^\mathrm{UL}T_O
-\kappa_n\Delta fN_s\Delta\hat T_{s,\gamma}^\mathrm{UL})}.
$$

If $\mathcal A_\mathrm{UL}$ is empty, the local frame provides no cross-symbol pilot fit. Residual CFO/SFO estimation and compensation have the same form in both directions, but each link uses its own references and observations; communication decoding does not rely on ideal reciprocity.

## Equalization and Information Recovery

With ZF or MMSE coefficient $G_{n,m}^\mathrm{UL}$,

$$
\hat d_{n,m,\gamma}^\mathrm{UL}
=G_{n,m}^\mathrm{UL}Y_{n,m,\gamma}^\mathrm{UL}.
$$

Equalized pilot residuals estimate $\hat\sigma_\mathrm{eq}^2$, which scales the QPSK LLRs. Soft deinterleaving, descrambling, and LDPC decoding then recover the UE information bits. All channel, frequency-offset, and noise estimates come from the uplink's own references.

## Relation to eRTM

The uplink also supplies the BS-side channel estimate $\hat H_{\mathrm{BS}}[n]$. When both directions are enabled, eRTM combines it with the UE-side downlink estimate $\hat H_{\mathrm{UE}}[n]$ and uses the relationship between the uplink and downlink channels to estimate the timing offsets at the two endpoints; see the [eRTM option in Bistatic Sensing](/docs/signal-processing/bistatic-sensing/#ertm-bidirectional-timing-option).
