---
title: Synchronization, CFO, and SFO
description: Frame timing, frequency acquisition, channel estimation, and long-term tracking shared by the downlink and uplink.
---

Synchronization continuously separates three quantities: integer or fractional timing offset, carrier-frequency offset (CFO), and sampling-frequency offset (SFO). For the dominant path, the received phase is approximately

$$
\phi_{n,m}
\approx
2\pi f_o mT_O
-2\pi\kappa_n\Delta f
\left(\tau_o+mN_s\Delta T_s\right),
$$

where $f_o=f_{D,1}+\Delta f_c$ combines dominant-path Doppler and residual CFO, and $\tau_o$ is the current timing offset. CFO changes phase along OFDM symbols; delay creates a subcarrier phase slope; SFO makes that slope drift over time.

## 1. ZC Frame Timing

Let $s_\mathrm{ZC}[k]$ be the local time-domain ZC symbol including its cyclic prefix, with length $N_s$. For candidate position $u$, compute

$$
\Lambda_\mathrm{ZC}[u]
=
\frac{
\left|\sum_{k=0}^{N_s-1}y[u+k]s_\mathrm{ZC}^{*}[k]\right|^2
}{
\left(\sum_{k=0}^{N_s-1}|y[u+k]|^2\right)
\left(\sum_{k=0}^{N_s-1}|s_\mathrm{ZC}[k]|^2\right)
}.
$$

The peak $\hat u=\arg\max_u\Lambda_\mathrm{ZC}[u]$ locates the synchronization symbol. If it is symbol $m_\mathrm{sync}$ within the frame, the integer frame timing estimate is

$$
\hat k_\mathrm{TO}
=\hat u-m_\mathrm{sync}N_s-N_\mathrm{lag},
$$

where $N_\mathrm{lag}$ retains margin for multipath components preceding the strongest arrival. A peak-to-search-region-average ratio provides detection confidence and avoids locking to noise.

## 2. Coarse CFO Acquisition

For a repeated training structure separated by $D$ samples,

$$
P_D[u]=\sum_{k=0}^{N-1}y^*[u+k]y[u+D+k],
$$

$$
\hat f_{o,\mathrm{mod}}
=\frac{\angle P_D[\hat u]}{2\pi DT_s}.
$$

This estimate is ambiguous modulo $1/(DT_s)$. A ZC metric or another repetition resolves the candidate, after which cyclic-prefix correlation and pilots refine the residual offset. A one-ZC configuration removes this overhead when the initial CFO is already controlled.

## 3. Initial Channel and Delay Profile

After cyclic-prefix removal and the FFT, either communication link satisfies

$$
Y_{n,m,\gamma}^{x}
=b_{n,m,\gamma}^{x}H_{n,m,\gamma}^{x}
+Z_{n,m,\gamma}^{x}.
$$

On a full-band ZC symbol,

$$
\hat H_{n,m_\mathrm{sync},\gamma}^{x,\mathrm{LS}}
=\frac{Y_{n,m_\mathrm{sync},\gamma}^{x}}{z_n^x},
$$

with delay-domain representation

$$
\hat h_\gamma^x[k]
=\frac{1}{N}\sum_{n=0}^{N-1}
\hat H_{n,m_\mathrm{sync},\gamma}^{x,\mathrm{LS}}
e^{j2\pi nk/N}.
$$

The dominant peak updates timing, while energy outside the cyclic-prefix support estimates noise. To reduce LS noise, retain the valid delay support, zero the remaining taps, apply Wiener weight

$$
w_\mathrm{W}=\frac{\widehat{\mathrm{SNR}}}{\widehat{\mathrm{SNR}}+1},
$$

and transform back to obtain $\hat H_{n,m_\mathrm{sync},\gamma}^{x}$.

## 4. Joint Pilot Tracking of CFO and SFO

For adjacent active symbols that contain the same known pilots,

$$
\bar R_\gamma^x[n]
=\frac{1}{|\mathcal A_x|}
\sum_{m\in\mathcal A_x}
\left(Y_{n,m,\gamma}^{x}\right)^*
Y_{n,m+1,\gamma}^{x},
\qquad n\in\mathcal P.
$$

$\mathcal A_x$ excludes inactive symbols, TDD guards, and duplex boundaries. After phase unwrapping,

$$
\varphi_\gamma^x[n]
=\arg\bar R_\gamma^x[n]
\approx
2\pi\left(f_{o,\gamma}^xT_O
-\kappa_n\Delta fN_s\Delta T_{s,\gamma}^x\right).
$$

A weighted linear fit gives

$$
(\hat a,\hat b)
=\arg\min_{a,b}
\sum_{n\in\mathcal P}
|\bar R_\gamma^x[n]|^2
\left|\operatorname{unwrap}(\varphi_\gamma^x[n])-a-b\kappa_n\right|^2,
$$

and therefore

$$
\hat f_{o,\gamma}^x=\frac{\hat a}{2\pi T_O},
\qquad
\Delta\hat T_{s,\gamma}^x
=-\frac{\hat b}{2\pi\Delta fN_s}.
$$

## 5. Intra-Frame Phase and Long-Term Timing

Without another full-band reference, propagate the initial channel as

$$
\hat H_{n,m,\gamma}^{x}
=\hat H_{n,m_\mathrm{sync},\gamma}^{x}
e^{j2\pi(m-m_\mathrm{sync})
(\hat f_{o,\gamma}^xT_O
-\kappa_n\Delta fN_s\Delta\hat T_{s,\gamma}^x)}.
$$

A communication receiver needs to move the integer frame origin only when accumulated timing drift approaches one sample or threatens cyclic-prefix margin; sub-sample residuals can remain in the channel phase. Bistatic sensing measures delay itself and therefore needs a [continuous sensing timing estimate](/docs/signal-processing/bistatic-sensing/#5-continuous-sensing-timing) rather than staircase corrections.
