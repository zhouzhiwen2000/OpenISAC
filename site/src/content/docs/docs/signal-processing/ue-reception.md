---
title: Downlink Communication
description: BS-to-UE channel estimation, CFO/SFO tracking, equalization, soft demapping, and LDPC decoding.
---

Downlink processing begins after the UE has completed [initial synchronization](/docs/signal-processing/initial-synchronization/) from the downlink fields and formed the FFT grid $\boldsymbol Y_\gamma^\mathrm{DL}$. The complete chain is

$$
\text{ZC channel estimation}
\rightarrow
\text{pilot phase tracking}
\rightarrow
\text{per-symbol channel reconstruction}
\rightarrow
\text{equalization}
\rightarrow
\text{QPSK LLRs}
\rightarrow
\text{LDPC decoding}.
$$

TDD processes $m\in\mathcal S_\mathrm{DL}$, while FDD processes the complete frame on the downlink carrier.

## Frequency-Domain Model

With sufficient cyclic-prefix length and small intra-symbol Doppler,

$$
Y_{n,m,\gamma}^\mathrm{DL}
=b_{n,m,\gamma}^\mathrm{DL}
H_{n,m,\gamma}^\mathrm{DL}
+Z_{n,m,\gamma}^\mathrm{DL},
$$

where $t_{m,\gamma}=(m+\gamma M)T_O$ and

$$
H_{n,m,\gamma}^\mathrm{DL}
=\sum_{l=1}^{L_\mathrm{DL}}
\alpha_l^\mathrm{DL}(t_{m,\gamma})
e^{j2\pi\left[
(f_{D,l}^\mathrm{DL}+\Delta\bar f_{c,\gamma}^\mathrm{DL})
t_{m,\gamma}
-\kappa_n\Delta f
(\tau_{l,\mathrm{prop}}(t_{m,\gamma})
+\tau_\mathrm{DL}^\mathrm{RF}
-\tau_d^\mathrm{UE}(t_{m,\gamma}))
\right]}.
$$

$\Delta\bar f_{c,\gamma}^\mathrm{DL}$ is the residual frequency offset after current compensation, while $\tau_d^\mathrm{UE}(t)$ is the time-varying offset of the UE's current demodulation window relative to the downlink transmitter frame boundary. See the [Signal Model](/docs/signal-processing/signal-model/#bidirectional-channel-delay-components) for the definitions of propagation delay, downlink RF group delay, and TO.

## Channel References, Residual CFO/SFO, and Intra-Frame Reconstruction

The full-band ZC gives

$$
\hat H_{n,m_\mathrm{sync},\gamma}^\mathrm{LS}
=\frac{Y_{n,m_\mathrm{sync},\gamma}^\mathrm{DL}}
{z_n^\mathrm{DL}}.
$$

Delay-support limiting and Wiener smoothing produce a lower-noise channel anchor. The dominant peak relative to $N_\mathrm{lag}$ also updates integer downlink timing; communication moves the frame origin only when the accumulated drift approaches the threshold.

For adjacent active downlink symbols that contain the same known pilots, define

$$
\bar R_\gamma^\mathrm{DL}[n]
=\frac{1}{|\mathcal A_\mathrm{DL}|}
\sum_{m\in\mathcal A_\mathrm{DL}}
(Y_{n,m,\gamma}^\mathrm{DL})^*
Y_{n,m+1,\gamma}^\mathrm{DL},
\qquad n\in\mathcal P.
$$

$\mathcal A_\mathrm{DL}$ contains only pairs wholly inside the active downlink region. After phase unwrapping,

$$
\varphi_\gamma^\mathrm{DL}[n]
=\arg\bar R_\gamma^\mathrm{DL}[n]
\approx2\pi\left(
f_{o,\gamma}^\mathrm{DL}T_O
-\kappa_n\Delta fN_s\Delta T_{s,\gamma}^\mathrm{DL}
\right).
$$

A fit weighted by $|\bar R_\gamma^\mathrm{DL}[n]|^2$ gives

$$
(\hat a,\hat b)
=\arg\min_{a,b}\sum_{n\in\mathcal P}
|\bar R_\gamma^\mathrm{DL}[n]|^2
\left|
\operatorname{unwrap}(\varphi_\gamma^\mathrm{DL}[n])-a-b\kappa_n
\right|^2,
$$

$$
\hat f_{o,\gamma}^\mathrm{DL}
=\frac{\hat a}{2\pi T_O},
\qquad
\Delta\hat T_{s,\gamma}^\mathrm{DL}
=-\frac{\hat b}{2\pi\Delta fN_s}.
$$

The common phase term estimates residual CFO plus dominant-path Doppler, while the evolving subcarrier slope estimates SFO. Both compensate the intra-frame channel phase.

Optional mid-frame full-band references provide more anchors $\hat H_{n,m_a,\gamma}$. Between adjacent anchors $m_a<m<m_b$,

$$
\hat H_{n,m,\gamma}^{\mathrm{base}}
=(1-\xi_m)\hat H_{n,m_a,\gamma}
+\xi_m\hat H_{n,m_b,\gamma},
\qquad
\xi_m=\frac{m-m_a}{m_b-m_a}.
$$

The residual CFO/SFO phase from the pilots is then applied at each symbol time. With no additional anchors,

$$
\hat H_{n,m,\gamma}^\mathrm{DL}
=\hat H_{n,m_\mathrm{sync},\gamma}^\mathrm{DL}
e^{j2\pi(m-m_\mathrm{sync})
(\hat f_{o,\gamma}^\mathrm{DL}T_O
-\kappa_n\Delta fN_s\Delta\hat T_{s,\gamma}^\mathrm{DL})}.
$$

The uplink uses the same residual-estimation and compensation form on its own references and observations.

## ZF and MMSE Equalization

The zero-forcing coefficient is

$$
G_{n,m}^{\mathrm{ZF}}
=\frac{(\hat H_{n,m}^\mathrm{DL})^*}
{\max(|\hat H_{n,m}^\mathrm{DL}|^2,\epsilon)},
$$

while the regularized MMSE coefficient is

$$
G_{n,m}^{\mathrm{MMSE}}
=\frac{(\hat H_{n,m}^\mathrm{DL})^*}
{|\hat H_{n,m}^\mathrm{DL}|^2+\hat\sigma_Z^2/E_s}.
$$

ZF directly inverts the estimated channel but can amplify noise in deep fades; MMSE regularization limits this noise enhancement. For data resources,

$$
\hat d_{n,m,\gamma}^\mathrm{DL}
=G_{n,m}Y_{n,m,\gamma}^\mathrm{DL}.
$$

Equalized pilot errors estimate the effective noise variance:

$$
\hat\sigma_\mathrm{eq}^2
=\frac{1}{|\Omega_\mathrm{ref}^\mathrm{DL}|}
\sum_{(n,m)\in\Omega_\mathrm{ref}^\mathrm{DL}}
|\hat p_{n,m}^\mathrm{DL}-p_{n,m}^\mathrm{DL}|^2.
$$

## QPSK Soft Information and LDPC

For equalized symbol $\hat d$, the max-log LLR of bit $i$ is

$$
L_i(\hat d)
\approx
\frac{
\min\limits_{a\in\mathcal A_i^{(1)}}|\hat d-a|^2
-\min\limits_{a\in\mathcal A_i^{(0)}}|\hat d-a|^2
}{\hat\sigma_\mathrm{eq}^2},
$$

where $\mathcal A_i^{(b)}$ is the QPSK subset whose $i$th label bit equals $b$. Keeping LLR magnitude gives the LDPC decoder reliability information that hard decisions discard.

The transmit order is LDPC encoding, scrambling, interleaving, and QPSK mapping. The UE reverses this order with soft deinterleaving, soft descrambling, and LDPC decoding. The resulting $\hat H_{n,m,\gamma}^\mathrm{DL}$ and symbol decisions also feed [bistatic sensing](/docs/signal-processing/bistatic-sensing/).
