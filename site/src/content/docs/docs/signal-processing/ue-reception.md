---
title: Downlink Communication
description: BS-to-UE channel estimation, CFO/SFO tracking, equalization, soft demapping, and LDPC decoding.
---

Downlink processing begins after the UE has acquired frame timing and formed the FFT grid $\boldsymbol Y_\gamma^\mathrm{DL}$. The complete chain is

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

where

$$
H_{n,m,\gamma}^\mathrm{DL}
=\sum_{l=1}^{L_\mathrm{DL}}
\alpha_l^\mathrm{DL}
e^{j2\pi\left[
(f_{D,l}^\mathrm{DL}+\Delta\bar f_{c,\gamma}^\mathrm{DL})
(m+\gamma M)T_O
-\kappa_n\Delta f
(\tau_l^\mathrm{DL}+\bar\tau_{d,\gamma,m}^\mathrm{DL})
\right]}.
$$

$\Delta\bar f_{c,\gamma}^\mathrm{DL}$ and $\bar\tau_{d,\gamma,m}^\mathrm{DL}$ are the residual frequency and timing offsets after current compensation.

## Channel References and Intra-Frame Reconstruction

The full-band ZC gives

$$
\hat H_{n,m_\mathrm{sync},\gamma}^\mathrm{LS}
=\frac{Y_{n,m_\mathrm{sync},\gamma}^\mathrm{DL}}
{z_n^\mathrm{DL}}.
$$

Delay-support limiting and Wiener smoothing produce a lower-noise channel anchor. Optional mid-frame full-band references provide more anchors $\hat H_{n,m_a,\gamma}$. Between adjacent anchors $m_a<m<m_b$,

$$
\hat H_{n,m,\gamma}^{\mathrm{base}}
=(1-\xi_m)\hat H_{n,m_a,\gamma}
+\xi_m\hat H_{n,m_b,\gamma},
\qquad
\xi_m=\frac{m-m_a}{m_b-m_a}.
$$

The residual CFO/SFO phase from the pilots is then applied at each symbol time. With no additional anchors, this reduces to propagation from the synchronization symbol across the frame.

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

ZF preserves the constellation directly at high SNR; MMSE limits noise enhancement in deep fades. For data resources,

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
