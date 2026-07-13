---
title: UE Reception
description: UE synchronization, CFO/SFO tracking, channel estimation, equalization, and payload recovery.
---

The UE receive path is a two-state finite-state machine: `SYNC_SEARCH` and `NORMAL`. `SYNC_SEARCH` finds frame timing and coarse carrier offset. `NORMAL` demodulates frames, tracks residual timing/frequency offsets, and decodes payloads.

![UE receive processing flow](/images/FlowGraph_UE.png)

## ZC Timing Metric

Let $N_s=N+N_\mathrm{CP}$ be the number of samples in one OFDM symbol. In the compact one-ZC configuration, the UE correlates the received block with the local time-domain ZC reference $s_\mathrm{ZC}[k]$:

$$
\Lambda_\mathrm{ZC}[u]
=
\frac{
\left|\sum_{k=0}^{N_s-1}y_\mathrm{UE}[u+k]s_\mathrm{ZC}^{*}[k]\right|^2
}{
\left(\sum_{k=0}^{N_s-1}|y_\mathrm{UE}[u+k]|^2\right)
\left(\sum_{k=0}^{N_s-1}|s_\mathrm{ZC}[k]|^2\right)
}
$$

The implementation compares the peak-to-average ratio

$$
\rho_\mathrm{ZC}
=
\frac{\Lambda_\mathrm{ZC}[\hat{k}_\mathrm{peak}]}
{\frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}\Lambda_\mathrm{ZC}[u]}
$$

against a threshold. If accepted, the initial timing estimate is

$$
\hat{k}_\mathrm{TO}
=
\hat{k}_\mathrm{peak}-m_\mathrm{sync}N_s-N_\mathrm{lag}
$$

The $N_\mathrm{lag}$ margin leaves room for multipath components that arrive earlier than the strongest path.

## Optional CFO Acquisition

For larger initial CFO, an optional second ZC symbol supports a Schmidl-Cox-style metric:

$$
P_\mathrm{SC}[u]
=
\sum_{k=0}^{N-1}
y_\mathrm{UE}^{*}[u+N_\mathrm{CP}+k]
y_\mathrm{UE}[u+N_s+N_\mathrm{CP}+k]
$$

$$
R_\mathrm{SC}[u]
=
\sum_{k=0}^{N-1}|y_\mathrm{UE}[u+N_s+N_\mathrm{CP}+k]|^2,
\quad
\Lambda_\mathrm{SC}[u]=\frac{|P_\mathrm{SC}[u]|^2}{R_\mathrm{SC}^2[u]}
$$

The modulo CFO estimate is

$$
\hat{f}_{o,\mathrm{mod}}
=
\frac{\angle P_\mathrm{SC}[\hat{u}]}{2\pi N_sT_s}
$$

The receiver then tests CFO candidates against the local ZC metric and refines the chosen frequency estimate with CP-tail correlation or the optional CFO training field.

## NORMAL-State Channel Model

After timing/frequency compensation, frame $\gamma$ is FFT-demodulated into

$$
(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m}
=
b_{n,m,\gamma}(\boldsymbol{H}_{\mathrm{UE},\gamma})_{n,m}
+(\boldsymbol{Z}_{\mathrm{UE},\gamma})_{n,m}
$$

where

$$
(\boldsymbol{H}_{\mathrm{UE},\gamma})_{n,m}
=
\sum_{l=1}^{L}
\alpha_l
e^{j2\pi\left((f_{D,l}+\Delta\bar{f}_{c,\gamma})mT_O
-n\Delta f(\tau_l+\bar{\tau}_{d,\gamma,mN_s})\right)}
$$

The full-band ZC symbol gives the initial channel estimate:

$$
(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_\mathrm{sync}}
=
\frac{(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m_\mathrm{sync}}}{z_n}
$$

Its delay spectrum is

$$
\tilde{p}_{\mathrm{delay},\gamma}[k]
=
\frac{1}{\sqrt{N}}
\sum_{n=0}^{N-1}
(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_\mathrm{sync}}
e^{j2\pi nk/N}
$$

and $|\tilde{p}_{\mathrm{delay},\gamma}[k]|^2$ is used for delay-peak tracking.

## CFO/SFO Tracking

For pilot subcarriers $n\in\mathcal{P}$, the receiver forms cross-symbol autocorrelation:

$$
(\boldsymbol{R}_{\mathrm{UE},\gamma})_{n,m}
=
(\boldsymbol{B}_{\mathrm{UE},\gamma}^{*})_{n,m}
(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m+1}
$$

After averaging across OFDM symbols:

$$
\bar{R}_{\mathrm{UE},\gamma}[n]
=
\frac{1}{M-1}
\sum_{m=0}^{M-2}
(\boldsymbol{R}_{\mathrm{UE},\gamma})_{n,m}
$$

the pilot phase is approximately linear in subcarrier index:

$$
\varphi_{\mathrm{UE},\gamma}[n]
=
\arg(\bar{R}_{\mathrm{UE},\gamma}[n])
\approx
2\pi(f_{o,\gamma}T_O-n\Delta f\,N_s\Delta T_{s,\gamma})
$$

Weighted linear regression over the pilot phases estimates

$$
\hat{\boldsymbol{\theta}}_\gamma
=
\begin{bmatrix}
\hat{f}_{o,\gamma}\\
\Delta\hat{T}_{s,\gamma}
\end{bmatrix}
$$

The full-frame channel estimate is then propagated from the ZC symbol:

$$
(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m}
=
(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_\mathrm{sync}}
e^{j2\pi(m-m_\mathrm{sync})(\hat{f}_{o,\gamma}T_O-n\Delta fN_s\Delta\hat{T}_{s,\gamma})}
$$

Finally, data symbols are equalized by a one-tap frequency-domain equalizer:

$$
\hat{b}_{n,m,\gamma}
=
\frac{(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m}}
{(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m}}
$$

The equalized symbols feed LLR computation, descrambling, and LDPC decoding.
