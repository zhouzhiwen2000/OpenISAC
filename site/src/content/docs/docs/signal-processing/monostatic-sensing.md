---
title: Monostatic Sensing
description: BS-side TF-channel extraction, clutter suppression, delay-Doppler processing, and micro-Doppler processing.
---

In monostatic sensing, the BS transmits the OFDM waveform and receives echoes through its sensing receive path. Because the BS knows every transmitted resource symbol, it can remove the modulation and estimate a time-frequency channel matrix for radar processing.

![Monostatic sensing flow](/images/FlowGraph.png)

## TF Grid Mapping

Assuming the maximum target delay fits inside the cyclic prefix and the Doppler shift is small relative to subcarrier spacing, the received BS samples for frame $\gamma$ are rearranged into an $N\times M$ matrix:

$$
(\boldsymbol{B}_{\mathrm{BS},\gamma})_{n,m}
=
b_{n,m,\gamma}
\sum_{p=1}^{P+C}
\beta_p
e^{j2\pi\left(f_{D,s,p}(m+\gamma M)T_O-n\Delta f\tau_{s,p}\right)}
+(\boldsymbol{Z}_{\mathrm{BS},\gamma})_{n,m}
$$

Because $b_{n,m,\gamma}$ is known, element-wise division removes the communication symbols:

$$
(\boldsymbol{F}_{\mathrm{BS},\gamma})_{n,m}
=
\frac{(\boldsymbol{B}_{\mathrm{BS},\gamma})_{n,m}}{b_{n,m,\gamma}}
$$

$$
=
\sum_{p=1}^{P+C}
\beta_p
e^{j2\pi\left(f_{D,s,p}(m+\gamma M)T_O-n\Delta f\tau_{s,p}\right)}
+(\tilde{\boldsymbol{Z}}_{\mathrm{BS},\gamma})_{n,m}
$$

The resulting $\boldsymbol{F}_{\mathrm{BS},\gamma}$ are the OFDM channel symbols used by the sensing pipeline.

## Clutter Suppression

Continuous frames are concatenated along slow time:

$$
(\boldsymbol{F}_{\mathrm{BS}})_{n,\gamma M+m}
\triangleq
(\boldsymbol{F}_{\mathrm{BS},\gamma})_{n,m}
$$

OpenISAC can downsample slow time by the sensing stride $M_D$:

$$
(\grave{\boldsymbol{F}}_{\mathrm{BS}})_{n,m}
=
(\boldsymbol{F}_{\mathrm{BS}})_{n,mM_D}
$$

Static and near-static clutter are suppressed by an IIR high-pass MTI filter along slow time:

$$
(\tilde{\boldsymbol{F}}_{\mathrm{BS}})_{n,m}
=
\frac{1}{a_0}
\left(
\sum_{i=0}^{I} b_i(\grave{\boldsymbol{F}}_{\mathrm{BS}})_{n,m-i}
-
\sum_{j=1}^{J} a_j(\tilde{\boldsymbol{F}}_{\mathrm{BS}})_{n,m-j}
\right)
$$

This creates a notch near zero Doppler, suppressing self-interference and static clutter while preserving moving targets.

## Delay-Doppler Processing

After MTI, channel symbols are repacked into sensing frames:

$$
(\tilde{\boldsymbol{F}}_{\mathrm{BS},\gamma})_{n,0:M_s-1}
=
(\tilde{\boldsymbol{F}}_{\mathrm{BS}})_{n,\gamma M_s:(\gamma+1)M_s-1}
$$

The delay-Doppler periodogram is then

$$
(\mathrm{Per}_{\gamma})_{k_\tau,k_f}
=
\frac{1}{NM_s}
\left|
\sum_{m=0}^{M_s-1}\sum_{n=0}^{N-1}
(\tilde{\boldsymbol{F}}_{\mathrm{BS},\gamma})_{n,m}
w[n,m]
e^{j2\pi nk_\tau/N_{\mathrm{Per}}}
e^{-j2\pi mk_f/M_{\mathrm{Per}}}
\right|^2
$$

Peak locations map to delay and Doppler:

$$
\hat{\tau}=\frac{\hat{k}_\tau}{N_{\mathrm{Per}}\Delta f},
\quad
\hat{f}_D=\frac{\hat{k}_f}{M_{\mathrm{Per}}M_DT_O}
$$

For monostatic range, $R\approx c\hat{\tau}/2$.

## Micro-Doppler Processing

Micro-Doppler analysis works directly on the MTI-filtered slow-time stream. First, transform subcarriers into delay bins:

$$
(\boldsymbol{R}_{\mathrm{BS}})_{k_\tau,m}
=
\frac{1}{N}
\sum_{n=0}^{N-1}
(\tilde{\boldsymbol{F}}_{\mathrm{BS}})_{n,m}
e^{j2\pi nk_\tau/N}
$$

Choose a working delay bin $k_\tau^\star$ and define $r_{\mathrm{BS}}[m]=(\boldsymbol{R}_{\mathrm{BS}})_{k_\tau^\star,m}$. The STFT is

$$
(\boldsymbol{G})_{m,k_f}
=
\sum_{\ell=0}^{M_w-1}
r_{\mathrm{BS}}[mM_H+\ell]\,
w_{\mathrm{md}}[\ell]\,
e^{-j2\pi k_f\ell/M_{\mathrm{md}}}
$$

and the displayed spectrogram is

$$
(\mathrm{SPT})_{m,k_f}
=
\frac{1}{M_w}|(\boldsymbol{G})_{m,k_f}|^2
$$

The viewer displays this spectrum in two-sided form so that the zero-Doppler bin is centered.
