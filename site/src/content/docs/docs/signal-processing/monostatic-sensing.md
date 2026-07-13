---
title: Multichannel Monostatic Sensing
description: BS-side multichannel modulation removal, clutter suppression, range–Doppler–angle, and micro-Doppler processing.
---

The BS illuminates the scene with its known downlink OFDM waveform and receives echoes through $R$ synchronized sensing channels. Keeping their relative complex phases allows the system to estimate angle under the ULA model in addition to range and Doppler.

## 1. Multichannel Time-Frequency Grid

Define the received vector on resource $(n,m,\gamma)$ as

$$
\boldsymbol Y_{n,m,\gamma}^\mathrm{sens}
=
\begin{bmatrix}
Y_{n,m,\gamma}^{(0)}&\cdots&Y_{n,m,\gamma}^{(R-1)}
\end{bmatrix}^{T}.
$$

When the maximum echo delay fits inside the cyclic prefix and intra-symbol Doppler is much smaller than $\Delta f$,

$$
\boldsymbol Y_{n,m,\gamma}^\mathrm{sens}
=b_{n,m,\gamma}^\mathrm{DL}
\sum_{p=1}^{P+C}
\beta_p\boldsymbol a(\theta_p)
e^{j2\pi\left(f_{D,s,p}t_{m,\gamma}
-\kappa_n\Delta f\tau_{s,p}^\mathrm{link}\right)}
+\boldsymbol Z_{n,m,\gamma},
$$

where $t_{m,\gamma}=(\gamma M+m)T_O$ and $\tau_{s,p}^\mathrm{link}=\tau_{s,p}^\mathrm{prop}+\tau_\mathrm{sens}^\mathrm{RF}$. Since $b_{n,m,\gamma}^\mathrm{DL}$ is known, modulation is removed element-wise:

$$
\boldsymbol F_{n,m,\gamma}
=\frac{\boldsymbol Y_{n,m,\gamma}^\mathrm{sens}}
{b_{n,m,\gamma}^\mathrm{DL}}
=\sum_{p=1}^{P+C}
\beta_p\boldsymbol a(\theta_p)
e^{j2\pi\left(f_{D,s,p}t_{m,\gamma}
-\kappa_n\Delta f\tau_{s,p}^\mathrm{link}\right)}
+\tilde{\boldsymbol Z}_{n,m,\gamma}.
$$

Retaining every channel forms the tensor, where $M_\mathrm{sens}$ is the sensing-symbol count per frame:

$$
\mathcal F_\gamma
\in\mathbb C^{N\times M_\mathrm{sens}\times R}.
$$

## 2. Array-Channel Calibration

Let fixed relative complex gains be

$$
\boldsymbol C=\operatorname{diag}(g_0,g_1,\ldots,g_{R-1}).
$$

The calibrated vector is

$$
\boldsymbol F_{n,m,\gamma}^{\mathrm{cal}}
=\boldsymbol C^{-1}\boldsymbol F_{n,m,\gamma}.
$$

Range and Doppler can be processed per channel, but angle estimation must use calibrated inter-channel phase. A common complex gain changes only total amplitude and phase; relative channel-phase error directly biases angle.

## 3. Slow Time and Clutter Suppression

Select uniformly spaced sensing symbols from continuous frames:

$$
\boldsymbol F[n,q]
=\boldsymbol F_{n,m_q,\gamma_q}^{\mathrm{cal}},
\qquad
t_q=qT_\mathrm{slow}.
$$

With one sample every $M_D$ OFDM symbols, $T_\mathrm{slow}=M_DT_O$. TDD uses only active downlink resources. A conventional Doppler FFT requires uniform samples; a nonuniform selection must use the actual $t_q$.

Static and near-static reflections concentrate near zero Doppler. Apply a high-pass MTI filter along $q$ for each subcarrier and array channel:

$$
\tilde{\boldsymbol F}[n,q]
=\frac{1}{a_0}
\left(
\sum_{i=0}^{I}b_i\boldsymbol F[n,q-i]
-\sum_{j=1}^{J}a_j\tilde{\boldsymbol F}[n,q-j]
\right).
$$

$\{b_i\}$ and $\{a_j\}$ are the feedforward and feedback coefficients. The filter creates a stopband around zero Doppler, suppressing fixed leakage and static clutter while retaining motion outside the notch. Near-stationary targets require a narrower notch or the unfiltered tensor.

## 4. Range–Doppler Processing

Split the slow-time stream into coherent intervals of $M_s$ samples. Let $N_\mathrm{Per}$ and $M_\mathrm{Per}$ be the delay-IFFT and Doppler-FFT lengths. With two-dimensional window $w[n,q]$, the array vector in each range–Doppler cell is

$$
\boldsymbol z_\gamma[k_\tau,k_f]
=\sum_{q=0}^{M_s-1}\sum_{n=0}^{N-1}
\tilde{\boldsymbol F}_\gamma[n,q]w[n,q]
e^{j2\pi\kappa_n k_\tau/N_\mathrm{Per}}
e^{-j2\pi qk_f/M_\mathrm{Per}}.
$$

Angle-independent array power is

$$
P_\mathrm{RD}[k_\tau,k_f]
=\frac{1}{NM_s}
\left\|\boldsymbol z_\gamma[k_\tau,k_f]\right\|_2^2.
$$

With signed, FFT-shifted Doppler index $k_f$,

$$
\hat\tau^\mathrm{link}=\frac{\hat k_\tau}{N_\mathrm{Per}\Delta f},
\qquad
\hat f_D=\frac{\hat k_f}{M_\mathrm{Per}T_\mathrm{slow}},
$$

$$
\hat\tau^\mathrm{prop}
=\hat\tau^\mathrm{link}-\tau_\mathrm{sens}^\mathrm{RF},
\qquad
\hat r=\frac{c\hat\tau^\mathrm{prop}}{2},
\qquad
\hat v=\frac{c\hat f_D}{2f_c}.
$$

Zero padding increases display sampling density, but fundamental range resolution remains set by $B$ and Doppler resolution by coherent duration $M_sT_\mathrm{slow}$.

## Resolution and Unambiguous Ranges

With contiguous bandwidth $B=N\Delta f$, delay resolution is $\Delta\tau=1/B$, giving monostatic range resolution

$$
\Delta r_\mathrm{mono}=\frac{c}{2B}.
$$

Uniform subcarrier sampling has circular delay-ambiguity period $1/\Delta f$, corresponding to monostatic range period $c/(2\Delta f)$; the interference-free echo delay should still remain within $T_\mathrm{CP}$. For slow-time interval $T_\mathrm{slow}$ and coherent length $M_s$,

$$
\Delta f_D=\frac{1}{M_sT_\mathrm{slow}},
\qquad
|f_D|<\frac{1}{2T_\mathrm{slow}}.
$$

The velocity resolution is $c\Delta f_D/(2f_c)$. Zero padding increases display sampling density but does not change these fundamental resolution or ambiguity limits.

## 5. ULA Angle Processing

Scan the ULA steering vector at a fixed $(k_\tau,k_f)$ cell:

$$
P_\mathrm{RDA}[k_\tau,k_f,\theta]
=\frac{
\left|\boldsymbol a^H(\theta)
\boldsymbol z_\gamma[k_\tau,k_f]\right|^2
}{R^2NM_s},
$$

$$
\hat\theta
=\arg\max_\theta P_\mathrm{RDA}[k_\tau,k_f,\theta].
$$

For one dominant target, calibrated array phase follows

$$
\angle z_r\approx\phi_0+r\mu,
\qquad
\mu=\frac{2\pi d_a}{\lambda}\sin\theta.
$$

After phase unwrapping and slope fitting,

$$
\hat\theta
=\arcsin\!\left(\frac{\lambda\hat\mu}{2\pi d_a}\right).
$$

$d_a\le\lambda/2$ prevents visible-region spatial aliasing. If several targets share a range–Doppler cell, one phase slope no longer represents one angle; beam scanning, a spatial FFT, or a higher-resolution array estimator is then required.

## 6. Multichannel Micro-Doppler

First transform subcarriers to per-channel range–slow-time data:

$$
\boldsymbol r[k_\tau,q]
=\frac{1}{N}\sum_{n=0}^{N-1}
\tilde{\boldsymbol F}[n,q]e^{j2\pi nk_\tau/N}.
$$

At target range cell $k_\tau^\star$, either combine channel powers or beamform toward $\theta_0$:

$$
r_{\theta_0}[q]
=\frac{1}{R}\boldsymbol a^H(\theta_0)
\boldsymbol r[k_\tau^\star,q].
$$

The short-time Fourier transform is

$$
G[u,k_f]
=\sum_{\ell=0}^{M_w-1}
r_{\theta_0}[uM_H+\ell]
w_\mathrm{md}[\ell]
e^{-j2\pi k_f\ell/M_\mathrm{md}}.
$$

$|G[u,k_f]|^2$ shows fine motion over time. Beamforming before the STFT can suppress other directions that occupy the same range cell.
