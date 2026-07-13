---
title: UE Bistatic Sensing
description: Downlink-symbol reconstruction, continuous timing/SFO compensation, and bistatic delay–Doppler interpretation.
---

UE bistatic sensing and downlink communication observe the same $h_\mathrm{DL}(t,\tau)$, but have opposite objectives: communication removes the channel, while sensing preserves its delay and slow-time structure. Relative to BS monostatic sensing, the UE must reconstruct unknown data and prevent BS–UE timing drift from moving the sensing delay axis.

## 1. Downlink-Symbol Reconstruction

ZC, pilot, and full-band channel-reference symbols are known. On data resources, hard QPSK decisions give

$$
\tilde b_{n,m,\gamma}^\mathrm{DL}
=\frac{1}{\sqrt2}
\left[
\operatorname{sgn}(\operatorname{Re}\{\hat d_{n,m,\gamma}^\mathrm{DL}\})
+j\operatorname{sgn}(\operatorname{Im}\{\hat d_{n,m,\gamma}^\mathrm{DL}\})
\right].
$$

Define the complete reconstructed grid as

$$
\tilde b_{n,m,\gamma}=
\begin{cases}
b_{n,m,\gamma}^\mathrm{DL},&(n,m)\in\Omega_\mathrm{ref}^\mathrm{DL}
\text{ or }m\in\mathcal S_\mathrm{ZC}^\mathrm{DL},\\
\tilde b_{n,m,\gamma}^\mathrm{DL},&(n,m)\in\Omega_\mathrm{data}^\mathrm{DL}.
\end{cases}
$$

Modulation removal then gives

$$
F_{n,m,\gamma}^\mathrm{UE}
=\frac{Y_{n,m,\gamma}^\mathrm{DL}}
{\tilde b_{n,m,\gamma}}.
$$

Correct decisions make $F_{n,m,\gamma}^\mathrm{UE}$ a time-frequency sample of the BS-to-UE channel. Decision errors become sparse outliers, so low-SNR sensing can retain only high-confidence decisions or known references.

## 2. Why Communication Timing Is Insufficient

Communication only requires total delay spread to remain inside the cyclic prefix. Sub-sample timing error can be absorbed into $\hat H_{n,m,\gamma}^\mathrm{DL}$, and the frame origin need not move until drift approaches one sample.

Delay itself is the bistatic sensing observable. Reusing those discrete integer corrections creates a staircase delay trajectory and artificial jumps in delay–Doppler or micro-Doppler products. Sensing must therefore reconstruct continuous timing rather than merely record communication frame-origin changes.

## 3. Fractional Timing Estimate

From the synchronization-ZC channel estimate, form

$$
p_\gamma[k]
=\frac{1}{N}\sum_{n=0}^{N-1}
\hat H_{n,m_\mathrm{sync},\gamma}^\mathrm{DL}
e^{j2\pi nk/N}.
$$

Let $k_{\max,\gamma}$ be the integer peak and define

$$
r_\gamma[q]
=\frac{p_\gamma[k_{\max,\gamma}+q]}
{p_\gamma[k_{\max,\gamma}]},
\qquad q\in\{-1,1\}.
$$

A Quinn-type estimate supplies two candidates:

$$
\hat\delta_{\tau,+}
=\frac{r_\gamma[1]}{r_\gamma[1]-1},
\qquad
\hat\delta_{\tau,-}
=\frac{r_\gamma[-1]}{1-r_\gamma[-1]}.
$$

After selecting $\hat\delta_{\tau,\gamma}$ from consistency between the real-part signs of the two candidates,

$$
\hat k_{\tau,\gamma}
=k_{\max,\gamma}+\hat\delta_{\tau,\gamma},
\qquad
\hat\tau_{o,\gamma}=\frac{\hat k_{\tau,\gamma}}{B}.
$$

This avoids a large zero-padded IFFT, but a single-frame result remains noisy and sensitive to multipath peak shape, so SFO's long-term structure is used next.

## 4. SFO Window Fit

In a window of $\Gamma_W$ frames beginning at $\gamma_w$, let $\hat k_\mathrm{TO,\gamma}$ be the integer timing corrections already applied by communication. Their in-window cumulative value is

$$
A_{\gamma_w+\ell}
=\sum_{i=0}^{\ell-1}
\hat k_\mathrm{TO,\gamma_w+i},
\qquad
\ell=0,\ldots,\Gamma_W-1.
$$

Adding them back reconstructs a continuous trajectory:

$$
\tilde k_{\tau,\gamma_w+\ell}
=\hat k_{\tau,\gamma_w+\ell}
+A_{\gamma_w+\ell}.
$$

Sampling-clock mismatch changes slowly within a short window, so fit

$$
\tilde k_{\tau,\gamma_w+\ell}
\approx
\epsilon_\mathrm{SIO,w}\ell+k_{\tau,\gamma_w}.
$$

The slope

$$
\epsilon_\mathrm{SIO,w}
=MN_sB\,\Delta T_{as,w}
$$

is the SIO-induced delay drift in samples per frame, with $\Delta T_{as,w}$ denoting the window-average sampling-interval error.

## 5. Continuous Sensing Timing

Update the sensing timing estimate as prediction plus applied correction plus observation feedback:

$$
\hat k_{\tau,\gamma}^\mathrm{sens}
=\hat k_{\tau,\gamma-1}^\mathrm{sens}
+\hat\epsilon_\mathrm{SIO,w-1}
-\hat k_\mathrm{TO,\gamma-1}
+\mu_\gamma e_\gamma,
$$

$$
e_\gamma
=\hat k_{\tau,\gamma}
-\hat k_{\tau,\gamma-1}^\mathrm{sens}.
$$

$\hat\epsilon_\mathrm{SIO,w-1}$ predicts the next-frame drift, $-\hat k_\mathrm{TO,\gamma-1}$ removes the coordinate change caused by moving the communication frame origin, and $\mu_\gamma e_\gamma$ prevents long-term prediction error from accumulating. A small $\mu_\gamma$ suppresses noise in steady state; persistent error can be corrected with greater feedback weight.

## 6. Timing and SFO Phase Compensation

$\hat k_{\tau,\gamma}^\mathrm{sens}$ is in samples, so its value in seconds is $\hat k_{\tau,\gamma}^\mathrm{sens}/B$. A dimensionally consistent compensation is

$$
\tilde F_{n,m,\gamma}^\mathrm{UE}
=F_{n,m,\gamma}^\mathrm{UE}
\exp\!\left\{
j2\pi\kappa_n\Delta f
\left(
\frac{\hat k_{\tau,\gamma}^\mathrm{sens}}{B}
+mN_s\Delta\hat T_{as,w-1}
\right)
\right\}.
$$

Positive delay produces a negative frequency-domain phase slope, hence the positive compensation exponent. The compensated symbols are concatenated in slow time and processed with the same clutter suppression, delay–Doppler, and micro-Doppler operations as monostatic sensing. A single UE observation has no ULA angle dimension.

## 7. Bistatic Interpretation

After LoS referencing,

$$
\Delta\tau_l
=\frac{d_{B,l}+d_{l,U}-d_{B,U}}{c},
\qquad
\Delta d_l=c\Delta\tau_l.
$$

The result is excess path length relative to the LoS, not monostatic range $c\tau/2$. Absolute propagation delay requires a known BS–UE baseline or an additional bidirectional timing constraint.

The OTA method is most reliable while the LoS or another stable dominant path remains visible. In rich NLoS conditions, dominant-peak switching changes the reference and calls for longer averaging, path-continuity constraints, or the additional constraint from [eRTM bidirectional timing](/docs/signal-processing/ota-ertm-timing/).
