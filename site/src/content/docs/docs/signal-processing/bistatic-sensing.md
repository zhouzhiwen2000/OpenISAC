---
title: Bistatic Sensing
description: UE-side bistatic sensing, symbol reconstruction, timing compensation, and delay interpretation.
---

In bistatic sensing, the BS transmits and the UE observes the waveform after it propagates through the scene. The processing resembles monostatic sensing, but the UE must solve two additional problems:

- The UE does not know data symbols a priori, so data resources must be reconstructed before modulation removal.
- The BS and UE are not co-located, so timing drift directly moves the sensing delay axis.

## Symbol Reconstruction

For data subcarriers $n\in\mathcal{D}$, the UE reconstructs QPSK symbols from the equalized communication symbols:

$$
\tilde{b}_{n,m,\gamma}
=
\frac{1}{\sqrt{2}}
\left(
\operatorname{sgn}(\operatorname{Re}\{\hat{b}_{n,m,\gamma}\})
+j\operatorname{sgn}(\operatorname{Im}\{\hat{b}_{n,m,\gamma}\})
\right)
$$

For ZC synchronization symbols and pilot subcarriers, the transmitted symbols are already known and are used directly.

The UE-side bistatic channel symbols are then

$$
(\boldsymbol{F}_{\mathrm{UE},\gamma})_{n,m}
=
\frac{(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m}}
{\tilde{b}_{n,m,\gamma}}
$$

After this division step, the UE can reuse the same delay-Doppler and micro-Doppler processing structure as the BS-side monostatic path.

## Why Communication Timing Is Not Enough

Communication only requires timing good enough to avoid inter-symbol interference. If the maximum relative delay remains inside the cyclic prefix, sub-sample drift mostly appears as a channel phase term and can still be tolerated.

Sensing is different: delay is itself the measurement. Integer-sample communication timing corrections create a staircase trajectory in delay. In delay-Doppler or micro-Doppler outputs, those steps appear as artificial jumps or spectral discontinuities.

The UE sensing path therefore maintains a continuous sensing timing estimate rather than using only the sporadic communication timing correction.

## Fractional Delay Estimate

Let $k_{\max,\gamma}$ be the delay-bin index of the dominant channel peak. Define the complex ratios around that peak:

$$
r_p[k]
\triangleq
\frac{\tilde{p}_{\mathrm{delay},\gamma}[k_{\max,\gamma}+k]}
{\tilde{p}_{\mathrm{delay},\gamma}[k_{\max,\gamma}]},
\quad
k\in\{-1,1\}
$$

Quinn-style fractional estimation gives two candidates:

$$
\hat{\delta}_{\tau,1}
=
\frac{r_p[1]}{r_p[1]-1},
\quad
\hat{\delta}_{\tau,-1}
=
\frac{r_p[-1]}{1-r_p[-1]}
$$

The selected fractional offset is

$$
\hat{\delta}_{\tau}
=
\begin{cases}
\hat{\delta}_{\tau,1}, &
\hat{\delta}_{\tau,-1}>0\ \text{and}\ \hat{\delta}_{\tau,1}>0,\\
\hat{\delta}_{\tau,-1}, & \text{otherwise}.
\end{cases}
$$

The overall timing offset is

$$
\hat{\tau}_{o,\gamma}
=
\frac{\hat{k}_{\tau,\gamma}}{B}
=
\frac{\hat{\delta}_\tau+k_{\max,\gamma}}{B}
$$

## SIO Window Fit

The fractional estimates are noisy, so OpenISAC further exploits the approximately linear delay drift caused by sampling interval offset (SIO). Within a window of $\Gamma_W$ frames starting at $\gamma_w$, accumulate the integer corrections already applied by communication tracking:

$$
A_{\gamma_w+\ell}
\triangleq
\sum_{i=0}^{\ell-1}
\hat{k}_{\mathrm{TO},\gamma_w+i},
\quad
\ell=0,\ldots,\Gamma_W-1
$$

Add them back to reconstruct a continuous delay trajectory:

$$
\tilde{k}_{\tau,\gamma_w+\ell}
\triangleq
\hat{k}_{\tau,\gamma_w+\ell}+A_{\gamma_w+\ell}
$$

Then fit

$$
\tilde{k}_{\tau,\gamma_w+\ell}
\approx
\epsilon_{\mathrm{SIO},w}\ell+k_{\tau,\gamma_w}
$$

by least squares. The slope $\hat{\epsilon}_{\mathrm{SIO},w}$ is the SIO-induced delay drift per frame.

## Continuous Sensing Timing

The continuous sensing timing estimate is updated recursively:

$$
\hat{k}^{\mathrm{sens}}_{\tau,\gamma}
=
\hat{k}^{\mathrm{sens}}_{\tau,\gamma-1}
\hat{\epsilon}_{\mathrm{SIO},w-1}
-
\hat{k}_{\mathrm{TO},\gamma-1}
\mu_\gamma e_\gamma
$$

where

$$
e_\gamma
\triangleq
\hat{k}_{\tau,\gamma}
-
\hat{k}^{\mathrm{sens}}_{\tau,\gamma-1}
$$

is the tracking error. The feedback gain $\mu_\gamma$ is kept small during stable operation and can be increased when the error persists.

## Channel Compensation

The compensated bistatic sensing symbols are

$$
(\tilde{\boldsymbol{F}}_{\mathrm{UE},\gamma})_{n,m}
=
(\boldsymbol{F}_{\mathrm{UE},\gamma})_{n,m}
e^{j2\pi n\Delta f
(\hat{k}_{\tau,\gamma}^{\mathrm{sens}}+mN_s\Delta\hat{T}_{as,w-1})}
$$

These compensated symbols are then processed like the monostatic channel symbols. Under this OTA synchronization scheme, bistatic delays are relative to the LoS reference path; absolute delay recovery requires the known physical BS-UE separation.

## Practical Limits

The OTA method is most reliable when the LoS path or another stable dominant path remains visible. In rich NLoS scenes, the strongest path can switch between scatterers, biasing the recovered delay trajectory. Longer averaging, path-consistency checks, or external synchronization may be needed in those cases.
