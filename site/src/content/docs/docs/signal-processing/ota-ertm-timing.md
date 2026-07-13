---
title: OTA and eRTM Timing
description: From LoS-relative timing to joint downlink/uplink absolute timing-offset estimation and its effect on clutter suppression.
---

Bistatic sensing separates two related problems:

- **OTA LoS tracking** uses a stable direct path to remove continuous UE-to-BS drift, producing delay relative to the LoS.
- **eRTM bidirectional timing** combines uplink and downlink channel delays with fixed group delays and transmit/receive timing relations to separate BS-side and UE-side offsets.

The first requires only a downlink observation and stabilizes relative delay. The second uses the current bidirectional link to construct an absolute delay reference.

## 1. Downlink/Uplink Delay-Profile Correlation

Form $L_\mathrm{os}$-times oversampled delay profiles from the downlink observed at the UE and the uplink observed at the BS:

$$
p_\mathrm{DL}[k]
=\operatorname{IFFT}_{L_\mathrm{os}N}
\{\hat H_\mathrm{DL}[n]\},
\qquad
p_\mathrm{UL}[k]
=\operatorname{IFFT}_{L_\mathrm{os}N}
\{\hat H_\mathrm{UL}[n]\}.
$$

Their circular complex cross-correlation is

$$
C[q]
=\sum_k p_\mathrm{UL}[k]
p_\mathrm{DL}^*[k-q].
$$

If $\hat q$ maximizes $|C[q]|$ and $\hat\delta_q$ is a fractional-bin refinement,

$$
\hat\tau_\mathrm{TO,BS-UE}
=\frac{\hat q+\hat\delta_q}{L_\mathrm{os}}
\quad\text{samples}.
$$

Correlation uses the complete multipath shape rather than subtracting two independent peak locations. Its sign is fixed by the order above; exchanging the downlink and uplink profiles reverses the sign.

## 2. eRTM Timing Equations

Use samples for every quantity below; division by $B$ converts to seconds. Define $\tau_\mathrm{DL,RF}$ and $\tau_\mathrm{UL,RF}$ as calibrated fixed downlink and uplink group delays, $t_\mathrm{DL-UL,BS}$ as the BS-referenced downlink/uplink timing interval, $t_\mathrm{TA,UE}$ as UE timing advance, and $\tau_\mathrm{TO,BS-UE}$ as the differential offset from delay-profile correlation.

First form

$$
\tau_c
=\tau_\mathrm{DL,RF}
+\tau_\mathrm{UL,RF}
-t_\mathrm{DL-UL,BS}
-t_\mathrm{TA,UE}.
$$

The endpoint offsets satisfy

$$
\tau_\mathrm{TO,BS}+\tau_\mathrm{TO,UE}=\tau_c,
$$

$$
\tau_\mathrm{TO,BS}-\tau_\mathrm{TO,UE}
=\tau_\mathrm{TO,BS-UE}.
$$

Therefore

$$
\boxed{
\tau_\mathrm{TO,UE}
=\frac{\tau_c-\tau_\mathrm{TO,BS-UE}}{2}
},
$$

$$
\boxed{
\tau_\mathrm{TO,BS}
=\frac{\tau_c+\tau_\mathrm{TO,BS-UE}}{2}
}.
$$

$\tau_c$ provides the sum of endpoint offsets and bidirectional correlation provides their difference, so both are required for a unique decomposition. Fixed-delay calibration error produces a common bias; correlation error mainly perturbs the difference.

## 3. From Timing Estimate to Frequency-Domain Compensation

If $\hat\tau_\mathrm{TO}$ is in samples, positive delay creates phase $-2\pi\kappa_n\Delta f\hat\tau_\mathrm{TO}/B$. Removing it gives

$$
H_\mathrm{corr}[n]
=H[n]
\exp\!\left(
j2\pi\kappa_n\Delta f
\frac{\hat\tau_\mathrm{TO}}{B}
\right).
$$

The physical timing estimate and the applied phase correction therefore have opposite signs. Integer frame-origin changes must also be added back into the observation coordinate so that absolute delay does not acquire artificial steps.

## 4. Why Timing Compensation Improves MTI

For one static bistatic path,

$$
F[n,m]
\approx
\alpha
e^{-j2\pi\kappa_n\Delta f(\tau+\tau_d[m])}.
$$

Using a two-pulse canceller to illustrate MTI,

$$
\tilde F[n,m]=F[n,m]-F[n,m-1],
$$

which gives

$$
|\tilde F[n,m]|^2
=4|\alpha|^2
\sin^2\!\left[
\pi\kappa_n\Delta f
(\tau_d[m]-\tau_d[m-1])
\right].
$$

An ideal static path cancels between adjacent slow-time samples. Residual timing drift changes its subcarrier phase and leaks energy through MTI. Continuous timing/SFO compensation reduces $\tau_d[m]-\tau_d[m-1]$ and therefore reduces static clutter residue.

The MTI suppression ratio is

$$
\mathrm{MSR}
=\frac{\sum_{m,n}|F[n,m]|^2}
{\sum_{m,n}|\tilde F[n,m]|^2}.
$$

For the same static scene and MTI filter, higher MSR indicates that compensation has made the static component more coherent and easier to reject. MSR measures long-term phase stability; it is not itself target-detection SNR.

## 5. Applicability

OTA LoS tracking requires a persistent reference path; dominant-path switching can be mistaken for clock drift. eRTM additionally requires corresponding structure in the uplink and downlink delay profiles and calibrated fixed group delays. In FDD the complex channel gains need not be reciprocal, but delay-profile correlation can still constrain timing while the principal path delays correspond. If the two carriers expose very different path sets, correlation confidence decreases.
