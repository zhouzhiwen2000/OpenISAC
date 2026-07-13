---
title: Signal Model
description: A unified model for bidirectional OFDM communication, multichannel ULA monostatic sensing, and UE bistatic sensing.
---

The current OpenISAC system contains a BS-to-UE downlink, a UE-to-BS uplink, $R$ monostatic sensing channels at the BS, and one bistatic observation at the UE. The BS downlink waveform is both the communication signal and the sensing illumination; the UE uses the same received waveform for downlink decoding and bistatic sensing.

## Bidirectional Communication Channels

Let $x\in\{\mathrm{DL},\mathrm{UL}\}$ denote the downlink or uplink. Its time-varying multipath channel is

$$
h_x(t,\tau)=
\sum_{l=1}^{L_x}
\alpha_l^x
\delta\!\left(\tau-\tau_l^x-\tau_d^x(t)\right)
e^{j2\pi(f_{D,l}^x+\Delta f_c^x)t}.
$$

$L_x$ is the number of resolvable paths. For path $l$, $\alpha_l^x$, $\tau_l^x$, and $f_{D,l}^x$ are its complex coefficient, propagation delay, and Doppler shift. The link timing offset is $\tau_d^x(t)$ and the carrier-frequency offset is $\Delta f_c^x$. The received signals are

$$
y_\mathrm{UE}^\mathrm{DL}(t)
=\int h_\mathrm{DL}(t,\tau)s_\mathrm{DL}(t-\tau)\,d\tau
+z_\mathrm{UE}(t),
$$

$$
y_\mathrm{BS}^\mathrm{UL}(t)
=\int h_\mathrm{UL}(t,\tau)s_\mathrm{UL}(t-\tau)\,d\tau
+z_\mathrm{BS}^\mathrm{UL}(t).
$$

The model does not require $h_\mathrm{DL}=h_\mathrm{UL}$. TDD may exploit propagation reciprocity within the channel coherence time, but timing offsets, frequency offsets, and endpoint responses still require separate estimation. FDD uses two carriers and therefore treats their frequency responses separately.

## Multichannel BS Monostatic Channel

Model the $R$ BS sensing channels as a uniform linear array (ULA) with element spacing $d_a$. Let $\lambda=c/f_c$ be the downlink wavelength and measure $\theta$ from array broadside. The steering vector is

$$
\boldsymbol a(\theta)=
\begin{bmatrix}
1 & e^{j\mu(\theta)} & \cdots & e^{j(R-1)\mu(\theta)}
\end{bmatrix}^{T},
\qquad
\mu(\theta)=\frac{2\pi d_a}{\lambda}\sin\theta.
$$

Let $Q=P+C$ denote $P$ moving-target components and $C$ static or near-static clutter components. The array-valued channel is

$$
\boldsymbol h_\mathrm{BS}^{\mathrm{sens}}(t,\tau)
=\sum_{p=1}^{Q}
\beta_p\boldsymbol a(\theta_p)
\delta(\tau-\tau_{s,p})
e^{j2\pi f_{D,s,p}t},
$$

and the received vector is

$$
\boldsymbol y_\mathrm{BS}^{\mathrm{sens}}(t)
=\int
\boldsymbol h_\mathrm{BS}^{\mathrm{sens}}(t,\tau)
s_\mathrm{DL}(t-\tau)\,d\tau
+\boldsymbol z_\mathrm{BS}^{\mathrm{sens}}(t),
$$

where $\boldsymbol z_\mathrm{BS}^{\mathrm{sens}}(t)\sim\mathcal{CN}(\boldsymbol0,\sigma^2\boldsymbol I_R)$. For a point target at range $r_p$, radial velocity $v_p$, and radar cross section $\sigma_{\mathrm{RCS},p}$,

$$
\beta_p=
\sqrt{\frac{c^2\sigma_{\mathrm{RCS},p}}
{(4\pi)^3r_p^4f_c^2}}e^{j\phi_p},
\qquad
\tau_{s,p}=\frac{2r_p}{c},
\qquad
f_{D,s,p}=\frac{2v_pf_c}{c}.
$$

Positive $v_p$ denotes an approaching target. Monostatic range and velocity follow from $r=c\tau/2$ and $v=cf_D/(2f_c)$.

## UE Bistatic Geometry

For a BS-to-UE path through scatterer $l$, let $d_{B,l}$, $d_{l,U}$, and $d_{B,U}$ denote the BS-to-scatterer, scatterer-to-UE, and BS-to-UE distances. Then

$$
\tau_l^\mathrm{bi}=\frac{d_{B,l}+d_{l,U}}{c},
\qquad
\tau_\mathrm{LoS}=\frac{d_{B,U}}{c}.
$$

After using the line-of-sight path as the timing reference, bistatic sensing measures

$$
\Delta\tau_l^\mathrm{bi}
=\tau_l^\mathrm{bi}-\tau_\mathrm{LoS},
\qquad
\Delta d_l^\mathrm{bi}=c\Delta\tau_l^\mathrm{bi}.
$$

A fixed bistatic delay therefore defines an ellipse with the BS and UE as its foci, rather than a monostatic range circle. Low-Doppler components are normally treated as clutter, while dynamic multipath components represent bistatic scatterers.

## Sampling-Clock Mismatch

The nominal sampling interval is $T_s=1/B$. If receiver $q$ uses $T_{s,q}=T_s-\Delta T_{s,q}$, one communication path is approximately

$$
y_{q,l}[k]\approx
\alpha_l
s\!\left(kT_s-\tau_l-\tau_d[0]-k\Delta T_{s,q}\right)
e^{j2\pi(f_{D,l}+\Delta f_c)kT_s}.
$$

This approximation neglects the second-order term $(f_{D,l}+\Delta f_c)\Delta T_{s,q}$. It separates three effects: a fixed timing offset moves the frame origin, CFO produces a common phase rotation over time, and SFO creates both accumulating timing drift and a subcarrier-dependent phase slope.
