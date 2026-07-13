---
title: Signal Model
description: A unified downlink, uplink, and sensing model separating propagation, RF group delay, and local demodulation boundaries.
---

OpenISAC contains three links: a BS-to-UE communication downlink, a UE-to-BS communication uplink, and a BS monostatic-sensing link. The BS downlink waveform is both the communication signal and the monostatic illumination; the UE uses its observation of that same downlink waveform for both communication decoding and bistatic sensing.

## Bidirectional-Channel Delay Components

Let $x\in\{\mathrm{DL},\mathrm{UL}\}$ denote the downlink or uplink and let $q_x$ be its receiver, with $q_\mathrm{DL}=\mathrm{UE}$ and $q_\mathrm{UL}=\mathrm{BS}$. The delay of path $l$ has three distinct layers.

The true wireless propagation delay is

$$
\tau_{l,\mathrm{prop}}^x(t),
$$

which describes only propagation through space and the scattering environment. Let $\tau_x^\mathrm{RF}$ be the fixed group delay of the combined transmit and receive RF chains for link $x$. The total physical link delay is

$$
\tau_{l,\mathrm{link}}^x(t)
=\tau_{l,\mathrm{prop}}^x(t)+\tau_x^\mathrm{RF}.
$$

Each receiver partitions OFDM symbols with its current local demodulation window. Let $\tau_d^{q_x}(t)$ be the time-varying offset of receiver $q_x$'s current demodulation window relative to link $x$'s transmitter frame boundary, defined as the demodulation-window start time minus the transmitter-frame-boundary time; a positive value means that the demodulation window follows the transmitter frame boundary. Sampling-frequency offset causes the demodulation window to drift gradually, while initial synchronization and later integer-sample corrections directly update the current window; the offset therefore varies with time.

Define TO on a local delay axis as the common displacement from true propagation delay to locally observed path delay. For the downlink and uplink,

$$
\tau_\mathrm{TO}^\mathrm{UE}(t)
\triangleq\tau_\mathrm{DL}^\mathrm{RF}-\tau_d^\mathrm{UE}(t),
\qquad
\tau_\mathrm{TO}^\mathrm{BS}(t)
\triangleq\tau_\mathrm{UL}^\mathrm{RF}-\tau_d^\mathrm{BS}(t).
$$

The delay of path $l$ observed by each receiver on its own local delay axis is therefore

$$
\tau_l^\mathrm{UE}(t)
=\tau_{l,\mathrm{prop}}(t)
+\tau_\mathrm{TO}^\mathrm{UE}(t),
$$

$$
\tau_l^\mathrm{BS}(t)
=\tau_{l,\mathrm{prop}}(t)
+\tau_\mathrm{TO}^\mathrm{BS}(t).
$$

$\tau_l^\mathrm{DL}$ and $\tau_l^\mathrm{UL}$ are physical link delays including RF group delay, whereas $\tau_l^\mathrm{UE}$ and $\tau_l^\mathrm{BS}$ are direct local-delay-axis observations. The difference between the latter and true propagation delay is the TO used throughout this document.

## Continuous-Time Receive Model

The equivalent time-varying baseband impulse response observed at receiver $q_x$ is

$$
h_x(t,\tau)=
\sum_{l=0}^{L_x-1}
\alpha_l^x(t)
e^{j2\pi(f_{D,l}^x+\Delta f_c^x)t}
\delta\!\left(
\tau-\tau_{l,\mathrm{prop}}^x(t)
-\tau_x^\mathrm{RF}
+\tau_d^{q_x}(t)
\right).
$$

$L_x$ is the number of resolvable paths. $\alpha_l^x(t)$, $\tau_{l,\mathrm{prop}}^x(t)$, and $f_{D,l}^x$ are path $l$'s complex scattering coefficient, true propagation delay, and Doppler shift; $\Delta f_c^x$ is the residual carrier-frequency offset. The received signals are

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

## TDD Uplink/Downlink Channel Relation

Let $t_{\mathrm{DL}}$ and $t_{\mathrm{UL}}$ be the downlink and uplink reference-symbol times used by eRTM. In TDD, when their separation is much shorter than the channel coherence time,

$$
\tau_{l,\mathrm{prop}}(t_{\mathrm{DL}})
\approx
\tau_{l,\mathrm{prop}}(t_{\mathrm{UL}})
\approx
\tau_{l,\mathrm{prop}},
$$

$$
\alpha_l^\mathrm{DL}(t_{\mathrm{DL}})
\approx
\alpha_l^\mathrm{UL}(t_{\mathrm{UL}})
\approx
\alpha_l,
$$

TO varies mainly with slow endpoint-clock drift, so a downlink/uplink pair that is sufficiently close in time satisfies

$$
\tau_\mathrm{TO}^\mathrm{UE}(t_{\mathrm{DL}})
\approx\tau_\mathrm{TO}^\mathrm{UE},
\qquad
\tau_\mathrm{TO}^\mathrm{BS}(t_{\mathrm{UL}})
\approx\tau_\mathrm{TO}^\mathrm{BS}.
$$

For this measurement pair, the TOs at the two nearby times are abbreviated as $\tau_\mathrm{TO}^\mathrm{UE}$ and $\tau_\mathrm{TO}^\mathrm{BS}$; this does not assume that they are equal. The downlink channel observed at the UE and the uplink channel observed at the BS are

$$
H_{\mathrm{UE}}[n]
=\sum_{l=0}^{L-1}\alpha_l
e^{-j2\pi\kappa_n\Delta f[
\tau_{l,\mathrm{prop}}
+\tau_\mathrm{TO}^\mathrm{UE}]},
$$

$$
H_{\mathrm{BS}}[n]
=\sum_{l=0}^{L-1}\alpha_l
e^{-j2\pi\kappa_n\Delta f[
\tau_{l,\mathrm{prop}}
+\tau_\mathrm{TO}^\mathrm{BS}]}.
$$

Define the endpoint TO difference as

$$
\tau_\mathrm{TO}^{\mathrm{BS-UE}}
\triangleq
\tau_\mathrm{TO}^\mathrm{BS}
-\tau_\mathrm{TO}^\mathrm{UE}
=\tau_l^\mathrm{BS}-\tau_l^\mathrm{UE}.
$$

Then,

$$
\boxed{
H_{\mathrm{BS}}[n]
\approx H_{\mathrm{UE}}[n]
e^{-j2\pi\kappa_n\Delta f
\tau_\mathrm{TO}^{\mathrm{BS-UE}}}
}.
$$

## Multichannel BS Monostatic Channel

Assume that the BS is equipped with an $R$-element uniform linear array (ULA) with element spacing $d_a$. Let $\lambda=c/f_c$ be the downlink wavelength and measure $\theta$ from array broadside. The steering vector is

$$
\boldsymbol a(\theta)=
\begin{bmatrix}
1 & e^{j\mu(\theta)} & \cdots & e^{j(R-1)\mu(\theta)}
\end{bmatrix}^{T},
\qquad
\mu(\theta)=\frac{2\pi d_a}{\lambda}\sin\theta.
$$

Let $Q=P+C$ denote $P$ moving-target components and $C$ static or near-static clutter components. Let $\tau_{s,p}^\mathrm{prop}$ be component $p$'s true round-trip propagation delay and $\tau_\mathrm{sens}^\mathrm{RF}$ the fixed monostatic transmit/receive group delay. Then

$$
\tau_{s,p}^\mathrm{link}
=\tau_{s,p}^\mathrm{prop}+\tau_\mathrm{sens}^\mathrm{RF},
$$

$$
\boldsymbol h_\mathrm{BS}^\mathrm{sens}(t,\tau)
=\sum_{p=1}^{Q}
\beta_p\boldsymbol a(\theta_p)
\delta(\tau-\tau_{s,p}^\mathrm{link})
e^{j2\pi f_{D,s,p}t}.
$$

The received vector is

$$
\boldsymbol y_\mathrm{BS}^\mathrm{sens}(t)
=\int
\boldsymbol h_\mathrm{BS}^\mathrm{sens}(t,\tau)
s_\mathrm{DL}(t-\tau)\,d\tau
+\boldsymbol z_\mathrm{BS}^\mathrm{sens}(t),
$$

where $\boldsymbol z_\mathrm{BS}^\mathrm{sens}(t)\sim\mathcal{CN}(\boldsymbol0,\sigma^2\boldsymbol I_R)$. For a point target at range $r_p$, radial velocity $v_p$, and radar cross section $\sigma_{\mathrm{RCS},p}$,

$$
\beta_p=
\sqrt{\frac{c^2\sigma_{\mathrm{RCS},p}}
{(4\pi)^3r_p^4f_c^2}}e^{j\phi_p},
\qquad
\tau_{s,p}^\mathrm{prop}=\frac{2r_p}{c},
\qquad
f_{D,s,p}=\frac{2v_pf_c}{c}.
$$

After calibrating and removing $\tau_\mathrm{sens}^\mathrm{RF}$, monostatic range and velocity follow from $r=c\tau^\mathrm{prop}/2$ and $v=cf_D/(2f_c)$. Positive $v_p$ denotes an approaching target.

## Sampling-Clock Mismatch

The nominal sampling interval is $T_s=1/B$. If receiver $q_x$ uses $T_{s,q_x}=T_s-\Delta T_{s,q_x}$, one communication path is approximately

$$
y_{q_x,l}[k]\approx
\alpha_l^x
s_x\!\left(
kT_s
-\tau_{l,\mathrm{prop}}^x
-\tau_x^\mathrm{RF}
+\tau_d^{q_x}[0]
-k\Delta T_{s,q_x}
\right)
e^{j2\pi(f_{D,l}^x+\Delta f_c^x)kT_s}.
$$

This approximation neglects the second-order term $(f_{D,l}^x+\Delta f_c^x)\Delta T_{s,q_x}$. Fixed RF group delay forms the static part of TO, while the current demodulation window's offset relative to the transmitter frame boundary forms its time-varying part and enters TO with a negative sign. CFO produces common phase rotation over time, and SFO makes TO drift slowly while producing a subcarrier-dependent phase slope.
