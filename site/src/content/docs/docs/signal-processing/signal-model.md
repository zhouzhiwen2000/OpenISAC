---
title: Signal Model
description: OFDM communication, monostatic sensing, and bistatic sensing models used by OpenISAC.
---

OpenISAC uses one continuous OFDM waveform for downlink communication, BS-side monostatic sensing, and UE-side bistatic sensing. The current public platform is a SISO prototype: one BS transmit path, one BS sensing receive path, and one UE receive path.

![OpenISAC system and channel model](/images/OpenISAC_SystemModel.png)

The BS transmits continuous OFDM frames:

$$
s(t)=\sum_{\gamma=0}^{\infty}s_\gamma(t-\gamma T_F)
$$

where $T_F=MT_O$, $M$ is the number of OFDM symbols per frame, and $T_O=T+T_\mathrm{CP}$ is the OFDM symbol duration including cyclic prefix.

For frame $\gamma$, the baseband waveform is

$$
s_\gamma(t)=
\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}
b_{n,m,\gamma}
e^{j2\pi n\Delta f(t-mT_O-T_\mathrm{CP})}
\cdot
\operatorname{rect}\!\left(\frac{t-mT_O}{T_O}\right)
$$

Here $b_{n,m,\gamma}$ is the constellation symbol on subcarrier $n$ and OFDM symbol $m$, $N$ is the FFT size, and $\Delta f$ is the subcarrier spacing.

## BS-UE Channel

The downlink communication channel observed at the UE is also the bistatic sensing channel:

$$
h_\mathrm{UE}(t,\tau)
=
\sum_{l=1}^{L}
\alpha_l
\delta(\tau-\tau_l-\tau_d)
e^{j2\pi(f_{D,l}+\Delta f_c)t}
$$

where $\alpha_l$, $\tau_l$, and $f_{D,l}$ are the complex coefficient, delay, and Doppler shift of path $l$. The terms $\tau_d$ and $\Delta f_c$ capture BS-UE timing offset and carrier-frequency offset.

For bistatic sensing, low- or zero-Doppler components are treated as clutter, while dynamic multipath components are interpreted as scatterers.

## Monostatic Channel

The BS-side monostatic sensing channel is modeled separately:

$$
h_\mathrm{BS}(t,\tau)
=
\sum_{p=1}^{P+C}
\beta_p
\delta(\tau-\tau_{s,p})
e^{j2\pi f_{D,s,p}t}
$$

where $P$ is the number of dynamic targets and $C$ is the number of near-static clutter echoes. Unlike the BS-UE link, this model has no BS-UE timing offset or carrier offset because the monostatic transmitter and sensing receiver are co-located at the BS.

For a reflection with range $d_p$, radial velocity $v_p$, carrier frequency $f_c$, and radar cross section $\sigma_{\mathrm{RCS},p}$:

$$
\beta_p=
\sqrt{\frac{c^2\sigma_{\mathrm{RCS},p}}{(4\pi)^3d_p^4f_c^2}}
e^{j\phi_p},
\quad
\tau_{s,p}=\frac{2d_p}{c},
\quad
f_{D,s,p}=\frac{2v_pf_c}{c}
$$

This gives the usual monostatic range relationship:

$$
R\approx\frac{c\tau}{2}
$$

## Received Samples

The BS sensing receiver samples with the same nominal clock as the transmitter:

$$
y_\mathrm{BS}[k]
=
\sum_{p=1}^{P+C}
\beta_p s(kT_s-\tau_{s,p})
e^{j2\pi f_{D,s,p}kT_s}
+z_\mathrm{BS}[k]
$$

The UE samples with a possibly different sampling interval $T_{s,\mathrm{UE}}=T_s-\Delta T_s$:

$$
y_{\mathrm{UE}}[k]
=
\sum_{l=1}^{L}y_l[k]+z_{\mathrm{UE}}[k]
$$

with the $l$th path approximated as

$$
y_l[k]\approx
\alpha_l
s(kT_s-\tau_l-\tau_d-k\Delta T_s)
e^{j2\pi(f_{D,l}+\Delta f_c)kT_s}
$$

This approximation ignores the small cross term between Doppler/CFO and sampling-interval offset. It is the starting point for the UE timing, CFO, SFO, and bistatic sensing compensation steps.
