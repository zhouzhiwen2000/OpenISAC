---
title: OTA and eRTM Timing
description: OTA timing compensation, eRTM timing terms, and MTI suppression ratio interpretation.
---

OTA timing workflows estimate and correct the relative timing terms that affect bistatic sensing. The documentation separates two related layers:

- The paper-level OTA sensing compensation, which keeps UE-side sensing delay continuous.
- The runtime eRTM timing variables, which expose semantic timing-offset terms used by the implementation.

## eRTM Timing Terms

The eRTM timing model separates configured RF delays from live runtime timing values:

$$
\tau_c
=
\tau_\mathrm{DL,RF}+\tau_\mathrm{UL,RF}-t_\mathrm{DL-UL,BS}-t_\mathrm{TA,UE}
$$

The bistatic timing relationship can then be interpreted as:

$$
\tau_\mathrm{TO,UE}
=
\frac{\tau_c-\tau_\mathrm{TO,BS-UE}}{2}
$$

$$
\tau_\mathrm{TO,BS}
=
\frac{\tau_c+\tau_\mathrm{TO,BS-UE}}{2}
$$

In implementation terms, RF-chain terms should come from YAML, while DUTI/TADV-style runtime terms should come from live control or timing state. Keep raw display values separate from semantic timing-offset variables and correction values.

## Why OTA Compensation Improves MTI

In a static bistatic scene, residual timing drift makes static clutter fail to cancel cleanly. OpenISAC uses the MTI suppression ratio (MSR) as a stability proxy:

$$
\mathrm{MSR}
=
\frac{
\sum_{m=m_\mathrm{start}}^{m_\mathrm{start}+M_\mathrm{avg}}
\sum_{n=0}^{N-1}
|(\grave{\boldsymbol{F}}_\mathrm{UE})_{n,m}|^2
}{
\sum_{m=m_\mathrm{start}}^{m_\mathrm{start}+M_\mathrm{avg}}
\sum_{n=0}^{N-1}
|(\tilde{\boldsymbol{F}}_\mathrm{UE})_{n,m}|^2
}
$$

Here $\grave{\boldsymbol{F}}_\mathrm{UE}$ is the pre-MTI bistatic TF-domain stream and $\tilde{\boldsymbol{F}}_\mathrm{UE}$ is the post-MTI stream.

For a single static component with $f_{D,1}=0$, the pre-MTI sample can be approximated as

$$
(\grave{\boldsymbol{F}}_\mathrm{UE})_{n,m}
\approx
\alpha_1
e^{-j2\pi n\Delta f(\tau_1+\bar{\tau}_{d,\gamma,mN_s})}
$$

For intuition, a two-pulse MTI would produce

$$
\begin{aligned}
(\tilde{\boldsymbol{F}}_\mathrm{UE})_{n,m}
&\approx
\alpha_1 e^{-j2\pi n\Delta f\tau_1}
\\
&\quad\cdot
\left(
e^{-j2\pi n\Delta f\bar{\tau}_{d,\gamma,mN_s}}
-
e^{-j2\pi n\Delta f\bar{\tau}_{d,\gamma,(m-1)N_s}}
\right)
\end{aligned}
$$

Taking squared magnitude cancels the common phase:

$$
|(\tilde{\boldsymbol{F}}_\mathrm{UE})_{n,m}|^2
=
4|\alpha_1|^2
\sin^2\!\left(
\pi n\Delta f
(\bar{\tau}_{d,\gamma,mN_s}
-
\bar{\tau}_{d,\gamma,(m-1)N_s})
\right)
$$

Larger inter-frame timing fluctuations therefore increase post-MTI residual energy and reduce achievable MSR. Although the runtime uses an IIR MTI filter rather than a two-pulse canceller, the same trend remains.

## Measured Stability Impact

![MSR improvement from OTA synchronization](/images/OpenISAC_MSRImprovement.png)

In the paper experiment, MSR improvement was measured under two center frequencies, $f_c\in\{2.4,3.1\}$ GHz, and two bandwidths, $B\in\{50,100\}$ MHz. The improvement is near 0 dB around zero clock error, around 10--14 dB near $\pm0.25$ ppm, and around 17--22 dB near $\pm0.5$ ppm.

The important operational interpretation is direct: when the UE clock is already well aligned, OTA compensation has little to correct; as clock error grows, continuous timing compensation prevents static-scene delay drift from leaking through MTI.
