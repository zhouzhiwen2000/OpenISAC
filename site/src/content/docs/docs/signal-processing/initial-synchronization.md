---
title: Initial Synchronization
description: UE initial timing and frequency acquisition from the downlink ZC, optional second synchronization symbol, and CFO training field.
---

The BS is the time/frequency reference, and the UE performs initial synchronization from the downlink. In `SYNC_SEARCH`, it locates a complete frame, estimates initial timing and frequency offsets, and applies their corrections. After entering `NORMAL`, the downlink and uplink use their own channel references and pilots to estimate residual CFO/SFO. The uplink ZC therefore supports BS-side channel estimation and residual alignment rather than an independent initial-acquisition procedure.

Let $s_\mathrm{ZC}[k]$ be the main time-domain ZC symbol including the cyclic prefix, with length $N_s$, and let $m_\mathrm{sync}$ be its position in the frame. OpenISAC supports compact one-ZC acquisition and a robust path enhanced by an optional second synchronization symbol and CFO training field.

## 1. Main-ZC Timing Detection

For candidate start $u$, the UE computes

$$
\Lambda_\mathrm{ZC}[u]
=\frac{
\left|\sum_{k=0}^{N_s-1}y_\mathrm{UE}[u+k]s_\mathrm{ZC}^{*}[k]\right|^2
}{
\left(\sum_{k=0}^{N_s-1}|y_\mathrm{UE}[u+k]|^2\right)
\left(\sum_{k=0}^{N_s-1}|s_\mathrm{ZC}[k]|^2\right)
}.
$$

With search region $\mathcal U$, let $\hat k_\mathrm{peak}=\arg\max_{u\in\mathcal U}\Lambda_\mathrm{ZC}[u]$. The implementation checks the peak-to-average ratio

$$
\rho_\mathrm{ZC}
=\frac{\Lambda_\mathrm{ZC}[\hat k_\mathrm{peak}]}
{\frac{1}{|\mathcal U|}\sum_{u\in\mathcal U}\Lambda_\mathrm{ZC}[u]}.
$$

After a valid detection, the integer frame-timing estimate is

$$
\hat k_\mathrm{TO}
=\hat k_\mathrm{peak}-m_\mathrm{sync}N_s-N_\mathrm{lag}.
$$

$N_\mathrm{lag}<N_\mathrm{CP}$ preserves margin before the strongest peak for earlier multipath arrivals. One-ZC mode has the lowest overhead, but a large initial CFO rotates phase across the correlation window and can weaken the ZC peak, motivating the optional fields below.

## 2. Optional Second Synchronization Symbol

When enabled, identical ZC OFDM symbols occupy $m_\mathrm{sync}-1$ and $m_\mathrm{sync}$. For trial start $u$, the UE evaluates

$$
P_\mathrm{SC}[u]
=\sum_{k=0}^{N-1}
y_\mathrm{UE}^{*}[u+N_\mathrm{CP}+k]
y_\mathrm{UE}[u+N_s+N_\mathrm{CP}+k],
$$

$$
R_\mathrm{SC}[u]
=\sum_{k=0}^{N-1}
\left|y_\mathrm{UE}[u+N_s+N_\mathrm{CP}+k]\right|^2,
\qquad
\Lambda_\mathrm{SC}[u]
=\frac{|P_\mathrm{SC}[u]|^2}{R_\mathrm{SC}^2[u]}.
$$

The coarse start and modulo CFO are

$$
\hat u=\arg\max_u\Lambda_\mathrm{SC}[u],
\qquad
\hat f_{o,\mathrm{mod}}
=\frac{\angle P_\mathrm{SC}[\hat u]}{2\pi N_sT_s}.
$$

The unambiguous range is $\pm1/(2N_sT_s)$. Within the configured CFO search range, the candidates are

$$
f_a=\hat f_{o,\mathrm{mod}}+\frac{a}{N_sT_s},
\qquad a\in\mathbb Z.
$$

The UE derotates the samples with each $f_a$ and evaluates local ZC correlation near the expected main ZC. Without the CFO training field, it selects the strongest candidate. The main ZC still refines timing with the previous section's equation; the repeated symbol does not replace fine ZC timing.

## 3. Optional CFO Training Field

The CFO field occupies $m_\mathrm{sync}+1$, with useful part

$$
s_\mathrm{CFO}[k+N_\mathrm{CFO}]
=s_\mathrm{CFO}[k].
$$

If $u_\mathrm{CFO}$ is its start, the independent CFO reference is

$$
\hat f_\mathrm{CFO,tr}
=\frac{1}{2\pi N_\mathrm{CFO}T_s}
\angle\!\left(
\sum_{k=0}^{N-N_\mathrm{CFO}-1}
y_\mathrm{UE}^{*}[u_\mathrm{CFO}+N_\mathrm{CP}+k]
y_\mathrm{UE}[u_\mathrm{CFO}+N_\mathrm{CP}+k+N_\mathrm{CFO}]
\right).
$$

Its unambiguous range is $\pm1/(2N_\mathrm{CFO}T_s)$. This estimate selects the CFO candidate closest to $\hat f_\mathrm{CFO,tr}$; it replaces neither main-ZC timing nor the subsequent CP-tail refinement. The two optional fields are independent: the second ZC improves coarse acquisition under large CFO, while the CFO field improves ambiguity resolution.

## 4. CP-Tail CFO Estimation and Ambiguity Resolution

When the second synchronization symbol is disabled or not detected, the UE first locates the frame from the main ZC and coherently accumulates cyclic-prefix/tail correlations over complete OFDM symbols in the current block. For available-symbol set $\mathcal M_\mathrm{CP}$,

$$
r_\mathrm{CP}
=\sum_{m\in\mathcal M_\mathrm{CP}}
\sum_{i=0}^{N_\mathrm{CP}-1}
y_\mathrm{UE}^{*}[u_m+i]
y_\mathrm{UE}[u_m+N+i],
$$

$$
\hat f_{o,\mathrm{CP,mod}}
=\frac{\angle r_\mathrm{CP}}{2\pi NT_s}.
$$

$\hat f_{o,\mathrm{CP,mod}}$ gives only the CFO modulo $\Delta f=1/(NT_s)$ and cannot identify the CFO by itself. For configured search range $\mathcal F_\mathrm{search}$, the candidate CFO set is

$$
\mathcal F_\mathrm{CFO}
=\left\{
\hat f_{o,\mathrm{CP,mod}}+a\Delta f
\;\middle|\;
a\in\mathbb Z,
\ \hat f_{o,\mathrm{CP,mod}}+a\Delta f
\in\mathcal F_\mathrm{search}
\right\}.
$$

The UE derotates the received samples with each candidate CFO and evaluates local correlation near the expected main-ZC position. Without the CFO training field, it selects the candidate CFO that produces the strongest ZC peak. With the CFO field, it selects the candidate CFO closest to $\hat f_\mathrm{CFO,tr}$. The selected value is the initial CFO estimate after ambiguity resolution.

When the second synchronization symbol is used, its inter-ZC phase gives the modulo CFO and the candidate CFOs are separated by $1/(N_sT_s)$. The UE again selects one candidate CFO using local ZC correlation or the CFO training field, then uses CP-tail correlation to estimate the residual frequency offset around that value. The second-ZC and main-ZC-only paths therefore construct their candidate CFO sets differently, but also use CP-tail correlation to improve the frequency estimate.

## 5. Correction and State Transition

After determining $\hat k_\mathrm{TO}$, the UE adjusts the next receive block's sample acquisition/discard position to update the downlink demodulation/FFT window so that the dominant path appears at the configured target peak position. The updated window directly becomes the current reference for $\tau_d^\mathrm{UE}$. The initial CFO is removed by digital frequency correction or reference-clock adjustment. The receiver then enters `NORMAL`, where [Downlink Communication](/docs/signal-processing/ue-reception/) performs channel estimation and residual CFO/SFO tracking and compensation. If detection confidence or residual validity is lost, the receiver returns to `SYNC_SEARCH`.
