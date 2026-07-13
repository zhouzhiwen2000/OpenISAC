---
title: 信号模型
description: OpenISAC 使用的 OFDM 通信、单站感知和双站感知模型。
---

OpenISAC 使用同一套连续 OFDM 波形支持下行通信、BS 侧单站感知和 UE 侧双站感知。当前公开平台是 SISO 原型：一个 BS 发射通道、一个 BS 感知接收通道和一个 UE 接收通道。

![OpenISAC 系统与信道模型](/images/OpenISAC_SystemModel.png)

BS 连续发送 OFDM 帧：

$$
s(t)=\sum_{\gamma=0}^{\infty}s_\gamma(t-\gamma T_F)
$$

其中 $T_F=MT_O$，$M$ 是每帧 OFDM 符号数，$T_O=T+T_\mathrm{CP}$ 是包含循环前缀的 OFDM 符号时长。

第 $\gamma$ 帧的基带波形为：

$$
s_\gamma(t)=
\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}
b_{n,m,\gamma}
e^{j2\pi n\Delta f(t-mT_O-T_\mathrm{CP})}
\cdot
\operatorname{rect}\!\left(\frac{t-mT_O}{T_O}\right)
$$

这里 $b_{n,m,\gamma}$ 是第 $n$ 个子载波、第 $m$ 个 OFDM 符号上的星座符号，$N$ 为 FFT 大小，$\Delta f$ 为子载波间隔。

## BS-UE 信道

UE 观察到的下行通信信道也是双站感知信道：

$$
h_\mathrm{UE}(t,\tau)
=
\sum_{l=1}^{L}
\alpha_l
\delta(\tau-\tau_l-\tau_d)
e^{j2\pi(f_{D,l}+\Delta f_c)t}
$$

其中 $\alpha_l$、$\tau_l$、$f_{D,l}$ 分别为第 $l$ 条路径的复系数、时延和多普勒频移。$\tau_d$ 和 $\Delta f_c$ 表示 BS-UE 时序偏移和载波频偏。

在双站感知中，低多普勒或零多普勒分量被视为 clutter，动态多径分量被解释为有效散射体。

## 单站信道

BS 侧单站感知信道单独建模为：

$$
h_\mathrm{BS}(t,\tau)
=
\sum_{p=1}^{P+C}
\beta_p
\delta(\tau-\tau_{s,p})
e^{j2\pi f_{D,s,p}t}
$$

其中 $P$ 是动态目标数，$C$ 是近静态 clutter 回波数。与 BS-UE 链路不同，单站模型不包含 BS-UE 时序偏移或载波偏移，因为发射端和感知接收端同在 BS。

对于距离 $d_p$、径向速度 $v_p$、载频 $f_c$、RCS 为 $\sigma_{\mathrm{RCS},p}$ 的反射：

$$
\beta_p=
\sqrt{\frac{c^2\sigma_{\mathrm{RCS},p}}{(4\pi)^3d_p^4f_c^2}}
e^{j\phi_p},
\quad
\tau_{s,p}=\frac{2d_p}{c},
\quad
f_{D,s,p}=\frac{2v_pf_c}{c}
$$

对应的单站距离关系为：

$$
R\approx\frac{c\tau}{2}
$$

## 接收采样

BS 感知接收端与发射端使用相同名义采样时钟：

$$
y_\mathrm{BS}[k]
=
\sum_{p=1}^{P+C}
\beta_p s(kT_s-\tau_{s,p})
e^{j2\pi f_{D,s,p}kT_s}
+z_\mathrm{BS}[k]
$$

UE 可能使用略有差异的采样间隔 $T_{s,\mathrm{UE}}=T_s-\Delta T_s$：

$$
y_{\mathrm{UE}}[k]
=
\sum_{l=1}^{L}y_l[k]+z_{\mathrm{UE}}[k]
$$

第 $l$ 条路径可近似为：

$$
y_l[k]\approx
\alpha_l
s(kT_s-\tau_l-\tau_d-k\Delta T_s)
e^{j2\pi(f_{D,l}+\Delta f_c)kT_s}
$$

该近似忽略了多普勒/CFO 与采样间隔偏移之间的小交叉项，是 UE 时序、CFO、SFO 和双站感知补偿的起点。
