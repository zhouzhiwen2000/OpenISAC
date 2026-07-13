---
title: OFDM 资源
description: 连续 OFDM 波形、TDD/FDD 帧划分以及上下行参考与数据资源。
---

OpenISAC 的上下行共享同一套 OFDM 数字参数，但使用彼此独立的资源栅格与 ZC 根序列。令 $N$ 为 FFT 大小，$N_\mathrm{CP}$ 为循环前缀采样数，$\Delta f$ 为子载波间隔，则

$$
T=\frac{1}{\Delta f},\qquad
B=N\Delta f,\qquad
T_s=\frac{1}{B},
$$

$$
N_s=N+N_\mathrm{CP},\qquad
T_\mathrm{CP}=N_\mathrm{CP}T_s,\qquad
T_O=N_sT_s=T+T_\mathrm{CP}.
$$

一帧包含 $M$ 个 OFDM 符号，帧长为 $T_F=MT_O$。$n\in\{0,\ldots,N-1\}$ 是 FFT 存储位置，对应的有符号子载波索引记为 $\kappa_n$，其基带频率为 $f_n=\kappa_n\Delta f$。$\operatorname{rect}(\cdot)$ 表示持续一个 $T_O$ 的矩形符号窗。

## 连续 OFDM 波形

对链路 $x\in\{\mathrm{DL},\mathrm{UL}\}$，连续发送信号写为

$$
s_x(t)=\sum_{\gamma=0}^{\infty}s_{x,\gamma}(t-\gamma T_F),
$$

$$
s_{x,\gamma}(t)=
\sum_{m=0}^{M-1}\sum_{n=0}^{N-1}
b_{n,m,\gamma}^{x}
e^{j2\pi\kappa_n\Delta f(t-mT_O-T_\mathrm{CP})}
\operatorname{rect}\!\left(\frac{t-mT_O}{T_O}\right).
$$

$\boldsymbol B_\gamma^x=[b_{n,m,\gamma}^x]\in\mathbb C^{N\times M}$ 是第 $\gamma$ 帧的发送资源栅格；不属于链路 $x$ 的资源取零。连续且等间隔的 OFDM 符号为慢时间处理提供均匀采样，避免不规则分组间隔给多普勒和微多普勒谱引入额外旁瓣。

## TDD 与 FDD 资源集合

记 $\mathcal M=\{0,\ldots,M-1\}$。TDD 将每一帧划分为互不重叠的集合

$$
\mathcal M
=\mathcal S_\mathrm{DL}\,\dot\cup\,
\mathcal S_\mathrm{G}\,\dot\cup\,
\mathcal S_\mathrm{UL},
$$

其中 $\mathcal S_\mathrm{DL}$、$\mathcal S_\mathrm{G}$ 和 $\mathcal S_\mathrm{UL}$ 分别为下行、保护和上行符号。上行有效区间在保护符号之后开始，并将第一个有效上行符号作为上行局部帧的 $m=0$。

FDD 在独立载波上同时保持两条链路连续，因此

$$
\mathcal S_\mathrm{DL}=\mathcal S_\mathrm{UL}=\mathcal M,
\qquad
\mathcal S_\mathrm{G}=\varnothing.
$$

下文以 $\mathcal S_x$ 表示链路 $x$ 的有效符号集合。两种双工方式只改变资源在时间与频率上的占用，不改变后续 FFT、信道估计、均衡和软解调的数学形式。

## 同步、参考与数据资源

令 $\mathcal P\subset\{0,\ldots,N-1\}$ 为梳状导频子载波集合，$\mathcal D=\{0,\ldots,N-1\}\setminus\mathcal P$ 为数据子载波集合。对每条链路分别定义：

- $\mathcal S_\mathrm{ZC}^x$：全带宽 ZC 同步符号集合；
- $\mathcal S_\mathrm{CFO}^x$：可选的重复 CFO 训练符号集合；
- $\Omega_\mathrm{ref}^x$：导频和可选全带宽信道参考资源；
- $\Omega_\mathrm{data}^x$：承载编码 QPSK 的数据资源。

资源符号统一写成

$$
b_{n,m,\gamma}^{x}=
\begin{cases}
z_n^{x},
&m\in\mathcal S_\mathrm{ZC}^{x},\\[2pt]
c_n^{x,\mathrm{CFO}},
&m\in\mathcal S_\mathrm{CFO}^{x},\\[2pt]
p_{n,m}^{x},
&(n,m)\in\Omega_\mathrm{ref}^{x},\\[2pt]
d_{n,m,\gamma}^{x},
&(n,m)\in\Omega_\mathrm{data}^{x},\\[2pt]
0,&m\notin\mathcal S_x.
\end{cases}
$$

$z_n^x$ 为链路专用的恒模 ZC 序列，$p_{n,m}^x$ 为已知导频，$c_n^{x,\mathrm{CFO}}$ 为可选 CFO 训练符号。数据符号来自归一化 QPSK 星座

$$
\mathcal A_\mathrm{QPSK}
=\left\{\frac{1}{\sqrt2}(\pm1\pm j)\right\}.
$$

当前下行每帧至少包含一个全带宽 ZC 符号，并可增加第二个 ZC、CFO 训练字段和帧中信道参考符号；当前上行在第一个有效上行符号放置一个全带宽 ZC，后续符号使用梳状导频与 QPSK 数据。两条链路采用不同的 ZC 根，便于区分各自的参考信号。

## 感知资源

BS 单站感知使用下行已知符号集合

$$
\Omega_\mathrm{sens}
\subseteq
\{(n,m):m\in\mathcal S_\mathrm{DL}\}.
$$

同步、导频、通信数据和无载荷时的随机 QPSK 都可以参与感知，只要接收端能够获得对应的发送符号。恒模 ZC/QPSK 使逐元素去调制不会因符号幅度过小而显著放大噪声。若只选择部分感知资源，应保持慢时间采样间隔已知；均匀间隔资源可直接使用 Doppler FFT，非均匀资源则需要按实际采样时刻处理。

## 分辨率与无模糊范围

使用连续 $N$ 个子载波时，基本时延分辨率为

$$
\Delta\tau=\frac{1}{B}.
$$

因此单站距离分辨率为 $\Delta r_\mathrm{mono}=c/(2B)$，以 LoS 为参考的双站超额路径分辨率为 $\Delta d_\mathrm{bi}=c/B$。循环前缀给出无符号间干扰的最大时延扩展边界；更长的有效带宽改善时延分辨率，而更长的相干慢时间观测改善多普勒分辨率。
