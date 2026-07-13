---
title: 上行通信
description: UE→BS 的紧凑 OFDM 帧、双工时域关系、信道估计、均衡与解码。
---

上行使用与下行相同的 $N$、$N_\mathrm{CP}$、$\Delta f$、帧内符号索引 $m$ 和导频位置，但采用独立的发送资源网格 $\boldsymbol B_\gamma^\mathrm{UL}$ 与 ZC 根序列。UE 已通过下行链路完成与 BS 的初始同步；上行是一条完整的 UE→BS 通信链路，而不是重新执行一套独立初始捕获或简单反向重放下行数据。TDD 中，上行资源网格在 $m\in\mathcal S_\mathrm{DL}\cup\mathcal S_\mathrm{G}$ 时置零，仅在 $m\in\mathcal S_\mathrm{UL}$ 时承载上行符号。

## 上行帧

令 $m_{\mathrm{UL},0}=\min\mathcal S_\mathrm{UL}$。帧内第一个有效上行 OFDM 符号为全带宽 ZC：

$$
b_{n,m_{\mathrm{UL},0},\gamma}^\mathrm{UL}=z_n^\mathrm{UL}.
$$

其余 $m\in\mathcal S_\mathrm{UL}\setminus\{m_{\mathrm{UL},0}\}$ 的符号在 $n\in\mathcal P$ 上放置已知导频，在 $n\in\mathcal D$ 上放置编码 QPSK：

$$
b_{n,m,\gamma}^\mathrm{UL}=
\begin{cases}
p_{n,m}^\mathrm{UL},&n\in\mathcal P,\\
d_{n,m,\gamma}^\mathrm{UL},&n\in\mathcal D.
\end{cases}
$$

TDD 中，这个紧凑帧位于保护区之后的 $\mathcal S_\mathrm{UL}$ 内；正的定时提前量 $t_\mathrm{TA,UE}$ 将 UE 波形向前移动，使其在传播后落入 BS 的上行接收窗口。FDD 中，上行在独立载波上使用 $M$ 个符号的连续完整帧，不需要 TDD 保护区。

## BS 侧频域模型

完成帧边界对齐、去除循环前缀并 FFT 后，BS 观察到

$$
Y_{n,m,\gamma}^\mathrm{UL}
=b_{n,m,\gamma}^\mathrm{UL}
H_{n,m,\gamma}^\mathrm{UL}
+Z_{n,m,\gamma}^\mathrm{UL},
$$

$$
H_{n,m,\gamma}^\mathrm{UL}
=\sum_{l=1}^{L_\mathrm{UL}}
\alpha_l^\mathrm{UL}(t_{m,\gamma}^\mathrm{UL})
e^{j2\pi\left[
(f_{D,l}^\mathrm{UL}+\Delta\bar f_{c,\gamma}^\mathrm{UL})
t_{m,\gamma}^\mathrm{UL}
-\kappa_n\Delta f
(\tau_{l,\mathrm{prop}}(t_{m,\gamma}^\mathrm{UL})
+\tau_\mathrm{UL}^\mathrm{RF}
-\tau_d^\mathrm{BS}(t_{m,\gamma}^\mathrm{UL}))
\right]}.
$$

这里 $t_{m,\gamma}^\mathrm{UL}$ 是帧内上行符号 $m$ 的实际参考时刻；$\tau_d^\mathrm{BS}(t)$ 是 BS 当前解调窗口相对于上行发射端帧边界的时变偏移。真实传播时延、上行 RF 群时延与本地观测 TO 的关系见[信号模型](/zh-cn/docs/signal-processing/signal-model/#双向通信的时延组成)。

上行 ZC 的 LS 估计为

$$
\hat H_{n,0,\gamma}^\mathrm{UL,LS}
=\frac{Y_{n,0,\gamma}^\mathrm{UL}}{z_n^\mathrm{UL}}.
$$

将其变换到时延域后，可依据循环前缀范围限制有效抽头并进行 Wiener 平滑，再变回频域。这样既保留多径结构，又降低全带宽 LS 估计中的噪声。

## 上行相位跟踪

令 $\mathcal A_\mathrm{UL}$ 为局部上行帧中相邻两个符号都包含导频的索引集合；当 $M_\mathrm{UL}\ge3$ 且所有数据符号连续时，$\mathcal A_\mathrm{UL}=\{1,\ldots,M_\mathrm{UL}-2\}$。对这些导频形成相邻符号相关：

$$
\bar R_\gamma^\mathrm{UL}[n]
=\frac{1}{|\mathcal A_\mathrm{UL}|}
\sum_{m\in\mathcal A_\mathrm{UL}}
(Y_{n,m,\gamma}^\mathrm{UL})^*
Y_{n,m+1,\gamma}^\mathrm{UL}.
$$

其解缠相位满足

$$
\arg\bar R_\gamma^\mathrm{UL}[n]
\approx
2\pi\left(
f_{o,\gamma}^\mathrm{UL}T_O
-\kappa_n\Delta fN_s\Delta T_{s,\gamma}^\mathrm{UL}
\right).
$$

因此采用与下行相同形式的加权线性回归

$$
(\hat a_\mathrm{UL},\hat b_\mathrm{UL})
=\arg\min_{a,b}\sum_{n\in\mathcal P}
|\bar R_\gamma^\mathrm{UL}[n]|^2
\left|
\operatorname{unwrap}(\arg\bar R_\gamma^\mathrm{UL}[n])-a-b\kappa_n
\right|^2,
$$

并得到

$$
\hat f_{o,\gamma}^\mathrm{UL}
=\frac{\hat a_\mathrm{UL}}{2\pi T_O},
\qquad
\Delta\hat T_{s,\gamma}^\mathrm{UL}
=-\frac{\hat b_\mathrm{UL}}{2\pi\Delta fN_s}.
$$

上行信道随后传播为

$$
\hat H_{n,m,\gamma}^\mathrm{UL}
=\hat H_{n,0,\gamma}^\mathrm{UL}
e^{j2\pi m(
\hat f_{o,\gamma}^\mathrm{UL}T_O
-\kappa_n\Delta fN_s\Delta\hat T_{s,\gamma}^\mathrm{UL})}.
$$

若 $\mathcal A_\mathrm{UL}$ 为空，则当前局部帧不进行导频跨符号拟合。CFO 与 SFO 的残差估计和补偿形式在上下行通用，但两条链路分别使用自己的参考资源与观测量；这里不依赖理想互易性来完成通信解调。

## 均衡与信息恢复

使用 ZF 或 MMSE 系数 $G_{n,m}^\mathrm{UL}$ 后，

$$
\hat d_{n,m,\gamma}^\mathrm{UL}
=G_{n,m}^\mathrm{UL}Y_{n,m,\gamma}^\mathrm{UL}.
$$

由均衡导频残差估计 $\hat\sigma_\mathrm{eq}^2$，再按 QPSK 星座距离计算 LLR。软信息依次经过解交织、解扰和 LDPC 解码，恢复 UE 发送的信息比特。该过程与下行采用相同的软判决原则，但信道估计、频偏和噪声方差均来自上行自身的参考资源。

## 与 eRTM 的关系

上行不仅提供反向通信，也提供 BS 端的上行信道估计 $\hat H_{\mathrm{BS}}[n]$。当上下行同时启用时，eRTM 可将它与 UE 端的下行信道估计 $\hat H_{\mathrm{UE}}[n]$ 联合使用，利用上下行信道间的关系进一步估计 BS 与 UE 两侧的时偏，详见[双站感知中的 eRTM 选项](/zh-cn/docs/signal-processing/bistatic-sensing/#ertm-双向定时选项)。
