---
title: 上行通信
description: UE→BS 的紧凑 OFDM 帧、双工时域关系、信道估计、均衡与解码。
---

上行使用与下行相同的 $N$、$N_\mathrm{CP}$、$\Delta f$ 和导频位置，但采用独立的发送栅格 $\boldsymbol B_\gamma^\mathrm{UL}$ 与 ZC 根序列。它是一条完整的 UE→BS 通信链路，而不是下行数据符号的简单反向重放。记每个上行局部帧的有效符号数为 $M_\mathrm{UL}$；TDD 中 $M_\mathrm{UL}=|\mathcal S_\mathrm{UL}|$，FDD 中 $M_\mathrm{UL}=M$。

## 上行帧

在每个上行局部帧中，第一个有效 OFDM 符号为全带宽 ZC：

$$
b_{n,0,\gamma}^\mathrm{UL}=z_n^\mathrm{UL}.
$$

其余有效符号在 $n\in\mathcal P$ 上放置已知导频，在 $n\in\mathcal D$ 上放置编码 QPSK：

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
\alpha_l^\mathrm{UL}
e^{j2\pi\left[
(f_{D,l}^\mathrm{UL}+\Delta\bar f_{c,\gamma}^\mathrm{UL})
(m+\gamma M)T_O
-\kappa_n\Delta f
(\tau_l^\mathrm{UL}+\bar\tau_{d,\gamma,m}^\mathrm{UL})
\right]}.
$$

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

因此同样可通过加权线性回归分离公共相位项与子载波相位斜率，并把同步符号处的信道估计传播到各数据符号。若 $\mathcal A_\mathrm{UL}$ 为空，则当前局部帧不进行这种导频跨符号拟合。上行和下行分别估计这些量，不依赖理想互易假设。

## 均衡与信息恢复

使用 ZF 或 MMSE 系数 $G_{n,m}^\mathrm{UL}$ 后，

$$
\hat d_{n,m,\gamma}^\mathrm{UL}
=G_{n,m}^\mathrm{UL}Y_{n,m,\gamma}^\mathrm{UL}.
$$

由均衡导频残差估计 $\hat\sigma_\mathrm{eq}^2$，再按 QPSK 星座距离计算 LLR。软信息依次经过解交织、解扰和 LDPC 解码，恢复 UE 发送的信息比特。该过程与下行采用相同的软判决原则，但信道估计、频偏和噪声方差均来自上行自身的参考资源。

## 与 eRTM 的关系

上行不仅提供反向通信，也提供 $\hat H^\mathrm{UL}[n]$ 的时延信息。将其时延谱与 UE 观测到的下行时延谱联合比较，可以估计两个方向的相对定时量，并进一步分离 BS 与 UE 两侧的时偏，详见 [OTA 与 eRTM 定时](/zh-cn/docs/signal-processing/ota-ertm-timing/)。
