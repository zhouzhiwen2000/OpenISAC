---
title: 单站感知
description: BS 侧 TF 信道提取、clutter 抑制、时延-多普勒处理和微多普勒处理。
---

单站感知中，BS 发送 OFDM 波形，并通过自身感知接收链路接收回波。由于 BS 已知每个发送资源符号，因此可以去除调制，得到用于雷达处理的时频信道矩阵。

![单站感知流程](/images/FlowGraph.png)

## TF 栅格映射

假设最大目标时延位于循环前缀内，且多普勒相对子载波间隔较小，第 $\gamma$ 帧 BS 接收样本可整理为 $N\times M$ 矩阵：

$$
(\boldsymbol{B}_{\mathrm{BS},\gamma})_{n,m}
=
b_{n,m,\gamma}
\sum_{p=1}^{P+C}
\beta_p
e^{j2\pi\left(f_{D,s,p}(m+\gamma M)T_O-n\Delta f\tau_{s,p}\right)}
+(\boldsymbol{Z}_{\mathrm{BS},\gamma})_{n,m}
$$

由于 $b_{n,m,\gamma}$ 已知，逐元素相除即可去除通信符号：

$$
(\boldsymbol{F}_{\mathrm{BS},\gamma})_{n,m}
=
\frac{(\boldsymbol{B}_{\mathrm{BS},\gamma})_{n,m}}{b_{n,m,\gamma}}
$$

$$
=
\sum_{p=1}^{P+C}
\beta_p
e^{j2\pi\left(f_{D,s,p}(m+\gamma M)T_O-n\Delta f\tau_{s,p}\right)}
+(\tilde{\boldsymbol{Z}}_{\mathrm{BS},\gamma})_{n,m}
$$

得到的 $\boldsymbol{F}_{\mathrm{BS},\gamma}$ 即感知流水线使用的 OFDM channel symbols。

## Clutter 抑制

连续帧沿慢时间拼接：

$$
(\boldsymbol{F}_{\mathrm{BS}})_{n,\gamma M+m}
\triangleq
(\boldsymbol{F}_{\mathrm{BS},\gamma})_{n,m}
$$

OpenISAC 可通过 sensing stride $M_D$ 对慢时间降采样：

$$
(\grave{\boldsymbol{F}}_{\mathrm{BS}})_{n,m}
=
(\boldsymbol{F}_{\mathrm{BS}})_{n,mM_D}
$$

静态和近静态 clutter 通过沿慢时间的 IIR 高通 MTI 滤波器抑制：

$$
(\tilde{\boldsymbol{F}}_{\mathrm{BS}})_{n,m}
=
\frac{1}{a_0}
\left(
\sum_{i=0}^{I} b_i(\grave{\boldsymbol{F}}_{\mathrm{BS}})_{n,m-i}
-
\sum_{j=1}^{J} a_j(\tilde{\boldsymbol{F}}_{\mathrm{BS}})_{n,m-j}
\right)
$$

该滤波器在零多普勒附近形成 notch，用于抑制自干扰和静态 clutter，同时保留运动目标。

## 时延-多普勒处理

MTI 后的 channel symbols 被重新打包为感知帧：

$$
(\tilde{\boldsymbol{F}}_{\mathrm{BS},\gamma})_{n,0:M_s-1}
=
(\tilde{\boldsymbol{F}}_{\mathrm{BS}})_{n,\gamma M_s:(\gamma+1)M_s-1}
$$

时延-多普勒 periodogram 为：

$$
(\mathrm{Per}_{\gamma})_{k_\tau,k_f}
=
\frac{1}{NM_s}
\left|
\sum_{m=0}^{M_s-1}\sum_{n=0}^{N-1}
(\tilde{\boldsymbol{F}}_{\mathrm{BS},\gamma})_{n,m}
w[n,m]
e^{j2\pi nk_\tau/N_{\mathrm{Per}}}
e^{-j2\pi mk_f/M_{\mathrm{Per}}}
\right|^2
$$

峰值位置映射为时延和多普勒：

$$
\hat{\tau}=\frac{\hat{k}_\tau}{N_{\mathrm{Per}}\Delta f},
\quad
\hat{f}_D=\frac{\hat{k}_f}{M_{\mathrm{Per}}M_DT_O}
$$

单站距离可由 $R\approx c\hat{\tau}/2$ 得到。

## 微多普勒处理

微多普勒分析直接作用于 MTI 后的慢时间流。首先将子载波变换为时延 bin：

$$
(\boldsymbol{R}_{\mathrm{BS}})_{k_\tau,m}
=
\frac{1}{N}
\sum_{n=0}^{N-1}
(\tilde{\boldsymbol{F}}_{\mathrm{BS}})_{n,m}
e^{j2\pi nk_\tau/N}
$$

选择工作时延 bin $k_\tau^\star$，并定义 $r_{\mathrm{BS}}[m]=(\boldsymbol{R}_{\mathrm{BS}})_{k_\tau^\star,m}$。STFT 为：

$$
(\boldsymbol{G})_{m,k_f}
=
\sum_{\ell=0}^{M_w-1}
r_{\mathrm{BS}}[mM_H+\ell]\,
w_{\mathrm{md}}[\ell]\,
e^{-j2\pi k_f\ell/M_{\mathrm{md}}}
$$

显示的谱图为：

$$
(\mathrm{SPT})_{m,k_f}
=
\frac{1}{M_w}|(\boldsymbol{G})_{m,k_f}|^2
$$

前端以双边频谱形式显示，使零多普勒 bin 位于中心。
