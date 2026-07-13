---
title: 同步、CFO 与 SFO
description: 上下行共同使用的帧定时、频偏获取、信道估计与长期跟踪方法。
---

同步的目标不是只找到一次帧起点，而是持续把三个量分开估计：整数或分数时偏、载波频偏（CFO），以及采样频偏（SFO）。对主导路径而言，接收相位可近似分解为

$$
\phi_{n,m}
\approx
2\pi f_o mT_O
-2\pi\kappa_n\Delta f
\left(\tau_o+mN_s\Delta T_s\right),
$$

其中 $f_o=f_{D,1}+\Delta f_c$ 是主导路径多普勒与残余 CFO 的合成频移，$\tau_o$ 是当前时偏。第一项沿 OFDM 符号方向变化，第二项沿子载波方向形成相位斜率，SFO 则使该斜率随时间缓慢漂移。

## 1. ZC 帧定时

令 $s_\mathrm{ZC}[k]$ 为包含循环前缀的本地 ZC 时域符号，长度为 $N_s$。接收端对候选起点 $u$ 计算归一化相关度量

$$
\Lambda_\mathrm{ZC}[u]
=
\frac{
\left|\sum_{k=0}^{N_s-1}y[u+k]s_\mathrm{ZC}^{*}[k]\right|^2
}{
\left(\sum_{k=0}^{N_s-1}|y[u+k]|^2\right)
\left(\sum_{k=0}^{N_s-1}|s_\mathrm{ZC}[k]|^2\right)
}.
$$

峰值位置 $\hat u=\arg\max_u\Lambda_\mathrm{ZC}[u]$ 给出同步符号到达位置。若该 ZC 位于帧内第 $m_\mathrm{sync}$ 个符号，则帧起点的整数时偏估计为

$$
\hat k_\mathrm{TO}
=\hat u-m_\mathrm{sync}N_s-N_\mathrm{lag},
$$

其中 $N_\mathrm{lag}$ 为最强路径之前保留的多径余量。峰值相对搜索区平均值的比值可作为检测置信度，避免在纯噪声或弱相关峰上建立错误定时。

## 2. 粗 CFO 获取

当初始 CFO 可能较大时，可使用相隔 $D$ 个采样的重复训练结构：

$$
P_D[u]=\sum_{k=0}^{N-1}y^*[u+k]y[u+D+k],
$$

$$
\hat f_{o,\mathrm{mod}}
=\frac{\angle P_D[\hat u]}{2\pi DT_s}.
$$

该估计在 $1/(DT_s)$ 周期内存在整数模糊，可结合 ZC 相关峰或另一段重复结构选择正确候选。频偏进入可跟踪范围后，再由循环前缀相关和导频相位细化。单 ZC 方案省去额外训练开销，适合初始 CFO 已受控的场景。

## 3. 初始信道与时延谱

去除循环前缀并完成 FFT 后，任一通信链路满足

$$
Y_{n,m,\gamma}^{x}
=b_{n,m,\gamma}^{x}H_{n,m,\gamma}^{x}
+Z_{n,m,\gamma}^{x}.
$$

在全带宽 ZC 符号上，最小二乘（LS）信道估计为

$$
\hat H_{n,m_\mathrm{sync},\gamma}^{x,\mathrm{LS}}
=\frac{Y_{n,m_\mathrm{sync},\gamma}^{x}}{z_n^x}.
$$

其时延域表示为

$$
\hat h_\gamma^x[k]
=\frac{1}{N}\sum_{n=0}^{N-1}
\hat H_{n,m_\mathrm{sync},\gamma}^{x,\mathrm{LS}}
e^{j2\pi nk/N}.
$$

主峰位置用于更新时偏；循环前缀范围外的能量主要用于估计噪声。为降低 LS 噪声，可在时延域保留有效信道支撑区、将其余抽头置零，并使用 Wiener 权重

$$
w_\mathrm{W}=\frac{\widehat{\mathrm{SNR}}}{\widehat{\mathrm{SNR}}+1}
$$

平滑有效抽头，再变换回频域得到 $\hat H_{n,m_\mathrm{sync},\gamma}^{x}$。

## 4. 导频联合跟踪 CFO 与 SFO

在连续且都含相同已知导频的相邻有效符号上，形成跨符号相关并沿慢时间平均：

$$
\bar R_\gamma^x[n]
=\frac{1}{|\mathcal A_x|}
\sum_{m\in\mathcal A_x}
\left(Y_{n,m,\gamma}^{x}\right)^*
Y_{n,m+1,\gamma}^{x},
\qquad n\in\mathcal P.
$$

$\mathcal A_x$ 只包含两个相邻符号都属于链路 $x$ 的索引，因此 TDD 的保护区和上下行边界不会进入相位拟合。对导频相位解缠后，

$$
\varphi_\gamma^x[n]
=\arg\bar R_\gamma^x[n]
\approx
2\pi\left(f_{o,\gamma}^xT_O
-\kappa_n\Delta fN_s\Delta T_{s,\gamma}^x\right).
$$

以 $|\bar R_\gamma^x[n]|^2$ 为权重做线性回归：

$$
(\hat a,\hat b)
=\arg\min_{a,b}
\sum_{n\in\mathcal P}
|\bar R_\gamma^x[n]|^2
\left|\operatorname{unwrap}(\varphi_\gamma^x[n])-a-b\kappa_n\right|^2.
$$

于是

$$
\hat f_{o,\gamma}^x=\frac{\hat a}{2\pi T_O},
\qquad
\Delta\hat T_{s,\gamma}^x
=-\frac{\hat b}{2\pi\Delta fN_s}.
$$

## 5. 帧内相位与长期定时更新

若没有新的全带宽信道参考，初始信道可传播到第 $m$ 个符号：

$$
\hat H_{n,m,\gamma}^{x}
=\hat H_{n,m_\mathrm{sync},\gamma}^{x}
e^{j2\pi(m-m_\mathrm{sync})
(\hat f_{o,\gamma}^xT_O
-\kappa_n\Delta fN_s\Delta\hat T_{s,\gamma}^x)}.
$$

通信链路只在累计时偏接近一个采样或威胁循环前缀裕量时更新整数帧起点，其余亚采样误差由频域相位吸收。双站感知的测量量本身是时延，不能接受这种阶梯式修正，因此还需要[连续感知时偏估计](/zh-cn/docs/signal-processing/bistatic-sensing/#连续感知时偏)。
