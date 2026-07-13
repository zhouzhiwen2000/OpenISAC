---
title: UE 接收
description: UE 同步、CFO/SFO 跟踪、信道估计、均衡和 payload 恢复。
---

UE 接收链路是两状态有限状态机：`SYNC_SEARCH` 和 `NORMAL`。`SYNC_SEARCH` 查找帧时序并估计粗载波偏移；`NORMAL` 解调帧、跟踪残余时序/频率偏移并解码 payload。

![UE 接收处理流程](/images/FlowGraph_UE.png)

## ZC 时序度量

令 $N_s=N+N_\mathrm{CP}$ 为一个 OFDM 符号的采样数。在单 ZC 配置中，UE 将接收块与本地时域 ZC 参考 $s_\mathrm{ZC}[k]$ 做相关：

$$
\Lambda_\mathrm{ZC}[u]
=
\frac{
\left|\sum_{k=0}^{N_s-1}y_\mathrm{UE}[u+k]s_\mathrm{ZC}^{*}[k]\right|^2
}{
\left(\sum_{k=0}^{N_s-1}|y_\mathrm{UE}[u+k]|^2\right)
\left(\sum_{k=0}^{N_s-1}|s_\mathrm{ZC}[k]|^2\right)
}
$$

实现中使用峰均比：

$$
\rho_\mathrm{ZC}
=
\frac{\Lambda_\mathrm{ZC}[\hat{k}_\mathrm{peak}]}
{\frac{1}{|\mathcal{U}|}\sum_{u\in\mathcal{U}}\Lambda_\mathrm{ZC}[u]}
$$

与阈值比较。若通过检测，初始时序估计为：

$$
\hat{k}_\mathrm{TO}
=
\hat{k}_\mathrm{peak}-m_\mathrm{sync}N_s-N_\mathrm{lag}
$$

$N_\mathrm{lag}$ 用于给早于最强路径到达的多径分量留出余量。

## 可选 CFO 捕获

对于较大的初始 CFO，可选第二个 ZC 符号支持 Schmidl-Cox 风格度量：

$$
P_\mathrm{SC}[u]
=
\sum_{k=0}^{N-1}
y_\mathrm{UE}^{*}[u+N_\mathrm{CP}+k]
y_\mathrm{UE}[u+N_s+N_\mathrm{CP}+k]
$$

$$
R_\mathrm{SC}[u]
=
\sum_{k=0}^{N-1}|y_\mathrm{UE}[u+N_s+N_\mathrm{CP}+k]|^2,
\quad
\Lambda_\mathrm{SC}[u]=\frac{|P_\mathrm{SC}[u]|^2}{R_\mathrm{SC}^2[u]}
$$

模糊 CFO 估计为：

$$
\hat{f}_{o,\mathrm{mod}}
=
\frac{\angle P_\mathrm{SC}[\hat{u}]}{2\pi N_sT_s}
$$

接收端随后用本地 ZC 度量测试 CFO 候选，并通过 CP-tail correlation 或可选 CFO training 字段细化频偏估计。

## NORMAL 状态信道模型

完成时序/频率补偿后，第 $\gamma$ 帧经 FFT 解调为：

$$
(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m}
=
b_{n,m,\gamma}(\boldsymbol{H}_{\mathrm{UE},\gamma})_{n,m}
+(\boldsymbol{Z}_{\mathrm{UE},\gamma})_{n,m}
$$

其中：

$$
(\boldsymbol{H}_{\mathrm{UE},\gamma})_{n,m}
=
\sum_{l=1}^{L}
\alpha_l
e^{j2\pi\left((f_{D,l}+\Delta\bar{f}_{c,\gamma})mT_O
-n\Delta f(\tau_l+\bar{\tau}_{d,\gamma,mN_s})\right)}
$$

全带宽 ZC 符号给出初始信道估计：

$$
(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_\mathrm{sync}}
=
\frac{(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m_\mathrm{sync}}}{z_n}
$$

其时延谱为：

$$
\tilde{p}_{\mathrm{delay},\gamma}[k]
=
\frac{1}{\sqrt{N}}
\sum_{n=0}^{N-1}
(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_\mathrm{sync}}
e^{j2\pi nk/N}
$$

$|\tilde{p}_{\mathrm{delay},\gamma}[k]|^2$ 用于时延峰跟踪。

## CFO/SFO 跟踪

对 pilot 子载波 $n\in\mathcal{P}$，接收端形成跨符号自相关：

$$
(\boldsymbol{R}_{\mathrm{UE},\gamma})_{n,m}
=
(\boldsymbol{B}_{\mathrm{UE},\gamma}^{*})_{n,m}
(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m+1}
$$

跨 OFDM 符号平均后：

$$
\bar{R}_{\mathrm{UE},\gamma}[n]
=
\frac{1}{M-1}
\sum_{m=0}^{M-2}
(\boldsymbol{R}_{\mathrm{UE},\gamma})_{n,m}
$$

pilot 相位近似为子载波索引的线性函数：

$$
\varphi_{\mathrm{UE},\gamma}[n]
=
\arg(\bar{R}_{\mathrm{UE},\gamma}[n])
\approx
2\pi(f_{o,\gamma}T_O-n\Delta f\,N_s\Delta T_{s,\gamma})
$$

对 pilot 相位做加权线性回归可估计：

$$
\hat{\boldsymbol{\theta}}_\gamma
=
\begin{bmatrix}
\hat{f}_{o,\gamma}\\
\Delta\hat{T}_{s,\gamma}
\end{bmatrix}
$$

随后从 ZC 符号传播得到完整帧的信道估计：

$$
(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m}
=
(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m_\mathrm{sync}}
e^{j2\pi(m-m_\mathrm{sync})(\hat{f}_{o,\gamma}T_O-n\Delta fN_s\Delta\hat{T}_{s,\gamma})}
$$

最后用单抽头频域均衡得到数据符号：

$$
\hat{b}_{n,m,\gamma}
=
\frac{(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m}}
{(\hat{\boldsymbol{H}}_{\mathrm{UE},\gamma})_{n,m}}
$$

均衡后的符号进入 LLR 计算、解扰和 LDPC 解码。
