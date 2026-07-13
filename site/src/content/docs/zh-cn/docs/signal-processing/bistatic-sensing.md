---
title: 双站感知
description: UE 侧双站感知、符号重构、时序补偿和时延解释。
---

双站感知中，BS 发射，UE 观察波形经过环境传播后的结果。处理流程与单站感知类似，但 UE 需要额外解决两个问题：

- UE 事先不知道数据符号，因此需要在去除调制前重构数据资源。
- BS 和 UE 不共址，因此时序漂移会直接移动感知时延轴。

## 符号重构

对于数据子载波 $n\in\mathcal{D}$，UE 从均衡后的通信符号中硬判决重构 QPSK：

$$
\tilde{b}_{n,m,\gamma}
=
\frac{1}{\sqrt{2}}
\left(
\operatorname{sgn}(\operatorname{Re}\{\hat{b}_{n,m,\gamma}\})
+j\operatorname{sgn}(\operatorname{Im}\{\hat{b}_{n,m,\gamma}\})
\right)
$$

对于 ZC 同步符号和 pilot 子载波，发送符号已知，直接使用。

UE 侧双站 channel symbols 为：

$$
(\boldsymbol{F}_{\mathrm{UE},\gamma})_{n,m}
=
\frac{(\boldsymbol{B}_{\mathrm{UE},\gamma})_{n,m}}
{\tilde{b}_{n,m,\gamma}}
$$

完成相除后，UE 可复用与 BS 侧单站相同的时延-多普勒和微多普勒处理结构。

## 为什么通信时序不够

通信只要求时序足以避免符号间干扰。只要最大相对时延仍在循环前缀内，亚采样级漂移主要表现为信道相位项，通信接收通常仍可容忍。

感知不同：时延本身就是测量量。若直接复用通信链路的整数采样时序修正，时延轨迹会呈阶梯状。在时延-多普勒或微多普勒输出中，这些阶跃会表现为人工跳变或谱线断裂。

因此 UE 感知链路维护连续的感知时序估计，而不是只使用零散的通信时序修正。

## 分数时延估计

令 $k_{\max,\gamma}$ 为主导信道峰的时延 bin。定义峰值两侧的复比值：

$$
r_p[k]
\triangleq
\frac{\tilde{p}_{\mathrm{delay},\gamma}[k_{\max,\gamma}+k]}
{\tilde{p}_{\mathrm{delay},\gamma}[k_{\max,\gamma}]},
\quad
k\in\{-1,1\}
$$

Quinn 风格的分数估计给出两个候选：

$$
\hat{\delta}_{\tau,1}
=
\frac{r_p[1]}{r_p[1]-1},
\quad
\hat{\delta}_{\tau,-1}
=
\frac{r_p[-1]}{1-r_p[-1]}
$$

最终选择规则为：

$$
\hat{\delta}_{\tau}
=
\begin{cases}
\hat{\delta}_{\tau,1}, &
\hat{\delta}_{\tau,-1}>0\ \text{and}\ \hat{\delta}_{\tau,1}>0,\\
\hat{\delta}_{\tau,-1}, & \text{otherwise}.
\end{cases}
$$

整体时序偏移为：

$$
\hat{\tau}_{o,\gamma}
=
\frac{\hat{k}_{\tau,\gamma}}{B}
=
\frac{\hat{\delta}_\tau+k_{\max,\gamma}}{B}
$$

## SIO 窗口拟合

分数估计带噪声，因此 OpenISAC 进一步利用采样间隔偏移（SIO）导致的近似线性时延漂移。在从 $\gamma_w$ 开始、长度为 $\Gamma_W$ 的窗口内，累计通信跟踪已经施加的整数修正：

$$
A_{\gamma_w+\ell}
\triangleq
\sum_{i=0}^{\ell-1}
\hat{k}_{\mathrm{TO},\gamma_w+i},
\quad
\ell=0,\ldots,\Gamma_W-1
$$

将这些修正加回，重构连续时延轨迹：

$$
\tilde{k}_{\tau,\gamma_w+\ell}
\triangleq
\hat{k}_{\tau,\gamma_w+\ell}+A_{\gamma_w+\ell}
$$

然后用最小二乘拟合：

$$
\tilde{k}_{\tau,\gamma_w+\ell}
\approx
\epsilon_{\mathrm{SIO},w}\ell+k_{\tau,\gamma_w}
$$

斜率 $\hat{\epsilon}_{\mathrm{SIO},w}$ 表示每帧由 SIO 引起的时延漂移。

## 连续感知时序

连续感知时序估计递归更新为：

$$
\hat{k}^{\mathrm{sens}}_{\tau,\gamma}
=
\hat{k}^{\mathrm{sens}}_{\tau,\gamma-1}
\hat{\epsilon}_{\mathrm{SIO},w-1}
-
\hat{k}_{\mathrm{TO},\gamma-1}
\mu_\gamma e_\gamma
$$

其中

$$
e_\gamma
\triangleq
\hat{k}_{\tau,\gamma}
-
\hat{k}^{\mathrm{sens}}_{\tau,\gamma-1}
$$

是跟踪误差。稳定运行时反馈增益 $\mu_\gamma$ 保持较小；误差持续存在时可增大以加快收敛。

## 信道补偿

补偿后的双站感知符号为：

$$
(\tilde{\boldsymbol{F}}_{\mathrm{UE},\gamma})_{n,m}
=
(\boldsymbol{F}_{\mathrm{UE},\gamma})_{n,m}
e^{j2\pi n\Delta f
(\hat{k}_{\tau,\gamma}^{\mathrm{sens}}+mN_s\Delta\hat{T}_{as,w-1})}
$$

这些补偿后的符号可像单站 channel symbols 一样进入后续感知处理。在该 OTA 同步方案下，双站时延是相对 LoS 参考路径的；若需要绝对时延，需要已知 BS-UE 物理距离。

## 实际限制

OTA 方法在 LoS 路径或另一个稳定主导路径持续可见时最可靠。在丰富 NLoS 场景中，最强路径可能在多个散射体之间切换，从而使恢复的时延轨迹出现偏差或间歇不连续。这类场景可能需要更长平均、路径一致性检查或外部同步。
