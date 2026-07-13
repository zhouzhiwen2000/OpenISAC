---
title: OTA 与 eRTM 定时
description: 从 LoS 相对定时到上下行联合绝对时偏估计，以及补偿对杂波抑制的作用。
---

双站感知需要区分两类问题：

- **OTA LoS 跟踪**以稳定直达径为参考，去除 UE 相对于 BS 的连续漂移，结果是相对 LoS 的超额时延。
- **eRTM 双向定时**联合上行与下行信道时延，并引入固定群时延与收发时刻关系，将总偏移进一步分解为 BS 侧和 UE 侧时偏。

前者只依赖下行观测，适合保持时延轨迹连续；后者需要当前系统提供的双向链路，适合构造绝对时延参考。

## 1. 上下行时延谱关联

由 UE 观测到的下行信道和 BS 观测到的上行信道分别形成 $L_\mathrm{os}$ 倍过采样时延谱：

$$
p_\mathrm{DL}[k]
=\operatorname{IFFT}_{L_\mathrm{os}N}
\{\hat H_\mathrm{DL}[n]\},
\qquad
p_\mathrm{UL}[k]
=\operatorname{IFFT}_{L_\mathrm{os}N}
\{\hat H_\mathrm{UL}[n]\}.
$$

对两个复时延谱做循环互相关

$$
C[q]
=\sum_k p_\mathrm{UL}[k]
p_\mathrm{DL}^*[k-q].
$$

设 $\hat q$ 为 $|C[q]|$ 的峰值，$\hat\delta_q$ 为峰值邻域的分数 bin 修正，则定义上下行观测之间的相对时偏为

$$
\hat\tau_\mathrm{TO,BS-UE}
=\frac{\hat q+\hat\delta_q}{L_\mathrm{os}}
\quad\text{samples}.
$$

互相关利用完整多径轮廓，而不是只比较两个最大峰，因此在两条链路具有相似时延结构时通常比单峰相减更稳定。这里的正负号由上述互相关顺序固定；交换 $p_\mathrm{DL}$ 与 $p_\mathrm{UL}$ 会使估计符号反转。

## 2. eRTM 定时方程

以下所有量统一采用采样数；除以 $B$ 即得到秒。定义

- $\tau_\mathrm{DL,RF}$、$\tau_\mathrm{UL,RF}$：下行与上行中可校准的固定群时延；
- $t_\mathrm{DL-UL,BS}$：BS 参考下行与上行观测之间的定时间隔；
- $t_\mathrm{TA,UE}$：UE 对上行施加的定时提前量；
- $\tau_\mathrm{TO,BS-UE}$：由上下行时延谱关联得到的差分时偏。

先构造总定时约束

$$
\tau_c
=\tau_\mathrm{DL,RF}
+\tau_\mathrm{UL,RF}
-t_\mathrm{DL-UL,BS}
-t_\mathrm{TA,UE}.
$$

BS 侧与 UE 侧时偏满足

$$
\tau_\mathrm{TO,BS}+\tau_\mathrm{TO,UE}=\tau_c,
$$

$$
\tau_\mathrm{TO,BS}-\tau_\mathrm{TO,UE}
=\tau_\mathrm{TO,BS-UE}.
$$

解得

$$
\boxed{
\tau_\mathrm{TO,UE}
=\frac{\tau_c-\tau_\mathrm{TO,BS-UE}}{2}
},
$$

$$
\boxed{
\tau_\mathrm{TO,BS}
=\frac{\tau_c+\tau_\mathrm{TO,BS-UE}}{2}
}.
$$

$\tau_c$ 给出两侧时偏之和，双向信道关联给出两侧时偏之差，因此两者必须同时可用才能唯一分解。固定群时延的校准误差会形成共同偏差，而时延谱相关误差主要影响两侧估计的差值。

## 3. 从时偏估计到频域补偿

若 $\hat\tau_\mathrm{TO}$ 以采样为单位，正时延在第 $n$ 个子载波上产生相位 $-2\pi\kappa_n\Delta f\hat\tau_\mathrm{TO}/B$。去除该时延可写为

$$
H_\mathrm{corr}[n]
=H[n]
\exp\!\left(
j2\pi\kappa_n\Delta f
\frac{\hat\tau_\mathrm{TO}}{B}
\right).
$$

因此必须区分“估计到的物理时偏”和“施加到频域数据上的相位修正”：二者描述同一现象，但符号相反。整数帧起点变化还会改变观测坐标，应像双站连续时偏递推那样把该变化加回，避免绝对时延输出出现人为阶跃。

## 4. 为什么定时补偿改善 MTI

对一个静态双站路径，去调制后的信道符号近似为

$$
F[n,m]
\approx
\alpha
e^{-j2\pi\kappa_n\Delta f(\tau+\tau_d[m])}.
$$

若用 two-pulse canceller 说明 MTI 的基本作用，

$$
\tilde F[n,m]=F[n,m]-F[n,m-1],
$$

则

$$
|\tilde F[n,m]|^2
=4|\alpha|^2
\sin^2\!\left[
\pi\kappa_n\Delta f
(\tau_d[m]-\tau_d[m-1])
\right].
$$

理想静态路径本应在相邻慢时间样本间完全抵消；残余定时漂移改变了跨子载波相位，使其泄漏到 MTI 输出。连续时偏与 SFO 补偿减小 $\tau_d[m]-\tau_d[m-1]$，因此能降低静态杂波残余。

可用 MTI suppression ratio（MSR）量化这一效果：

$$
\mathrm{MSR}
=\frac{
\sum_{m,n}|F[n,m]|^2
}{
\sum_{m,n}|\tilde F[n,m]|^2
}.
$$

在同一静态场景和同一 MTI 滤波器下，MSR 越高，说明定时补偿后静态分量越接近零多普勒并被更充分地抑制。该指标衡量的是长期相位稳定性，不应被解释为目标检测信噪比本身。

## 5. 适用边界

OTA LoS 跟踪要求参考路径长期可见；主导路径切换会把几何变化误当作时钟漂移。eRTM 还要求上下行时延谱在比较时保持足够的结构一致性，并要求固定群时延已校准。FDD 中载波不同，多径幅相不必互易，但只要主要时延结构仍可对应，时延谱关联仍可提供约束；若两个载波的可见路径集合差异过大，相关峰的可信度会下降。
